"""
highlight Extract (HSV + Deadzone/Headroom Normalize + Sigmoid)

※ロジックは一切変更せず、ノードの入力名（UIに出るラベル）だけ分かりやすくしました。
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from PIL import Image


def _img_to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    """
    ComfyUI IMAGE tensor (H,W,C) float[0..1] -> uint8 RGB ndarray (H,W,3)
    """
    if img.ndim != 3:
        raise ValueError(f"Expected (H,W,C) tensor, got shape={tuple(img.shape)}")

    # strip alpha if exists
    if img.shape[2] >= 4:
        img = img[:, :, :3]

    arr = img.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def _resize_uint8_rgb(np_img: np.ndarray, target_w: int, target_h: int, resample: int) -> np.ndarray:
    """
    uint8 RGB ndarray -> resized uint8 RGB ndarray
    """
    if np_img.shape[0] == target_h and np_img.shape[1] == target_w:
        return np_img
    pil = Image.fromarray(np_img, mode="RGB")
    pil = pil.resize((target_w, target_h), resample=resample)
    return np.array(pil, dtype=np.uint8)


def _resample_mode(name: str) -> int:
    name = (name or "").lower()
    if name == "nearest":
        return Image.NEAREST
    if name == "bilinear":
        return Image.BILINEAR
    if name == "bicubic":
        return Image.BICUBIC
    # default: lanczos
    return Image.LANCZOS


class HighlightExtractNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "highlight_image": ("IMAGE",),  # 影あり（RGB保持）
                "base_image": ("IMAGE",),    # 影なし（比較用）

                # --- deadzone params ---
                "min_v_diff_threshold": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "extra_threshold_near_white": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "white_strictness_power": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                }),

                # --- headroom normalize ---
                "headroom_divisor_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 1.0,
                    "display": "slider",
                }),

                # --- sigmoid ---
                "sigmoid_start_point": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "sigmoid_steepness": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider",
                }),

                # --- alpha cutoff ---
                "alpha_cutoff_threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                }),

                # --- misc ---
                "swap_inputs": ("BOOLEAN", {"default": False}),
                "resize_base_to_highlight": ("BOOLEAN", {"default": True}),
                "resize_resample": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "extract"
    CATEGORY = "FixableFlow"

    def extract(
        self,
        highlight_image,
        base_image,
        min_v_diff_threshold=2.0,
        extra_threshold_near_white=5.0,
        white_strictness_power=2.0,
        headroom_divisor_min=0.0,
        sigmoid_start_point=0.25,
        sigmoid_steepness=12.0,
        alpha_cutoff_threshold=128,
        swap_inputs=False,
        resize_base_to_highlight=True,
        resize_resample="lanczos",
    ):
        """
        Returns:
            (IMAGE,) RGBA
        """
        if not isinstance(highlight_image, torch.Tensor) or not isinstance(base_image, torch.Tensor):
            raise ValueError("Inputs must be ComfyUI IMAGE tensors.")

        if highlight_image.ndim != 4 or base_image.ndim != 4:
            raise ValueError(f"Expected (B,H,W,C). Got image1={tuple(highlight_image.shape)}, image2={tuple(base_image.shape)}")

        b1 = highlight_image.shape[0]
        b2 = base_image.shape[0]
        b = max(b1, b2)

        # broadcast: if one is batch=1 and other >1, reuse index 0
        def pick(t: torch.Tensor, i: int) -> torch.Tensor:
            if t.shape[0] == 1:
                return t[0]
            return t[i]

        resample = _resample_mode(resize_resample)

        out_list = []
        for i in range(b):
            img1_t = pick(highlight_image, i)
            img2_t = pick(base_image, i)

            if swap_inputs:
                img1_t, img2_t = img2_t, img1_t

            img1_u8 = _img_to_uint8_rgb(img1_t)  # RGB保持側
            img2_u8 = _img_to_uint8_rgb(img2_t)  # 比較側

            # サイズ調整
            h1, w1 = img1_u8.shape[:2]
            h2, w2 = img2_u8.shape[:2]
            if (h1 != h2) or (w1 != w2):
                if resize_base_to_highlight:
                    img2_u8 = _resize_uint8_rgb(img2_u8, w1, h1, resample=resample)
                else:
                    # 逆に highlight を base に合わせる
                    img1_u8 = _resize_uint8_rgb(img1_u8, w2, h2, resample=resample)
                    h1, w1 = img1_u8.shape[:2]

            # RGB -> HSV (OpenCV: H 0-180, S/V 0-255)
            hsv1 = cv2.cvtColor(img1_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv2 = cv2.cvtColor(img2_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

            _, S1, V1 = cv2.split(hsv1)
            _, S2, V2 = cv2.split(hsv2)

            # --- 元スクリプト通り ---
            # 明度差（光方向）: delta_V = max(0, V1 - V2)
            delta_V = np.maximum(0.0, V1 - V2)

            Vbase = V2  # 下地(影なし)の明度を基準にする

            # 1) 白に近いほど除去を強めるデッドゾーン
            v = Vbase / 255.0
            thr = float(min_v_diff_threshold) + float(extra_threshold_near_white) * (v ** float(white_strictness_power))
            delta_V2 = np.maximum(0.0, delta_V - thr)

            # 2) headroom 正規化（下限付き）
            headroom = 255.0 - Vbase
            den = np.maximum(headroom, float(headroom_divisor_min))
            rel_V = delta_V2 / den
            light_score = np.clip(rel_V, 0.0, 1.0)

            # --- Sigmoid ---
            z = float(sigmoid_steepness) * (light_score - float(sigmoid_start_point))
            z = np.clip(z, -60.0, 60.0)
            sig = 1.0 / (1.0 + np.exp(-z))

            # 0入力で0にしたい補正
            z0 = float(sigmoid_steepness) * (0.0 - float(sigmoid_start_point))
            z0 = np.clip(z0, -60.0, 60.0)
            sig0 = 1.0 / (1.0 + np.exp(-z0))

            alpha_f = (sig - sig0) / (1.0 - sig0)
            alpha_f = np.clip(alpha_f, 0.0, 1.0)

            alpha_u8 = (alpha_f * 255.0).round().astype(np.uint8)
            cutoff = int(alpha_cutoff_threshold)
            if cutoff > 0:
                alpha_u8[alpha_u8 <= cutoff] = 0

            # RGBA (RGB=highlight_image)
            rgba_u8 = np.dstack([img1_u8, alpha_u8])
            rgba_f = rgba_u8.astype(np.float32) / 255.0  # (H,W,4) float
            out_list.append(rgba_f)

        out_np = np.stack(out_list, axis=0)  # (B,H,W,4)
        out = torch.from_numpy(out_np)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "HighlightExtractNode": HighlightExtractNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighlightExtractNode": "HighlightExtractNode"
}
