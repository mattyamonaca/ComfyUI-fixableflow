"""
Simple PSD Layer Stack Node (Frontend PSD Generation)

4つの画像（base, shade, highlight, lineart）を受け取り、
フロント側でPSD生成するためのレイヤーPNGとJSONを出力し、
合成画像も返すノード。

修正点（致命的3点）:
1) INPUT_TYPES にある highlight を prepare_layers の引数に追加（未定義参照の解消）
2) composite_images 内の typo highlightt_f を highlight_f に修正
3) highlight 合成で shade を参照していたミスを修正し、合成の流れを
   base -> shade -> highlight -> lineart の result 連鎖に修正
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
import folder_paths
from PIL import Image


class SimplePSDStackNode:
    """
    base/shade/highlight/lineart を保存してPSD生成用の情報(JSON)を作り、合成画像を返す
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),        # ベース画像（一番下）
                "shade": ("IMAGE",),       # 影レイヤー（中）
                "highlight": ("IMAGE",),   # ハイライト等のレイヤー（中〜上）
                "lineart": ("IMAGE",),     # 線画（一番上）
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "layered",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "prepare_layers"
    CATEGORY = "FixableFlow"
    OUTPUT_NODE = True

    def prepare_layers(self, base, shade, highlight, lineart, filename_prefix="layered"):
        """
        4つの画像をレイヤーとして保存し、フロント側PSD生成用JSONを出力する

        Args:
            base: ベース画像（下）
            shade: 影（中）
            highlight: ハイライト（中〜上）
            lineart: 線画（上）
            filename_prefix: 出力ファイル名プレフィックス

        Returns:
            合成画像 (IMAGE)
        """
        images_list = [base, shade, highlight, lineart]
        layer_names = ["base", "shade", "highlight", "lineart"]

        output_dir = folder_paths.get_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 画像サイズを取得（先頭のbaseから）
        first_image = images_list[0]
        img_sample = first_image[0] if first_image.shape[0] > 0 else first_image
        height = img_sample.shape[0]
        width = img_sample.shape[1]

        print(f"Preparing layers for frontend PSD generation, size: {width}x{height}")
        print(f"Layer order: base (bottom) → shade → highlight → lineart (top)")

        # 各レイヤーをPNGとして保存
        layer_info = []
        composite_layers = []

        for image_tensor, layer_name in zip(images_list, layer_names):
            # バッチの最初の画像を取得
            img = image_tensor[0] if image_tensor.shape[0] > 0 else image_tensor

            # tensor -> uint8 (0..255)
            img_np = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            composite_layers.append(img_np)

            # PNGとして保存（RGB/RGBA を安全に扱うため mode を明示）
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                pil_img = Image.fromarray(img_np, mode="RGBA")
            else:
                pil_img = Image.fromarray(img_np, mode="RGB")

            filename = f"{filename_prefix}_{timestamp}_{layer_name}.png"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath)

            layer_info.append({"name": layer_name, "filename": filename})
            print(f"  Layer saved: {filename}")

        # レイヤー情報をJSONとして保存（フロントが読み取る）
        info_filename = f"{filename_prefix}_{timestamp}_layers.json"
        info_file = os.path.join(output_dir, info_filename)
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prefix": filename_prefix,
                    "timestamp": timestamp,
                    "layers": layer_info,
                    "width": int(width),
                    "height": int(height),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 最新の info file 名をログとして保存（フロントが追跡）
        log_path = os.path.join(output_dir, "simple_psd_stack_info.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(info_filename)

        print(f"Layer info saved: {info_filename}")
        print("Frontend can now generate PSD from these layers")

        # 4枚を合成
        composite = self.composite_images(
            composite_layers[0],
            composite_layers[1],
            composite_layers[2],
            composite_layers[3],
        )

        # ComfyUI形式のテンソルに変換
        composite_tensor = torch.from_numpy(composite.astype(np.float32) / 255.0).unsqueeze(0)
        return (composite_tensor,)

    def composite_images(self, base, shade, highlight, lineart):
        """
        base -> shade -> highlight -> lineart の順で合成

        Args:
            base, shade, highlight, lineart:
                NumPy配列 (H,W,3 or 4), uint8

        Returns:
            合成画像（NumPy配列、uint8、RGB）
        """
        # float32 (0..1)
        base_f = base.astype(np.float32) / 255.0
        shade_f = shade.astype(np.float32) / 255.0
        highlight_f = highlight.astype(np.float32) / 255.0
        lineart_f = lineart.astype(np.float32) / 255.0

        # base を RGB に
        base_rgb = base_f[:, :, :3] if base_f.shape[2] == 4 else base_f
        result = base_rgb

        # --- shade 合成 ---
        if shade_f.shape[2] == 4:
            shade_rgb = shade_f[:, :, :3]
            shade_alpha = shade_f[:, :, 3:4]
            result = result * (1.0 - shade_alpha) + shade_rgb * shade_alpha
        else:
            # アルファが無い場合は「上書き」扱い（必要なら別ブレンドに変更）
            shade_rgb = shade_f[:, :, :3] if shade_f.shape[2] == 4 else shade_f
            result = shade_rgb

        # --- highlight 合成（修正: typo/参照/合成対象） ---
        if highlight_f.shape[2] == 4:
            highlight_rgb = highlight_f[:, :, :3]
            highlight_alpha = highlight_f[:, :, 3:4]
            result = result * (1.0 - highlight_alpha) + highlight_rgb * highlight_alpha
        else:
            highlight_rgb = highlight_f[:, :, :3] if highlight_f.shape[2] == 4 else highlight_f
            result = highlight_rgb

        # --- lineart 合成 ---
        if lineart_f.shape[2] == 4:
            lineart_rgb = lineart_f[:, :, :3]
            lineart_alpha = lineart_f[:, :, 3:4]
            result = result * (1.0 - lineart_alpha) + lineart_rgb * lineart_alpha
        else:
            # アルファ無し線画は乗算（白地/黒線前提）
            lineart_rgb = lineart_f[:, :, :3] if lineart_f.shape[2] == 4 else lineart_f
            result = result * lineart_rgb

        # uint8 へ
        out = (result * 255.0).clip(0, 255).astype(np.uint8)
        return out


NODE_CLASS_MAPPINGS = {
    "SimplePSDStackNode": SimplePSDStackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplePSDStackNode": "Simple PSD Stack"
}
