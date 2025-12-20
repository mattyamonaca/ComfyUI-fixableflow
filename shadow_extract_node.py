"""
Shadow Extract Node (HSV Version)
影抽出ノード - HSV空間で影を検出してアルファチャンネルとして出力

- shade: 影あり画像（影の色を抽出したい画像）
- base: 影なし画像（フラット・ベース画像）
- 明度低下（V）と彩度上昇（S）にのみ反応
- 色相差は無視
- 出力: RGBA画像（RGB=shade側の色, Alpha=影量）
"""

import torch
import numpy as np
import cv2


class ShadowExtractNode:
    """
    2枚の画像から影を抽出してアルファチャンネルとして出力するノード
    shade側の色を保持し、影の部分のみを抽出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shade": ("IMAGE",),  # 影あり画像
                "base": ("IMAGE",),   # 影なし画像（フラット）
                "weight_V": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "weight_S": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "normalize_factor": ("FLOAT", {
                    "default": 40.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 1.0,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "extract_shadow"
    CATEGORY = "image/processing"
    
    def extract_shadow(self, shade, base, weight_V=1.0, weight_S=0.5, normalize_factor=40.0):
        """
        shade画像から影の色を抽出してRGBA画像として出力
        
        Args:
            shade: 影あり画像（この画像の色を使用）
            base: 影なし画像（フラット・比較用）
            weight_V: 明度差の重み
            weight_S: 彩度差の重み
            normalize_factor: 正規化係数
        
        Returns:
            RGBA画像（RGB=shade側の色、Alpha=影の量）
        """
        from PIL import Image
        
        # shade（影あり）とbase（影なし）を取得
        shade_img = shade[0].clone()
        base_img = base[0].clone()
        
        # アルファチャンネルを削除
        if shade_img.shape[2] == 4:
            shade_img = shade_img[:, :, :3]
        if base_img.shape[2] == 4:
            base_img = base_img[:, :, :3]
        
        # サイズを合わせる
        if shade_img.shape[:2] != base_img.shape[:2]:
            base_np = (base_img.cpu().numpy() * 255).astype(np.uint8)
            base_pil = Image.fromarray(base_np)
            target_size = (shade_img.shape[1], shade_img.shape[0])
            base_pil = base_pil.resize(target_size, Image.LANCZOS)
            base_img = torch.from_numpy(np.array(base_pil).astype(np.float32) / 255.0)
        
        # NumPy配列に変換
        shade_np = (shade_img.cpu().numpy() * 255).astype(np.uint8)
        base_np = (base_img.cpu().numpy() * 255).astype(np.uint8)
        
        # HSV色空間に変換
        hsv_shade = cv2.cvtColor(shade_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_base = cv2.cvtColor(base_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # H, S, Vチャンネルを分離
        _, S_shade, V_shade = cv2.split(hsv_shade)
        _, S_base, V_base = cv2.split(hsv_base)
        
        # 影の検出
        # delta_V: baseよりshadeが暗い部分（影で明度が下がっている）
        delta_V = np.maximum(0.0, V_base - V_shade)
        # delta_S: shadeの方が彩度が高い部分（影で彩度が上がる場合）
        delta_S = np.maximum(0.0, S_shade - S_base)
        
        # 影スコアの計算
        shadow_score = weight_V * delta_V + weight_S * delta_S
        
        # アルファチャンネルに変換（0-1の範囲）
        alpha = np.clip(shadow_score / normalize_factor, 0.0, 1.0)
        alpha = alpha[:, :, np.newaxis]
        
        # shade側のRGB + 影のアルファでRGBA画像を作成
        shade_float = shade_img.cpu().numpy()
        rgba = np.concatenate([shade_float, alpha], axis=2)
        
        # テンソルに変換
        result = torch.from_numpy(rgba.astype(np.float32))
        result = result.unsqueeze(0)
        
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ShadowExtractNode": ShadowExtractNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShadowExtractNode": "Shadow Extract (HSV)"
}
