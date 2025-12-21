"""
Overlay Images Node
2つの画像を受け取り、input1の上にinput2を重ねてPNG画像として出力するノード
"""

import torch
import numpy as np
from PIL import Image


class OverlayImagesNode:
    """
    2つの画像を重ね合わせるノード
    input1を背景として、input2を上に重ねます
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("IMAGE",),  # 背景画像
                "input2": ("IMAGE",),  # 前景画像
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "overlay_images"
    CATEGORY = "FixableFlow"
    
    def overlay_images(self, input1, input2):
        """
        2つの画像を重ね合わせる
        input2をinput1の上に重ねる（アルファブレンディング）
        
        Args:
            input1: 背景画像 (ComfyUI形式: torch.Tensor [B, H, W, C])
            input2: 前景画像 (ComfyUI形式: torch.Tensor [B, H, W, C])
        
        Returns:
            重ね合わせた画像 (ComfyUI形式)
        """
        # バッチの最初の画像を取得
        img1 = input1[0].clone()
        img2 = input2[0].clone()
        
        # RGBの場合はRGBAに変換（完全不透明のアルファチャンネルを追加）
        if img1.shape[2] == 3:
            alpha1 = torch.ones(img1.shape[0], img1.shape[1], 1, dtype=img1.dtype, device=img1.device)
            img1 = torch.cat([img1, alpha1], dim=2)
        
        if img2.shape[2] == 3:
            alpha2 = torch.ones(img2.shape[0], img2.shape[1], 1, dtype=img2.dtype, device=img2.device)
            img2 = torch.cat([img2, alpha2], dim=2)
        
        # サイズを揃える（input1のサイズに合わせる）
        if img1.shape[:2] != img2.shape[:2]:
            # テンソルをNumPy配列に変換
            img2_np = (img2.cpu().numpy() * 255).astype(np.uint8)
            img2_pil = Image.fromarray(img2_np, mode='RGBA')
            
            # リサイズ (height, width)
            target_size = (img1.shape[1], img1.shape[0])  # (width, height) for PIL
            img2_pil = img2_pil.resize(target_size, Image.LANCZOS)
            
            # テンソルに戻す
            img2 = torch.from_numpy(np.array(img2_pil).astype(np.float32) / 255.0)
        
        # アルファブレンディング
        # result_rgb = img1_rgb * (1 - alpha2) + img2_rgb * alpha2
        rgb1 = img1[:, :, :3]
        rgb2 = img2[:, :, :3]
        alpha2 = img2[:, :, 3:4]  # 前景のアルファチャンネル
        
        # アルファ合成
        result_rgb = rgb1 * (1 - alpha2) + rgb2 * alpha2
        
        # 結果のアルファチャンネルを計算（Porter-Duff合成）
        alpha1 = img1[:, :, 3:4]
        result_alpha = alpha2 + alpha1 * (1 - alpha2)
        
        # RGBAを結合
        result = torch.cat([result_rgb, result_alpha], dim=2)
        
        # バッチ次元を追加
        result = result.unsqueeze(0)
        
        return (result,)


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    "OverlayImagesNode": OverlayImagesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayImagesNode": "Overlay Images"
}
