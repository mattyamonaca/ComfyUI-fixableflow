"""
Overlay Images Node
2つの画像を受け取り、input1の上にinput2を重ねてPNG画像として出力するノード
"""

import torch
import numpy as np
from PIL import Image
import os
import folder_paths


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
    CATEGORY = "image/composite"
    
    def overlay_images(self, input1, input2):
        """
        2つの画像を重ね合わせる
        
        Args:
            input1: 背景画像 (ComfyUI形式: torch.Tensor [B, H, W, C])
            input2: 前景画像 (ComfyUI形式: torch.Tensor [B, H, W, C])
        
        Returns:
            重ね合わせた画像 (ComfyUI形式)
        """
        # ComfyUI形式のテンソルをNumPy配列に変換
        # ComfyUIの画像は [B, H, W, C] の形式で、値は0-1の範囲
        img1_np = (input1[0].cpu().numpy() * 255).astype(np.uint8)
        img2_np = (input2[0].cpu().numpy() * 255).astype(np.uint8)
        
        # NumPy配列をPIL Imageに変換
        img1_pil = Image.fromarray(img1_np, mode='RGB')
        img2_pil = Image.fromarray(img2_np, mode='RGB')
        
        # 画像サイズを揃える（input1のサイズに合わせる）
        if img1_pil.size != img2_pil.size:
            img2_pil = img2_pil.resize(img1_pil.size, Image.Resampling.LANCZOS)
        
        # input2にアルファチャンネルがない場合は追加
        # RGBをRGBAに変換（完全不透明）
        if img2_pil.mode == 'RGB':
            img2_pil = img2_pil.convert('RGBA')
        
        # input1もRGBAに変換
        if img1_pil.mode == 'RGB':
            img1_pil = img1_pil.convert('RGBA')
        
        # 画像を重ね合わせる
        # alpha_compositeはimg1の上にimg2を重ねる
        result_pil = Image.alpha_composite(img1_pil, img2_pil)
        
        # RGBに戻す（PNGでもRGBとして出力）
        result_pil = result_pil.convert('RGB')
        
        # PIL ImageをNumPy配列に変換
        result_np = np.array(result_pil).astype(np.float32) / 255.0
        
        # ComfyUI形式のテンソルに変換 [B, H, W, C]
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    "OverlayImagesNode": OverlayImagesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayImagesNode": "Overlay Images"
}
