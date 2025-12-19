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
    CATEGORY = "image/composite"
    
    def overlay_images(self, input1, input2):
        """
        2つの画像を重ね合わせる
        input2をinput1の上に重ねる（input2が優先される）
        
        Args:
            input1: 背景画像 (ComfyUI形式: torch.Tensor [B, H, W, C])
            input2: 前景画像 (ComfyUI形式: torch.Tensor [B, H, W, C])
        
        Returns:
            重ね合わせた画像 (ComfyUI形式)
        """
        # バッチの最初の画像を取得
        img1 = input1[0].clone()
        img2 = input2[0].clone()
        
        # サイズを揃える（input1のサイズに合わせる）
        if img1.shape[:2] != img2.shape[:2]:
            # テンソルをNumPy配列に変換
            img2_np = (img2.cpu().numpy() * 255).astype(np.uint8)
            img2_pil = Image.fromarray(img2_np)
            
            # リサイズ (height, width)
            target_size = (img1.shape[1], img1.shape[0])  # (width, height) for PIL
            img2_pil = img2_pil.resize(target_size, Image.LANCZOS)
            
            # テンソルに戻す
            img2 = torch.from_numpy(np.array(img2_pil).astype(np.float32) / 255.0)
        
        # input2の黒以外の部分をマスクとして使用
        # 黒 (0,0,0) 以外のピクセルをinput2から採用
        # グレースケール変換して閾値判定
        gray = img2.mean(dim=2, keepdim=True)
        mask = (gray > 0.01).float()  # 黒に近い部分は背景を使う
        
        # マスクを使って合成
        result = img1 * (1 - mask) + img2 * mask
        
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
