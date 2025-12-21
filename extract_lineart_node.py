"""
Extract Line Art Node for ComfyUI
線画の背景を透過させるノード
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
fixableflow_path = f'{comfy_path}/custom_nodes/ComfyUI-FixableFlow'
output_dir = f"{fixableflow_path}/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def convert_non_white_to_black(image):
    """
    白に近い色を完全な白に、それ以外を保持
    """
    image_np = np.array(image)
    # 200より大きい値（白に近い）は完全に白(255)にする
    image_np[image_np > 200] = 255
    return Image.fromarray(image_np)


def extract_lineart(image):
    """
    線画から背景を透過させる処理
    入力: RGBまたはRGBA画像
    出力: RGBA画像（線画部分は黒、背景は透明）
    """
    # グレースケールに変換
    image_gray = image.convert('L')
    
    # 白に近い部分を完全に白にする
    image_gray = convert_non_white_to_black(image_gray)
    
    # スムージングフィルタを適用
    image_gray = image_gray.filter(ImageFilter.SMOOTH)
    
    # RGBA画像を作成
    result = Image.new('RGBA', image.size)
    
    # ピクセル単位で処理（NumPyで高速化）
    gray_array = np.array(image_gray)
    result_array = np.zeros((*gray_array.shape, 4), dtype=np.uint8)
    
    # グレー値に基づいてアルファ値を設定
    # 黒（0）は不透明（alpha=255）、白（255）は透明（alpha=0）
    alpha_array = 255 - gray_array
    
    # 黒い線画として設定（R=0, G=0, B=0）
    result_array[:, :, 0] = 0  # R
    result_array[:, :, 1] = 0  # G
    result_array[:, :, 2] = 0  # B
    result_array[:, :, 3] = alpha_array  # Alpha
    
    result = Image.fromarray(result_array, mode='RGBA')
    
    return result


def tensor_to_pil(tensor):
    """
    ComfyUIのテンソル形式をPIL Imageに変換
    """
    # tensor shape: [batch, height, width, channels]
    image_np = tensor.cpu().detach().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # バッチの最初の画像を取得
    image_np = (image_np * 255).astype(np.uint8)
    
    # チャンネル数に応じてモードを選択
    if image_np.shape[2] == 3:
        mode = 'RGB'
    elif image_np.shape[2] == 4:
        mode = 'RGBA'
    else:
        mode = 'L'
    
    return Image.fromarray(image_np, mode=mode)


def pil_to_tensor(image):
    """
    PIL ImageをComfyUIのテンソル形式に変換
    """
    # RGBAをnumpy配列に変換
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # バッチ次元を追加
    image_np = np.expand_dims(image_np, axis=0)
    
    # テンソルに変換
    return torch.from_numpy(image_np)


class ExtractLineArtNode:
    """
    線画の背景を透過させるノード
    RGB線画を入力として受け取り、背景を透過したRGBA画像を出力
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "white_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "display_label": "White Threshold"
                }),
                "apply_smoothing": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Apply Smoothing"
                }),
                "invert_alpha": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Invert Alpha"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "FixableFlow"
    
    def execute(self, image, white_threshold=200, apply_smoothing=True, invert_alpha=False):
        """
        線画の背景透過処理を実行
        
        Args:
            image: ComfyUIの画像テンソル
            white_threshold: 白と判定する閾値
            apply_smoothing: スムージングフィルタを適用するか
            invert_alpha: アルファチャンネルを反転するか（白い部分を不透明にする）
        
        Returns:
            image: 背景透過済みのRGBA画像
            alpha_mask: アルファチャンネルのマスク
        """
        
        # テンソルをPIL Imageに変換
        pil_image = tensor_to_pil(image)
        
        # グレースケールに変換
        image_gray = pil_image.convert('L')
        
        # 白の閾値処理
        gray_array = np.array(image_gray)
        gray_array[gray_array > white_threshold] = 255
        image_gray = Image.fromarray(gray_array)
        
        # スムージング適用
        if apply_smoothing:
            image_gray = image_gray.filter(ImageFilter.SMOOTH)
        
        # RGBA画像を作成
        gray_array = np.array(image_gray)
        result_array = np.zeros((*gray_array.shape, 4), dtype=np.uint8)
        
        # アルファ値の設定
        if invert_alpha:
            # 白い部分を不透明に（通常の逆）
            alpha_array = gray_array
        else:
            # 黒い部分を不透明に（通常の線画処理）
            alpha_array = 255 - gray_array
        
        # 線画の色設定（黒）
        result_array[:, :, 0] = 0  # R
        result_array[:, :, 1] = 0  # G  
        result_array[:, :, 2] = 0  # B
        result_array[:, :, 3] = alpha_array  # Alpha
        
        # PIL Imageに変換
        result_image = Image.fromarray(result_array, mode='RGBA')
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(result_image)
        
        # アルファマスクを別途作成（グレースケール）
        alpha_mask = alpha_array.astype(np.float32) / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=0)  # バッチ次元を追加
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)  # チャンネル次元を追加
        alpha_mask_tensor = torch.from_numpy(alpha_mask)
        
        return (output_tensor, alpha_mask_tensor)


class ExtractLineArtAdvancedNode:
    """
    高度な線画抽出ノード
    色付き線画にも対応し、線の色を保持したまま背景を透過
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "white_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "display_label": "White Threshold"
                }),
                "apply_smoothing": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Apply Smoothing"
                }),
                "preserve_colors": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Preserve Line Colors"
                }),
                "line_darkness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "display_label": "Line Darkness"
                }),
                "edge_detection": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Use Edge Detection"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "preview", "alpha_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "FixableFlow"
    
    def execute(self, image, white_threshold=200, apply_smoothing=True, 
                preserve_colors=False, line_darkness=1.0, edge_detection=False):
        """
        高度な線画抽出処理
        
        Args:
            image: 入力画像
            white_threshold: 白判定の閾値
            apply_smoothing: スムージング適用
            preserve_colors: 線の色を保持
            line_darkness: 線の濃さ調整
            edge_detection: エッジ検出を使用
        
        Returns:
            rgba_image: 背景透過済み画像
            preview: プレビュー画像（チェッカーボード背景）
            alpha_mask: アルファマスク
        """
        
        # テンソルをPIL Imageに変換
        pil_image = tensor_to_pil(image)
        original_image = pil_image.copy()
        
        # グレースケール版を作成（アルファチャンネル計算用）
        image_gray = pil_image.convert('L')
        
        if edge_detection:
            # エッジ検出を使用する場合
            import cv2
            gray_np = np.array(image_gray)
            edges = cv2.Canny(gray_np, 50, 150)
            gray_array = 255 - edges  # エッジを黒に
        else:
            # 通常の閾値処理
            gray_array = np.array(image_gray)
            gray_array[gray_array > white_threshold] = 255
        
        image_gray = Image.fromarray(gray_array)
        
        # スムージング適用
        if apply_smoothing:
            image_gray = image_gray.filter(ImageFilter.SMOOTH)
        
        # アルファ値の計算
        gray_array = np.array(image_gray)
        alpha_array = 255 - gray_array
        
        # 線の濃さ調整
        if line_darkness != 1.0:
            alpha_array = np.clip(alpha_array * line_darkness, 0, 255).astype(np.uint8)
        
        # RGBA画像を作成
        result_array = np.zeros((*gray_array.shape, 4), dtype=np.uint8)
        
        if preserve_colors:
            # 元の色を保持
            original_array = np.array(original_image.convert('RGB'))
            result_array[:, :, :3] = original_array
        else:
            # 黒い線画
            result_array[:, :, 0] = 0  # R
            result_array[:, :, 1] = 0  # G
            result_array[:, :, 2] = 0  # B
        
        result_array[:, :, 3] = alpha_array  # Alpha
        
        # PIL Imageに変換
        result_image = Image.fromarray(result_array, mode='RGBA')
        
        # プレビュー画像の作成（チェッカーボード背景）
        preview = create_preview_with_checkerboard(result_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(result_image)
        preview_tensor = pil_to_tensor(preview)
        
        # アルファマスクを作成
        alpha_mask = alpha_array.astype(np.float32) / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=0)
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)
        alpha_mask_tensor = torch.from_numpy(alpha_mask)
        
        return (output_tensor, preview_tensor, alpha_mask_tensor)


def create_preview_with_checkerboard(rgba_image, tile_size=10):
    """
    チェッカーボード背景付きのプレビュー画像を作成
    """
    width, height = rgba_image.size
    
    # チェッカーボード背景を作成
    checkerboard = Image.new('RGB', (width, height))
    pixels = checkerboard.load()
    
    for y in range(height):
        for x in range(width):
            if (x // tile_size + y // tile_size) % 2 == 0:
                pixels[x, y] = (255, 255, 255)
            else:
                pixels[x, y] = (200, 200, 200)
    
    # RGBA画像をチェッカーボードの上に合成
    checkerboard.paste(rgba_image, (0, 0), rgba_image)
    
    return checkerboard.convert('RGBA')


# ノードクラスのマッピングを更新
NODE_CLASS_MAPPINGS = {
    "Extract Line Art": ExtractLineArtNode,
    "Extract Line Art Advanced": ExtractLineArtAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract Line Art": "Extract Line Art",
    "Extract Line Art Advanced": "Extract Line Art (Advanced)",
}
