"""
Morphology Operations Node for ComfyUI
モルフォロジー演算（膨張・収縮）によるノイズ除去ノード
"""

import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths
import os

# パス設定
comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = f'{comfy_path}/custom_nodes/ComfyUI-fixableflow'
output_dir = f"{custom_nodes_path}/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def tensor_to_pil(tensor):
    """ComfyUIのテンソル形式をPIL Imageに変換"""
    image_np = tensor.cpu().detach().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # バッチの最初の画像を取得
    image_np = (image_np * 255).astype(np.uint8)
    
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 3:
            mode = 'RGB'
        elif image_np.shape[2] == 4:
            mode = 'RGBA'
        else:
            mode = 'L'
    else:
        mode = 'L'
    
    return Image.fromarray(image_np, mode=mode)


def pil_to_tensor(image):
    """PIL ImageをComfyUIのテンソル形式に変換"""
    image_np = np.array(image).astype(np.float32) / 255.0
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=2)
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)


def apply_morphology_operations(image, operation_type="close", kernel_size=3, iterations=1, 
                               kernel_shape="ellipse", binary_threshold=127):
    """
    モルフォロジー演算を適用
    
    Args:
        image: PIL Image
        operation_type: 演算タイプ ("close", "open", "dilate", "erode", "gradient", "tophat", "blackhat")
        kernel_size: カーネルサイズ
        iterations: 繰り返し回数
        kernel_shape: カーネル形状 ("ellipse", "rectangle", "cross")
        binary_threshold: 二値化の閾値
    
    Returns:
        処理済みのPIL Image
    """
    # 画像をnumpy配列に変換
    image_np = np.array(image)
    
    # グレースケールに変換
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # カーネル形状の選択
    if kernel_shape == "rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:  # ellipse
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # モルフォロジー演算の適用
    if operation_type == "close":
        # クロージング（膨張→収縮）: 小さな穴や切れ目を埋める
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation_type == "open":
        # オープニング（収縮→膨張）: 小さな突起やノイズを除去
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation_type == "dilate":
        # 膨張: 白い領域を拡大
        result = cv2.dilate(gray, kernel, iterations=iterations)
    elif operation_type == "erode":
        # 収縮: 白い領域を縮小
        result = cv2.erode(gray, kernel, iterations=iterations)
    elif operation_type == "gradient":
        # モルフォロジー勾配: エッジ検出
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif operation_type == "tophat":
        # トップハット: 明るい小さな領域の抽出
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif operation_type == "blackhat":
        # ブラックハット: 暗い小さな領域の抽出
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    else:
        result = gray
    
    # 元の画像がカラーの場合は3チャンネルに戻す
    if len(image_np.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(result)


def apply_advanced_noise_removal(image, min_contour_area=10, close_kernel_size=3, 
                                open_kernel_size=3, binary_threshold=127):
    """
    高度なノイズ除去処理
    
    Args:
        image: PIL Image
        min_contour_area: 保持する最小輪郭面積
        close_kernel_size: クロージングのカーネルサイズ
        open_kernel_size: オープニングのカーネルサイズ
        binary_threshold: 二値化の閾値
    
    Returns:
        処理済みのPIL Image
    """
    # 画像をnumpy配列に変換
    image_np = np.array(image)
    
    # グレースケールに変換
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # 二値化
    _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Step 1: クロージング処理で線の断片を接続
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (close_kernel_size, close_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    
    # Step 2: 小さなノイズを除去（輪郭解析）
    if min_contour_area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 面積でフィルタリング
        mask = np.zeros_like(binary)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        binary = mask
    
    # Step 3: オープニング処理で小さな突起を除去
    if open_kernel_size > 0:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (open_kernel_size, open_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    
    # 元の画像がカラーの場合は3チャンネルに戻す
    if len(image_np.shape) == 3:
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    else:
        result = binary
    
    return Image.fromarray(result)


class MorphologyNode:
    """
    モルフォロジー演算ノード
    線画の断片接続やノイズ除去に使用
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": (["close", "open", "dilate", "erode", 
                             "gradient", "tophat", "blackhat"],),
                "kernel_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 21,
                    "step": 2,
                    "display": "slider",
                    "display_label": "Kernel Size"
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Iterations"
                }),
                "kernel_shape": (["ellipse", "rectangle", "cross"],),
                "binary_threshold": ("INT", {
                    "default": 127,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Binary Threshold"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("processed", "comparison")
    
    FUNCTION = "execute"
    
    CATEGORY = "FixableFlow"
    
    def execute(self, image, operation="close", kernel_size=3, iterations=1,
                kernel_shape="ellipse", binary_threshold=127):
        """
        モルフォロジー演算を実行
        """
        # 画像をPIL Imageに変換
        image_pil = tensor_to_pil(image)
        
        # モルフォロジー演算を適用
        result = apply_morphology_operations(
            image_pil, 
            operation,
            kernel_size,
            iterations,
            kernel_shape,
            binary_threshold
        )
        
        # 比較画像を作成
        comparison = create_comparison_image(image_pil, result)
        
        # テンソルに変換
        result_tensor = pil_to_tensor(result)
        comparison_tensor = pil_to_tensor(comparison)
        
        return (result_tensor, comparison_tensor)

# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "MorphologyOperation": MorphologyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorphologyOperation": "Morphology Operation",
}
