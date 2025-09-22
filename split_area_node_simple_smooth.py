"""
Split Area Node with Simple Smoothing - 簡易スムージング版
領域検出の改善版（デバッグ済み）
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label, binary_fill_holes
import cv2
import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'
output_dir = f"{layer_divider_path}/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def tensor_to_pil(tensor):
    """ComfyUIのテンソル形式をPIL Imageに変換"""
    image_np = tensor.cpu().detach().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # バッチの最初の画像を取得
    image_np = (image_np * 255).astype(np.uint8)
    
    if image_np.shape[2] == 3:
        mode = 'RGB'
    elif image_np.shape[2] == 4:
        mode = 'RGBA'
    else:
        mode = 'L'
    
    return Image.fromarray(image_np, mode=mode)


def pil_to_tensor(image):
    """PIL ImageをComfyUIのテンソル形式に変換"""
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)


def find_contours(binary_image):
    """バイナリ画像から輪郭を検出する"""
    binary_array = np.array(binary_image, dtype=np.uint8)
    labeled_array, num_features = label(binary_array)
    return labeled_array, num_features


def get_binary_image(image, target_rgb):
    """指定された色のピクセルを黒に、それ以外を白にする"""
    copy_image = image.copy()
    pixels = copy_image.load()
    
    width, height = image.size
    rgb_list = list(target_rgb)
    rgb_list.append(255)
    target_rgb = tuple(rgb_list)
    
    for y in range(height):
        for x in range(width):
            current_rgb = pixels[x, y]
            if current_rgb == target_rgb:
                pixels[x, y] = (0, 0, 0)
            else:
                pixels[x, y] = (255, 255, 255)
    
    return copy_image


def thicken_and_recolor_lines(base_image, lineart, thickness=3, new_color=(0, 0, 0)):
    """
    線画の線を太くして、新しい色で再着色し、ベース画像に合成する
    """
    # 両方の画像をRGBA形式に変換
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')
    if lineart.mode != 'RGBA':
        lineart = lineart.convert('RGBA')
    
    # OpenCV形式に変換
    lineart_cv = np.array(lineart)
    
    # 各チャンネルを分離
    b, g, r, a = cv2.split(lineart_cv)
    
    # アルファ値の処理
    new_a = np.where(a == 0, 255, 255).astype(np.uint8)
    new_r = np.where(a == 0, 255, r).astype(np.uint8)
    new_g = np.where(a == 0, 255, g).astype(np.uint8)
    new_b = np.where(a == 0, 255, b).astype(np.uint8)
    
    # 画像を再結合
    lineart_cv = cv2.merge((new_r, new_g, new_b, new_a))
    
    white_pixels = np.sum(lineart_cv == 255)
    black_pixels = np.sum(lineart_cv == 0)
    
    lineart_gray = cv2.cvtColor(lineart_cv, cv2.COLOR_RGBA2GRAY)
    
    if white_pixels > black_pixels:
        lineart_gray = cv2.bitwise_not(lineart_gray)
    
    # 線を太くする
    kernel = np.ones((thickness, thickness), np.uint8)
    lineart_thickened = cv2.dilate(lineart_gray, kernel, iterations=1)
    
    # 新しい色で再着色
    lineart_recolored = np.zeros_like(lineart_cv)
    lineart_recolored[:, :, :3] = new_color  # 新しいRGB色を設定
    lineart_recolored[:, :, 3] = np.where(lineart_thickened < 10, 0, 255)  # アルファチャンネル
    
    # PIL Imageに変換
    lineart_recolored_pil = Image.fromarray(lineart_recolored, 'RGBA')
    
    # ベース画像に合成
    combined_image = Image.alpha_composite(base_image, lineart_recolored_pil)
    return combined_image


def simple_smooth_regions(labeled_array, num_features, kernel_size=5, min_area=50):
    """
    シンプルなスムージング処理
    """
    smoothed_array = np.zeros_like(labeled_array)
    
    for region_id in range(1, num_features + 1):
        # 各領域を個別に処理
        region_mask = (labeled_array == region_id).astype(np.uint8) * 255
        
        # 領域が小さすぎる場合はそのまま使用
        if np.sum(region_mask > 0) < min_area:
            # 小さい領域でもそのまま保持
            smoothed_array[labeled_array == region_id] = region_id
            continue
        
        # 1. メディアンフィルタでノイズ除去
        denoised = cv2.medianBlur(region_mask, 3)
        
        # 2. クロージング処理で穴を埋める
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # 3. 穴埋め処理
        filled = binary_fill_holes(closed > 127).astype(np.uint8) * 255
        
        # 4. ガウシアンブラーでエッジをスムース化
        smoothed = cv2.GaussianBlur(filled, (kernel_size, kernel_size), 1)
        
        # 5. 二値化
        _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
        # 結果を格納
        smoothed_array[binary > 0] = region_id
    
    return smoothed_array


def process_split_area_simple(lineart_image, fill_image=None, thickness=1, threshold=128, 
                              use_fill_colors=False, random_seed=None,
                              enable_smoothing=True, smoothing_strength=5, min_area=50):
    """
    簡易版の領域分割処理
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # ランダムな色を生成
    new_color = tuple(np.random.randint(0, 256, size=3))
    
    # 線を太くして元画像に貼り付け
    split_area_line = thicken_and_recolor_lines(
        lineart_image, lineart_image, thickness=thickness, new_color=new_color
    )
    
    # 太くした線のみを抽出
    tmp = get_binary_image(split_area_line, new_color)
    
    # グレースケール化と二値化
    gray_image = ImageOps.grayscale(tmp)
    binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, '1')
    
    # 輪郭検出
    labeled_array, num_features = find_contours(binary_image)
    
    # スムージング処理（有効な場合）
    if enable_smoothing and num_features > 0:
        # カーネルサイズをスムージング強度から計算
        kernel_size = max(3, min(11, smoothing_strength))
        if kernel_size % 2 == 0:  # 奇数にする
            kernel_size += 1
        
        smoothed_array = simple_smooth_regions(labeled_array, num_features, kernel_size, min_area)
        
        # スムース化が成功したかチェック
        if np.max(smoothed_array) > 0:
            labeled_array = smoothed_array
    
    # 出力画像の作成
    split_image = np.array(lineart_image.convert("RGBA"))
    
    if use_fill_colors and fill_image is not None:
        # 塗り画像から色を取得
        fill_array = np.array(fill_image.convert("RGBA"))
        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            # 各領域の中心点を見つける
            y_coords, x_coords = np.where(region_mask)
            if len(y_coords) > 0:
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                # 塗り画像から色を取得
                if center_y < fill_array.shape[0] and center_x < fill_array.shape[1]:
                    color = fill_array[center_y, center_x]
                else:
                    color = np.random.randint(0, 256, size=4).tolist()
                    color[3] = 255
            else:
                color = np.random.randint(0, 256, size=4).tolist()
                color[3] = 255
            split_image[region_mask] = color
    else:
        # ランダムな色で塗り分け
        for region_id in range(1, num_features + 1):
            random_color = np.random.randint(0, 256, size=3).tolist() + [255]
            region_mask = labeled_array == region_id
            split_image[region_mask] = random_color
    
    # PIL形式に変換
    colored_image = Image.fromarray(split_image.astype(np.uint8))
    
    return colored_image, binary_image, labeled_array


class SplitAreaSimpleSmoothNode:
    """
    簡易スムージング版の領域分割ノード
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lineart_image": ("IMAGE",),
                "thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Line Thickness"
                }),
                "threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Binary Threshold"
                }),
                "enable_smoothing": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Enable Smoothing"
                }),
                "smoothing_strength": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 11,
                    "step": 2,
                    "display": "slider",
                    "display_label": "Smoothing Strength"
                }),
                "min_area": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "display": "slider",
                    "display_label": "Minimum Region Area"
                }),
            },
            "optional": {
                "fill_image": ("IMAGE",),
                "use_fill_colors": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Use Fill Image Colors"
                }),
                "random_seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "display_label": "Random Seed (-1 for random)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("split_image", "binary_image", "region_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, lineart_image, thickness=1, threshold=128,
                enable_smoothing=True, smoothing_strength=5, min_area=50,
                fill_image=None, use_fill_colors=False, random_seed=-1):
        """
        領域分割処理を実行
        """
        
        # テンソルをPIL Imageに変換
        lineart_pil = tensor_to_pil(lineart_image)
        fill_pil = tensor_to_pil(fill_image) if fill_image is not None else None
        
        # ランダムシードの設定
        seed = None if random_seed == -1 else random_seed
        
        # 処理実行
        colored_image, binary_image, labeled_array = process_split_area_simple(
            lineart_pil,
            fill_pil,
            thickness,
            threshold,
            use_fill_colors,
            seed,
            enable_smoothing,
            smoothing_strength,
            min_area
        )
        
        # 領域マスクの作成
        region_mask = (labeled_array > 0).astype(np.float32)
        region_mask = np.expand_dims(region_mask, axis=0)
        region_mask = np.expand_dims(region_mask, axis=-1)
        region_mask_tensor = torch.from_numpy(region_mask)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(colored_image)
        binary_tensor = pil_to_tensor(binary_image.convert('RGB'))
        
        return (output_tensor, binary_tensor, region_mask_tensor)


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "SplitAreaSimpleSmooth": SplitAreaSimpleSmoothNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitAreaSimpleSmooth": "Split Area (Simple Smooth)",
}
