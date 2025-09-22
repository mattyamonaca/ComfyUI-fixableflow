"""
Enhanced Fill Area Node for ComfyUI
線画領域の特定と境界拡張機能を追加した塗りつぶしノード
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation, distance_transform_edt
import folder_paths
import os

# パス設定
comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-fixableflow'
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


def rgba_to_binary(image):
    """RGBA画像をバイナリ画像（線画）に変換"""
    if image.mode != 'RGBA':
        return image.convert('RGB')
    
    # アルファチャンネルを取得
    alpha = np.array(image.split()[3])
    
    # アルファチャンネルから線画を生成
    binary = np.ones((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8) * 255
    mask = alpha > 128  # 閾値を調整可能
    binary[mask] = 0  # 線画部分を黒に
    
    return Image.fromarray(binary, mode='RGB')


def find_contours_with_line_detection(binary_image):
    """バイナリ画像から輪郭を検出し、線画マスクも返す"""
    # PIL ImageをNumPy配列に変換
    binary_array = np.array(binary_image, dtype=np.uint8)
    
    # グレースケールに変換
    if len(binary_array.shape) == 3:
        gray = np.mean(binary_array, axis=2).astype(np.uint8)
    else:
        gray = binary_array
    
    # 線画マスクを作成（黒い部分が線画）
    line_mask = gray <= 128  # 黒い部分が線画
    
    # 白黒反転（線画が黒、背景が白の場合）
    binary_mask = gray > 128  # 白い部分を塗り領域として扱う
    
    # 連結成分のラベリング
    labeled_array, num_features = label(binary_mask)
    
    print(f"[FillAreaEnhanced] Detected {num_features} regions")
    
    return labeled_array, num_features, line_mask


def identify_line_art_region(labeled_array, num_features, line_mask):
    """線画と最も重なりが大きい領域を特定"""
    max_overlap = 0
    line_art_label = -1
    
    # 各領域について線画との重なりをチェック
    for label_id in range(0, num_features + 1):
        region_mask = labeled_array == label_id
        overlap = np.sum(region_mask & line_mask)
        
        if overlap > max_overlap:
            max_overlap = overlap
            line_art_label = label_id
            
    print(f"[FillAreaEnhanced] Line art region identified: label={line_art_label}, overlap={max_overlap} pixels")
    
    return line_art_label


def expand_regions(labeled_array, num_features, expansion_pixels, line_art_label):
    """各領域の境界を指定ピクセル数だけ拡張"""
    if expansion_pixels <= 0:
        return labeled_array
    
    expanded_array = labeled_array.copy()
    
    # 線画領域以外の各領域を拡張
    for label_id in range(1, num_features + 1):
        if label_id == line_art_label:
            continue  # 線画領域はスキップ
            
        # 現在の領域マスク
        region_mask = (labeled_array == label_id)
        
        # 構造要素を作成（円形カーネル）
        structure = np.ones((2 * expansion_pixels + 1, 2 * expansion_pixels + 1))
        y, x = np.ogrid[:2 * expansion_pixels + 1, :2 * expansion_pixels + 1]
        center = expansion_pixels
        mask = (x - center) ** 2 + (y - center) ** 2 <= expansion_pixels ** 2
        structure[~mask] = 0
        
        # 膨張処理
        expanded_mask = binary_dilation(region_mask, structure=structure)
        
        # 拡張部分のみを抽出（元の領域以外の部分）
        expansion_only = expanded_mask & ~region_mask
        
        # 線画領域や他の領域と重ならない部分のみ更新
        valid_expansion = expansion_only & (expanded_array == 0) | expansion_only & (expanded_array == line_art_label)
        expanded_array[valid_expansion] = label_id
    
    return expanded_array


def get_most_frequent_color(image_array, mask):
    """指定された領域内の最頻出色を取得"""
    pixels = image_array[mask]
    if len(pixels) == 0:
        return (0, 0, 0)
    
    if len(pixels.shape) == 1:
        return tuple(pixels)
    else:
        # ユニークな色とその頻度を計算
        pixels_reshaped = pixels.reshape(-1, pixels.shape[-1])
        unique_colors, counts = np.unique(pixels_reshaped, axis=0, return_counts=True)
        most_frequent_idx = np.argmax(counts)
        return tuple(unique_colors[most_frequent_idx])


def fill_areas_enhanced(image, labeled_array, num_features, line_art_label, line_color=(0, 255, 0)):
    """各領域を最頻出色で塗りつぶし、線画領域は指定色で塗る"""
    image_array = np.array(image)
    result_array = np.zeros_like(image_array)
    
    print(f"Total regions detected: {num_features}")
    print(f"Line art label: {line_art_label}")
    
    # デバッグ用：label=0の領域の色を調査
    if line_art_label == 0:
        line_mask = labeled_array == 0
        line_pixels = image_array[line_mask]
        if len(line_pixels) > 0:
            # ユニークな色とその頻度を表示
            unique_colors, counts = np.unique(line_pixels, axis=0, return_counts=True)
            print(f"[DEBUG] Colors in label=0 region:")
            for color, count in zip(unique_colors[:10], counts[:10]):  # 上位10色を表示
                print(f"  Color {color}: {count} pixels")
    
    # 各ラベル領域を処理
    for label_id in range(0, num_features + 1):
        mask = labeled_array == label_id
        if not np.any(mask):
            continue
            
        if label_id == line_art_label:
            # 線画領域は緑色（または指定色）で塗りつぶし
            result_array[mask] = line_color
            print(f"Region {label_id} (Line Art): filled with {line_color}")
        else:
            # その他の領域は最頻出色で塗りつぶし
            most_frequent_color = get_most_frequent_color(image_array, mask)
            result_array[mask] = most_frequent_color
            print(f"Region {label_id}: color = {most_frequent_color}, pixels = {np.sum(mask)}")
    
    return Image.fromarray(result_array)


def process_fill_area_enhanced(binary_image, fill_image, expansion_pixels=0, line_color=(0, 255, 0)):
    """
    拡張された塗り領域処理
    
    Args:
        binary_image: 輪郭画像（線画）
        fill_image: 塗り画像
        expansion_pixels: 境界拡張のピクセル数
        line_color: 線画領域の塗りつぶし色
    
    Returns:
        処理済みの画像
    """
    # 画像サイズの検証と調整
    if binary_image.size != fill_image.size:
        fill_image = fill_image.resize(binary_image.size, Image.Resampling.LANCZOS)
    
    # 輪郭検出とラベリング（線画マスク付き）
    labeled_array, num_features, line_mask = find_contours_with_line_detection(binary_image)
    
    if num_features == 0:
        # 輪郭が見つからない場合は元の画像をそのまま返す
        return fill_image
    
    # 線画領域を特定
    line_art_label = identify_line_art_region(labeled_array, num_features, line_mask)
    
    # 領域の境界を拡張
    expanded_array = expand_regions(labeled_array, num_features, expansion_pixels, line_art_label)
    
    # 各領域を塗りつぶし
    result_image = fill_areas_enhanced(fill_image, expanded_array, num_features, line_art_label, line_color)
    
    return result_image


class FillAreaEnhancedNode:
    """
    拡張版塗り領域処理ノード
    線画領域の特定と境界拡張機能付き
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "fill_image": ("IMAGE",),
                "expansion_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "slider"
                }),
                "line_color_r": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "line_color_g": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "line_color_b": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "MASK")
    RETURN_NAMES = ("filled_image", "preview", "region_count", "line_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, fill_image, expansion_pixels, 
                line_color_r, line_color_g, line_color_b):
        """
        拡張版塗り領域処理を実行
        
        Args:
            binary_image: 輪郭画像（線画）のテンソル
            fill_image: 塗り画像のテンソル
            expansion_pixels: 境界拡張のピクセル数
            line_color_r/g/b: 線画領域の塗りつぶし色
        
        Returns:
            filled_image: 処理済み画像
            preview: 処理前後の比較画像
            region_count: 検出された領域数
            line_mask: 線画領域のマスク
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        fill_pil = tensor_to_pil(fill_image)
        
        # ExtractLineArtNodeからのRGBA画像を処理
        if binary_pil.mode == 'RGBA':
            binary_pil = rgba_to_binary(binary_pil)
        
        # 線画色を設定
        line_color = (line_color_r, line_color_g, line_color_b)
        
        # 処理実行
        result_image = process_fill_area_enhanced(
            binary_pil.convert("RGB"),
            fill_pil.convert("RGB"),
            expansion_pixels=expansion_pixels,
            line_color=line_color
        )
        
        # 領域数と線画マスクを取得
        labeled_array, num_features, line_mask = find_contours_with_line_detection(
            binary_pil.convert("RGB")
        )
        
        # 線画マスクをテンソルに変換
        line_mask_float = line_mask.astype(np.float32)
        line_mask_tensor = torch.from_numpy(line_mask_float).unsqueeze(0)
        
        # 比較用のプレビュー画像を作成
        preview = create_comparison_preview(fill_pil, result_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(result_image)
        preview_tensor = pil_to_tensor(preview)
        
        return (output_tensor, preview_tensor, num_features, line_mask_tensor)


def create_comparison_preview(original, processed):
    """処理前後の比較画像を作成"""
    width = original.width
    height = original.height
    
    # 左右に並べた比較画像を作成
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original.convert('RGB'), (0, 0))
    comparison.paste(processed.convert('RGB'), (width, 0))
    
    # 中央に区切り線を追加
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    draw.line([(width, 0), (width, height)], fill=(255, 0, 0), width=2)
    
    # ラベルを追加
    try:
        # フォントサイズを設定
        font_size = max(12, min(30, height // 20))
        # システムフォントを使用
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
    draw.text((width + 10, 10), "Enhanced", fill=(255, 255, 255), font=font)
    
    return comparison


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillAreaEnhanced": FillAreaEnhancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillAreaEnhanced": "Fill Area Enhanced (Line & Expansion)",
}
