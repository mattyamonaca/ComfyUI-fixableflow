"""
Simple Fill Area Node for ComfyUI
線画の塗り領域を単純に均一化するノード（統合処理なし）
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
import folder_paths
import os

# パス設定
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


def rgba_to_binary(image):
    """RGBA画像をバイナリ画像（線画）に変換
    
    ExtractLineArtNodeの出力（RGBA）を受け取り、
    アルファチャンネルを使って線画を生成する
    """
    if image.mode != 'RGBA':
        return image.convert('RGB')
    
    # アルファチャンネルを取得
    alpha = np.array(image.split()[3])
    
    # アルファチャンネルから線画を生成
    # アルファ値が高い（不透明）部分を黒、低い（透明）部分を白にする
    binary = np.ones((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8) * 255
    mask = alpha > 128  # 閾値を調整可能
    binary[mask] = 0  # 線画部分を黒に
    
    return Image.fromarray(binary, mode='RGB')


def find_contours(binary_image):
    """バイナリ画像から輪郭を検出する"""
    binary_array = np.array(binary_image, dtype=np.uint8)
    
    # グレースケールに変換してからラベリング
    if len(binary_array.shape) == 3:
        binary_array = np.mean(binary_array, axis=2).astype(np.uint8)
    
    labeled_array, num_features = label(binary_array)
    return labeled_array, num_features


def get_most_frequent_color(image_array, mask):
    """指定された領域内の最頻出色を取得"""
    pixels = image_array[mask]
    if len(pixels) == 0:
        return (0, 0, 0)
    
    if len(pixels.shape) == 1:
        return tuple(pixels)
    else:
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_frequent_idx = np.argmax(counts)
        return tuple(unique_colors[most_frequent_idx])


def fill_areas_simple(image, labeled_array, num_features):
    """各領域を最頻出色で塗りつぶす（統合処理なし）"""
    image_array = np.array(image)
    result_array = image_array.copy()
    
    # 各ラベル領域を単一色で塗りつぶし
    for label_id in range(1, num_features + 1):
        mask = labeled_array == label_id
        if np.any(mask):
            # その領域の最頻出色を取得
            most_frequent_color = get_most_frequent_color(image_array, mask)
            # 領域を単一色で塗りつぶし
            result_array[mask] = most_frequent_color
    
    return Image.fromarray(result_array)


def process_fill_area_simple(binary_image, fill_image):
    """
    シンプルな塗り領域均一化処理
    
    Args:
        binary_image: 輪郭画像（線画）
        fill_image: 塗り画像
    
    Returns:
        処理済みの画像
    """
    # 画像サイズの検証と調整
    if binary_image.size != fill_image.size:
        fill_image = fill_image.resize(binary_image.size, Image.Resampling.LANCZOS)
    
    # 輪郭検出とラベリング
    labeled_array, num_features = find_contours(binary_image)
    
    if num_features == 0:
        # 輪郭が見つからない場合は元の画像をそのまま返す
        return fill_image
    
    # 各領域を最頻出色で塗りつぶし（統合処理なし）
    result_image = fill_areas_simple(fill_image, labeled_array, num_features)
    
    return result_image


class FillAreaSimpleNode:
    """
    シンプルな塗り領域均一化ノード
    線画の輪郭を検出し、各領域を単色で塗りつぶす（統合処理なし）
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "fill_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("filled_image", "preview", "region_count")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, fill_image):
        """
        シンプルな塗り領域均一化処理を実行
        
        Args:
            binary_image: 輪郭画像（線画）のテンソル - ExtractLineArtNodeからのRGBA画像にも対応
            fill_image: 塗り画像のテンソル
        
        Returns:
            filled_image: 均一化された塗り画像
            preview: 処理前後の比較画像
            region_count: 検出された領域数
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        fill_pil = tensor_to_pil(fill_image)
        
        # ExtractLineArtNodeからのRGBA画像を処理
        if binary_pil.mode == 'RGBA':
            binary_pil = rgba_to_binary(binary_pil)
        
        # 処理実行
        result_image = process_fill_area_simple(
            binary_pil.convert("RGB"),
            fill_pil.convert("RGB")
        )
        
        # 領域数を取得（デバッグ用）
        labeled_array, num_features = find_contours(binary_pil.convert("RGB"))
        
        # 比較用のプレビュー画像を作成
        preview = create_comparison_preview(fill_pil, result_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(result_image)
        preview_tensor = pil_to_tensor(preview)
        
        return (output_tensor, preview_tensor, num_features)


class FillAreaSimpleVisualizeNode:
    """
    シンプルな塗り領域均一化ノード（可視化機能付き）
    各領域の境界や領域番号を表示できる
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "fill_image": ("IMAGE",),
                "visualization_mode": (["filled", "regions", "boundaries", "comparison"],),
                "show_region_numbers": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Show Region Numbers"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "MASK")
    RETURN_NAMES = ("filled_image", "visualization", "region_count", "region_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, fill_image, visualization_mode="filled", show_region_numbers=False):
        """
        シンプルな塗り領域均一化処理を実行（可視化機能付き）
        
        Args:
            binary_image: 輪郭画像（線画）のテンソル
            fill_image: 塗り画像のテンソル
            visualization_mode: 可視化モード
            show_region_numbers: 領域番号を表示するか
        
        Returns:
            filled_image: 均一化された塗り画像
            visualization: 選択された可視化
            region_count: 検出された領域数
            region_mask: 領域マスク
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        fill_pil = tensor_to_pil(fill_image)
        
        # ExtractLineArtNodeからのRGBA画像を処理
        if binary_pil.mode == 'RGBA':
            binary_pil = rgba_to_binary(binary_pil)
        
        # RGB変換
        binary_rgb = binary_pil.convert("RGB")
        fill_rgb = fill_pil.convert("RGB")
        
        # 輪郭検出とラベリング
        labeled_array, num_features = find_contours(binary_rgb)
        
        # 領域マスクの作成
        region_mask = (labeled_array > 0).astype(np.float32)
        region_mask = np.expand_dims(region_mask, axis=0)
        region_mask = np.expand_dims(region_mask, axis=-1)
        region_mask_tensor = torch.from_numpy(region_mask)
        
        # 塗りつぶし処理
        result_image = fill_areas_simple(fill_rgb, labeled_array, num_features)
        
        # 可視化の作成
        if visualization_mode == "regions":
            visualization = visualize_regions(labeled_array, num_features, show_region_numbers)
        elif visualization_mode == "boundaries":
            visualization = visualize_boundaries(labeled_array)
        elif visualization_mode == "comparison":
            visualization = create_comparison_preview(fill_pil, result_image)
        else:  # "filled"
            visualization = result_image
        
        # テンソルに変換
        output_tensor = pil_to_tensor(result_image)
        visualization_tensor = pil_to_tensor(visualization)
        
        return (output_tensor, visualization_tensor, num_features, region_mask_tensor)


def create_comparison_preview(original, processed):
    """処理前後の比較画像を作成"""
    width = original.width
    height = original.height
    
    # 左右に並べた比較画像を作成
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original.convert('RGB'), (0, 0))
    comparison.paste(processed.convert('RGB'), (width, 0))
    
    # 中央に区切り線を追加
    from PIL import ImageDraw
    draw = ImageDraw.Draw(comparison)
    draw.line([(width, 0), (width, height)], fill=(255, 0, 0), width=2)
    
    # ラベルを追加
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((width + 10, 10), "Processed", fill=(255, 255, 255))
    
    return comparison


def visualize_regions(labeled_array, num_features, show_numbers=False):
    """領域を異なる色で可視化"""
    # カラーマップの作成（各領域に異なる色を割り当て）
    np.random.seed(42)  # 再現性のため
    colors = np.random.randint(50, 255, size=(num_features + 1, 3))
    colors[0] = [0, 0, 0]  # 背景は黒
    
    # ラベル画像をカラー画像に変換
    result = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    for label_id in range(num_features + 1):
        mask = labeled_array == label_id
        result[mask] = colors[label_id]
    
    result_image = Image.fromarray(result)
    
    # 領域番号を表示
    if show_numbers and num_features > 0:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result_image)
        
        # 各領域の重心を計算して番号を表示
        for label_id in range(1, min(num_features + 1, 100)):  # 最大100個まで表示
            mask = labeled_array == label_id
            if np.any(mask):
                # 重心を計算
                coords = np.argwhere(mask)
                center_y, center_x = coords.mean(axis=0).astype(int)
                
                # 番号を描画
                draw.text((center_x, center_y), str(label_id), 
                         fill=(255, 255, 255), anchor="mm")
    
    return result_image


def visualize_boundaries(labeled_array):
    """領域の境界線を可視化"""
    from scipy import ndimage
    
    # エッジ検出で境界を抽出
    edges = ndimage.sobel(labeled_array)
    edges = (edges > 0).astype(np.uint8) * 255
    
    # 境界線を赤色で表示
    result = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    result[:, :, 0] = edges  # Red channel
    
    # 背景を薄いグレーにして境界を見やすくする
    background = labeled_array == 0
    result[background] = [50, 50, 50]
    
    return Image.fromarray(result)


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillAreaSimple": FillAreaSimpleNode,
    "FillAreaSimpleVisualize": FillAreaSimpleVisualizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillAreaSimple": "Fill Area Simple",
    "FillAreaSimpleVisualize": "Fill Area Simple (Visualize)",
}
