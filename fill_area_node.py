"""
Fill Area Node for ComfyUI
線画の塗り領域を均一化するノード
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from skimage import color as sk_color
from collections import Counter
import cv2
import folder_paths
import os
import logging

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    try:
        binary_array = np.array(binary_image, dtype=np.uint8)
        
        # グレースケールに変換してからラベリング
        if len(binary_array.shape) == 3:
            binary_array = np.mean(binary_array, axis=2).astype(np.uint8)
        
        labeled_array, num_features = label(binary_array)
        return labeled_array, num_features
    except Exception as e:
        logger.error(f"Error in find_contours: {str(e)}")
        raise


def get_most_frequent_color_fast(image_array, mask):
    """高速化された最頻出色取得"""
    pixels = image_array[mask]
    if len(pixels) == 0:
        return (0, 0, 0)
    
    if len(pixels.shape) == 1:
        return tuple(pixels)
    else:
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_frequent_idx = np.argmax(counts)
        return tuple(unique_colors[most_frequent_idx])


def get_all_most_frequent_colors_from_array(image_array, labeled_array, num_features):
    """NumPy配列から直接色を抽出（高速化版）"""
    colors_dict = {}
    
    for label_id in range(1, num_features + 1):
        mask = labeled_array == label_id
        if np.any(mask):
            pixels = image_array[mask]
            if len(pixels.shape) == 1:
                colors_dict[label_id] = tuple(pixels)
            else:
                unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
                most_frequent_idx = np.argmax(counts)
                colors_dict[label_id] = tuple(unique_colors[most_frequent_idx])
    
    return colors_dict


def get_all_most_frequent_colors(image, labeled_array, num_features):
    """全ラベルの最頻出色を一括計算"""
    image_array = np.array(image)
    return get_all_most_frequent_colors_from_array(image_array, labeled_array, num_features)


def rgb_to_lab_vectorized(colors):
    """ベクトル化されたRGB→Lab変換"""
    if isinstance(colors, tuple):
        colors = np.array([colors])
    elif len(colors.shape) == 1:
        colors = colors.reshape(1, -1)
    
    # RGBA to RGB if needed
    if colors.shape[1] == 4:
        colors = colors[:, :3]
    
    colors_normalized = colors.astype(np.float32) / 255.0
    lab = sk_color.rgb2lab(colors_normalized.reshape(-1, 1, 3))
    return lab.reshape(-1, 3)


def merge_similar_labels_optimized(labeled_array, colors, similarity_threshold=10):
    """最適化された類似ラベル統合"""
    if not colors:
        return labeled_array, 0
    
    # 色をNumPy配列に変換
    label_ids = list(colors.keys())
    color_array = np.array([colors[label_id] for label_id in label_ids])
    
    # Lab色空間に一括変換
    lab_colors = rgb_to_lab_vectorized(color_array)
    
    # 距離行列を一括計算
    distance_matrix = cdist(lab_colors, lab_colors, metric='euclidean')
    
    # Union-Find データ構造で効率的なグループ化
    parent = {label_id: label_id for label_id in label_ids}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
    
    # 類似色をグループ化
    for i in range(len(label_ids)):
        for j in range(i + 1, len(label_ids)):
            if distance_matrix[i, j] < similarity_threshold:
                union(label_ids[i], label_ids[j])
    
    # 新しいラベル配列を作成
    new_labeled_array = np.zeros_like(labeled_array)
    label_mapping = {}
    new_label_id = 1
    
    for old_label in label_ids:
        root = find(old_label)
        if root not in label_mapping:
            label_mapping[root] = new_label_id
            new_label_id += 1
        
        mask = labeled_array == old_label
        new_labeled_array[mask] = label_mapping[root]
    
    return new_labeled_array, len(label_mapping)


def merge_small_labels_optimized(image, labeled_array, colors, min_pixels=20):
    """最適化された小領域統合"""
    image_array = np.array(image)
    
    # 各ラベルのピクセル数を一括計算
    unique_labels, counts = np.unique(labeled_array, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    # 小さなラベルを特定
    small_labels = [label_id for label_id, count in label_counts.items() 
                   if label_id != 0 and count <= min_pixels and label_id in colors]
    
    if not small_labels:
        return labeled_array
    
    # 残りのラベル（統合先候補）
    remaining_labels = [label_id for label_id in colors.keys() if label_id not in small_labels]
    
    if not remaining_labels:
        return labeled_array
    
    # 色を一括でLab色空間に変換
    small_colors = np.array([colors[label_id] for label_id in small_labels])
    remaining_colors = np.array([colors[label_id] for label_id in remaining_labels])
    
    small_lab = rgb_to_lab_vectorized(small_colors)
    remaining_lab = rgb_to_lab_vectorized(remaining_colors)
    
    # 距離行列を一括計算
    distance_matrix = cdist(small_lab, remaining_lab, metric='euclidean')
    
    # 各小ラベルを最も近い色のラベルに統合
    for i, small_label in enumerate(small_labels):
        closest_idx = np.argmin(distance_matrix[i])
        closest_label = remaining_labels[closest_idx]
        
        # ラベル配列を更新
        mask = labeled_array == small_label
        labeled_array[mask] = closest_label
        
        # 色辞書を更新
        colors[closest_label] = get_most_frequent_color_fast(image_array, labeled_array == closest_label)
        del colors[small_label]
    
    return labeled_array


def fill_contours_with_color_vectorized(image, labeled_array, num_features):
    """ベクトル化による高速な色塗り"""
    image_array = np.array(image)
    
    # 各ラベルの最頻出色を一括計算
    colors_dict = get_all_most_frequent_colors_from_array(image_array, labeled_array, num_features)
    
    # ベクトル化された塗りつぶし
    for label_id in range(1, num_features + 1):
        if label_id in colors_dict:
            mask = labeled_array == label_id
            image_array[mask] = colors_dict[label_id]
    
    return Image.fromarray(image_array)


def process_fill_area(binary_image, fill_image, min_area_pixels=1000, similarity_threshold=10):
    """
    塗り領域を均一化する処理
    
    Args:
        binary_image: 輪郭画像（線画）
        fill_image: 塗り画像
        min_area_pixels: 小領域と判定するピクセル数の閾値
        similarity_threshold: 色の類似度閾値
    
    Returns:
        処理済みの画像
    """
    # 画像サイズの検証と調整
    if binary_image.size != fill_image.size:
        fill_image = fill_image.resize(binary_image.size, Image.Resampling.LANCZOS)
    
    # 輪郭検出
    labeled_array, num_features = find_contours(binary_image)
    
    if num_features == 0:
        # 輪郭が見つからない場合は元の画像をそのまま返す
        return fill_image
    
    # 色情報の取得
    colors = get_all_most_frequent_colors(fill_image, labeled_array, num_features)
    
    # 小領域の統合
    labeled_array = merge_small_labels_optimized(fill_image, labeled_array, colors, min_area_pixels)
    
    # 類似色の統合
    merged_labeled_array, merged_num_features = merge_similar_labels_optimized(
        labeled_array, colors, similarity_threshold
    )
    
    # 最終的な色塗り
    flat_image = fill_contours_with_color_vectorized(fill_image, merged_labeled_array, merged_num_features)
    
    return flat_image


class FillAreaNode:
    """
    塗り領域を均一化するノード
    線画の輪郭を検出し、各領域を単色で塗りつぶす
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "fill_image": ("IMAGE",),
                "min_area_pixels": ("INT", {
                    "default": 1000,
                    "min": 10,
                    "max": 10000,
                    "step": 10,
                    "display": "slider",
                    "display_label": "Minimum Area (pixels)"
                }),
                "similarity_threshold": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Color Similarity Threshold"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "preview")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, fill_image, min_area_pixels=1000, similarity_threshold=10):
        """
        塗り領域均一化処理を実行
        
        Args:
            binary_image: 輪郭画像（線画）のテンソル - ExtractLineArtNodeからのRGBA画像にも対応
            fill_image: 塗り画像のテンソル
            min_area_pixels: 小領域判定の閾値
            similarity_threshold: 色類似度の閾値
        
        Returns:
            filled_image: 均一化された塗り画像
            preview: 処理前後の比較画像
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        fill_pil = tensor_to_pil(fill_image)
        
        # ExtractLineArtNodeからのRGBA画像を処理
        if binary_pil.mode == 'RGBA':
            binary_pil = rgba_to_binary(binary_pil)
        
        # 処理実行
        result_image = process_fill_area(
            binary_pil.convert("RGB"),
            fill_pil.convert("RGB"),
            min_area_pixels,
            similarity_threshold
        )
        
        # 比較用のプレビュー画像を作成
        preview = create_comparison_preview(fill_pil, result_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(result_image)
        preview_tensor = pil_to_tensor(preview)
        
        return (output_tensor, preview_tensor)


class FillAreaAdvancedNode:
    """
    高度な塗り領域均一化ノード
    エッジ検出やアルファチャンネル出力などの追加機能付き
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "fill_image": ("IMAGE",),
                "min_area_pixels": ("INT", {
                    "default": 1000,
                    "min": 10,
                    "max": 10000,
                    "step": 10,
                    "display": "slider",
                    "display_label": "Minimum Area (pixels)"
                }),
                "similarity_threshold": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Color Similarity Threshold"
                }),
                "edge_enhancement": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Enhance Edges"
                }),
                "preserve_alpha": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Preserve Alpha Channel"
                }),
                "output_mode": (["filled", "contours", "labels", "comparison"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("filled_image", "visualization", "original", "contour_mask")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, fill_image, min_area_pixels=1000, 
                similarity_threshold=10, edge_enhancement=False,
                preserve_alpha=True, output_mode="filled"):
        """
        高度な塗り領域均一化処理
        
        Args:
            binary_image: 輪郭画像（線画） - ExtractLineArtNodeからのRGBA画像にも対応
            fill_image: 塗り画像
            min_area_pixels: 小領域判定の閾値
            similarity_threshold: 色類似度の閾値
            edge_enhancement: エッジ強調の有無
            preserve_alpha: アルファチャンネルを保持
            output_mode: 出力モード選択
        
        Returns:
            filled_image: 均一化された画像
            visualization: 選択されたモードの可視化
            original: 元の塗り画像
            contour_mask: 輪郭マスク
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        fill_pil = tensor_to_pil(fill_image)
        original_fill = fill_pil.copy()
        
        # ExtractLineArtNodeからのRGBA画像を処理
        if binary_pil.mode == 'RGBA':
            binary_pil = rgba_to_binary(binary_pil)
        
        # RGBに変換（処理用）
        binary_rgb = binary_pil.convert("RGB")
        fill_rgb = fill_pil.convert("RGB")
        
        # エッジ強調処理
        if edge_enhancement:
            binary_array = np.array(binary_rgb)
            gray = cv2.cvtColor(binary_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            binary_rgb = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        
        # 輪郭検出
        labeled_array, num_features = find_contours(binary_rgb)
        
        # 輪郭マスクの作成
        contour_mask = (labeled_array > 0).astype(np.float32)
        contour_mask = np.expand_dims(contour_mask, axis=0)
        contour_mask = np.expand_dims(contour_mask, axis=-1)
        contour_mask_tensor = torch.from_numpy(contour_mask)
        
        if num_features == 0:
            # 輪郭が見つからない場合
            output_tensor = pil_to_tensor(fill_pil)
            return (output_tensor, output_tensor, output_tensor, contour_mask_tensor)
        
        # 色情報の取得
        colors = get_all_most_frequent_colors(fill_rgb, labeled_array, num_features)
        
        # 小領域の統合
        labeled_array = merge_small_labels_optimized(fill_rgb, labeled_array, colors, min_area_pixels)
        
        # 類似色の統合
        merged_labeled_array, merged_num_features = merge_similar_labels_optimized(
            labeled_array, colors, similarity_threshold
        )
        
        # 最終的な色塗り
        flat_image = fill_contours_with_color_vectorized(fill_rgb, merged_labeled_array, merged_num_features)
        
        # アルファチャンネルの処理
        if preserve_alpha and fill_pil.mode == 'RGBA':
            # 元のアルファチャンネルを保持
            flat_image_rgba = flat_image.convert('RGBA')
            flat_image_rgba.putalpha(fill_pil.getchannel('A'))
            flat_image = flat_image_rgba
        
        # 可視化の作成
        if output_mode == "contours":
            visualization = visualize_contours(merged_labeled_array)
        elif output_mode == "labels":
            visualization = visualize_labels(merged_labeled_array, merged_num_features)
        elif output_mode == "comparison":
            visualization = create_comparison_preview(original_fill, flat_image)
        else:  # "filled"
            visualization = flat_image
        
        # テンソルに変換
        output_tensor = pil_to_tensor(flat_image)
        visualization_tensor = pil_to_tensor(visualization)
        original_tensor = pil_to_tensor(original_fill)
        
        return (output_tensor, visualization_tensor, original_tensor, contour_mask_tensor)


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
    
    return comparison


def visualize_contours(labeled_array):
    """輪郭の可視化"""
    # エッジ検出で輪郭を抽出
    from scipy import ndimage
    edges = ndimage.sobel(labeled_array)
    edges = (edges > 0).astype(np.uint8) * 255
    
    # RGB画像として返す
    edges_rgb = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_rgb)


def visualize_labels(labeled_array, num_features):
    """ラベルの可視化（各領域を異なる色で表示）"""
    # カラーマップの作成
    np.random.seed(42)  # 再現性のため
    colors = np.random.randint(0, 255, size=(num_features + 1, 3))
    colors[0] = [0, 0, 0]  # 背景は黒
    
    # ラベル画像をカラー画像に変換
    result = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    for label_id in range(num_features + 1):
        mask = labeled_array == label_id
        result[mask] = colors[label_id]
    
    return Image.fromarray(result)


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillArea": FillAreaNode,
    "FillArea Advanced": FillAreaAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillArea": "Fill Area",
    "FillArea Advanced": "Fill Area (Advanced)",
}
