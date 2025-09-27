"""
Fill Space Node V2 for ComfyUI
線画の下のピクセルを、クラスタ化された色から最も近い色で塗りつぶすノード
CIEDE2000色差計算を使用（プログレスバー付き）
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from skimage import color as skcolor
import folder_paths
import os
from tqdm import tqdm

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


def rgb_to_lab(rgb_color):
    """RGB色をLAB色空間に変換"""
    # RGB値を0-1の範囲に正規化
    rgb_normalized = np.array(rgb_color).reshape(1, 1, 3) / 255.0
    # LAB色空間に変換
    lab = skcolor.rgb2lab(rgb_normalized)
    return lab[0, 0]


def ciede2000_distance(lab1, lab2):
    """CIEDE2000色差を計算"""
    # skimageのdeltae_ciede2000関数を使用
    # lab1とlab2は(L, a, b)の形式
    lab1_reshaped = np.array(lab1).reshape(1, 1, 3)
    lab2_reshaped = np.array(lab2).reshape(1, 1, 3)
    
    # CIEDE2000色差を計算
    delta_e = skcolor.deltaE_ciede2000(lab1_reshaped, lab2_reshaped)
    return delta_e[0, 0]


def find_closest_cluster_color(pixel_color, cluster_colors):
    """
    ピクセルの色に最も近いクラスタ色を見つける
    
    Args:
        pixel_color: RGB形式のピクセル色 (R, G, B)
        cluster_colors: クラスタID -> RGB色の辞書
    
    Returns:
        最も近いクラスタのRGB色
    """
    if not cluster_colors:
        return pixel_color
    
    # ピクセル色をLAB色空間に変換
    pixel_lab = rgb_to_lab(pixel_color)
    
    min_distance = float('inf')
    closest_color = pixel_color
    
    # 各クラスタ色との距離を計算
    for cluster_id, cluster_color in cluster_colors.items():
        cluster_lab = rgb_to_lab(cluster_color)
        distance = ciede2000_distance(pixel_lab, cluster_lab)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = cluster_color
    
    return closest_color


def process_fill_space_with_clusters_progress(binary_image, original_image, cluster_info, 
                                             invert_binary=True, progress_callback=None, flat_image=None):
    """
    線画の下のピクセルをクラスタ色で塗りつぶす（K-means最適化版）
    
    Args:
        binary_image: バイナリ画像（線画）
        original_image: 元の塗り画像（線画の下の色を取得するため）
        cluster_info: Fill Area Simpleからのクラスタ情報
        invert_binary: バイナリ画像を反転するか
        progress_callback: プログレスバーのコールバック
        flat_image: Fill Areaで処理された画像（ベース画像として使用）
    
    Returns:
        処理済みの画像
    """
    from sklearn.cluster import KMeans
    
    # バイナリ画像をグレースケールに変換
    if binary_image.mode != 'L':
        binary_gray = binary_image.convert('L')
    else:
        binary_gray = binary_image
    
    # 必要に応じてバイナリ画像を反転
    if invert_binary:
        binary_gray = ImageOps.invert(binary_gray)
    
    binary_array = np.array(binary_gray)
    original_array = np.array(original_image.convert('RGB'))
    
    # ベース画像としてflat_imageを使用（なければoriginal_imageを使用）
    if flat_image is not None:
        base_array = np.array(flat_image.convert('RGB'))
    else:
        base_array = original_array.copy()
    
    # クラスタ色情報を取得
    cluster_colors = cluster_info.get('colors', {})
    num_clusters = len(cluster_colors)
    
    if num_clusters == 0:
        if flat_image is not None:
            return flat_image
        return original_image
    
    # 出力画像を初期化（flat_imageのコピー）
    output_array = base_array.copy()
    
    # 白ピクセル（線画の下）のマスクを取得
    white_mask = binary_array == 255
    total_pixels = np.sum(white_mask)
    
    print(f"[FillSpaceV2] Processing {total_pixels} pixels under line art")
    print(f"[FillSpaceV2] Using {num_clusters} color clusters from Fill Area Simple")
    
    if total_pixels == 0:
        return original_image
    
    # 線画下のピクセル色を取得
    white_pixel_colors = original_array[white_mask]
    
    # 線画下のピクセルを同じクラスタ数でK-meansクラスタリング
    print(f"[FillSpaceV2] Clustering line art pixels into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=min(num_clusters, len(np.unique(white_pixel_colors, axis=0))), 
                    random_state=42, n_init=1)  # n_init=1で高速化
    line_art_labels = kmeans.fit_predict(white_pixel_colors)
    line_art_centers = kmeans.cluster_centers_
    
    print(f"[FillSpaceV2] Created {len(line_art_centers)} line art clusters")
    
    # Fill Areaのクラスタ色を配列に変換
    fill_cluster_ids = list(cluster_colors.keys())
    fill_cluster_array = np.array([cluster_colors[cid] for cid in fill_cluster_ids])
    
    # クラスタ間のマッピングを計算（クラスタ中心同士の比較）
    print(f"[FillSpaceV2] Computing cluster mappings ({len(line_art_centers)} x {len(fill_cluster_array)} comparisons)...")
    cluster_mapping = np.zeros(len(line_art_centers), dtype=np.int32)
    
    for i, center in enumerate(line_art_centers):
        # RGB空間での単純な距離計算
        distances = np.sum((fill_cluster_array - center) ** 2, axis=1)
        cluster_mapping[i] = np.argmin(distances)
    
    print(f"[FillSpaceV2] Applying mapped colors to pixels...")
    
    # 各ピクセルに対してマッピングされた色を適用
    for pixel_idx in range(len(white_pixel_colors)):
        line_cluster_id = line_art_labels[pixel_idx]
        fill_cluster_idx = cluster_mapping[line_cluster_id]
        mapped_color = fill_cluster_array[fill_cluster_idx]
        white_pixel_colors[pixel_idx] = mapped_color
    
    # 結果を出力画像に反映
    output_array[white_mask] = white_pixel_colors
    
    print(f"[FillSpaceV2] Completed processing")
    
    return Image.fromarray(output_array.astype(np.uint8))


def find_closest_cluster_color_optimized(pixel_color, cluster_colors, cluster_labs):
    """
    最適化版：事前計算されたLAB値を使用してピクセルの色に最も近いクラスタ色を見つける
    
    Args:
        pixel_color: RGB形式のピクセル色 (R, G, B)
        cluster_colors: クラスタID -> RGB色の辞書
        cluster_labs: クラスタID -> LAB色の辞書（事前計算済み）
    
    Returns:
        最も近いクラスタのRGB色
    """
    if not cluster_colors:
        return pixel_color
    
    # ピクセル色をLAB色空間に変換
    pixel_lab = rgb_to_lab(pixel_color)
    
    min_distance = float('inf')
    closest_color = pixel_color
    
    # 各クラスタ色との距離を計算
    for cluster_id, cluster_lab in cluster_labs.items():
        distance = ciede2000_distance(pixel_lab, cluster_lab)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = cluster_colors[cluster_id]
    
    return closest_color


def process_fill_space_batch_optimized(binary_image, original_image, cluster_info, invert_binary=True, flat_image=None):
    """
    バッチ処理最適化版：K-meansでクラスタ間マッピング（超高速版）
    
    Args:
        binary_image: バイナリ画像（線画）
        original_image: 元の塗り画像（線画の下の色を取得するため）
        cluster_info: Fill Area Simpleからのクラスタ情報
        invert_binary: バイナリ画像を反転するか
        flat_image: Fill Areaで処理された画像（ベース画像として使用）
    
    Returns:
        処理済みの画像
    """
    from sklearn.cluster import KMeans
    
    # バイナリ画像をグレースケールに変換
    if binary_image.mode != 'L':
        binary_gray = binary_image.convert('L')
    else:
        binary_gray = binary_image
    
    # 必要に応じてバイナリ画像を反転
    if invert_binary:
        binary_gray = ImageOps.invert(binary_gray)
    
    binary_array = np.array(binary_gray)
    original_array = np.array(original_image.convert('RGB'))
    
    # ベース画像としてflat_imageを使用（なければoriginal_imageを使用）
    if flat_image is not None:
        base_array = np.array(flat_image.convert('RGB'))
    else:
        base_array = original_array.copy()
    
    # クラスタ色情報を取得
    cluster_colors = cluster_info.get('colors', {})
    num_clusters = len(cluster_colors)
    
    if num_clusters == 0:
        if flat_image is not None:
            return flat_image
        return original_image
    
    # 出力画像を初期化（flat_imageのコピー）
    output_array = base_array.copy()
    
    # 白ピクセルのマスクを取得
    white_mask = binary_array == 255
    total_pixels = np.sum(white_mask)
    
    print(f"[FillSpaceV2] Batch processing {total_pixels} pixels")
    print(f"[FillSpaceV2] Using {num_clusters} color clusters")
    
    if total_pixels == 0:
        return original_image
    
    # 白ピクセルの色を取得
    white_pixel_colors = original_array[white_mask]
    
    # K-meansで線画下のピクセルをクラスタリング
    print(f"[FillSpaceV2] K-means clustering into {num_clusters} clusters...")
    unique_colors = np.unique(white_pixel_colors, axis=0)
    n_clusters = min(num_clusters, len(unique_colors))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    line_art_labels = kmeans.fit_predict(white_pixel_colors)
    line_art_centers = kmeans.cluster_centers_
    
    # Fill Areaクラスタ色を配列に変換
    cluster_ids = list(cluster_colors.keys())
    cluster_rgb_array = np.array([cluster_colors[cid] for cid in cluster_ids])
    
    # クラスタ中心間のマッピングを一括計算（ベクトル化）
    print(f"[FillSpaceV2] Computing optimal cluster mapping...")
    # 距離行列を計算（line_art_centers x cluster_rgb_array）
    distances = np.sum((line_art_centers[:, np.newaxis, :] - cluster_rgb_array[np.newaxis, :, :]) ** 2, axis=2)
    cluster_mapping = np.argmin(distances, axis=1)
    
    # マッピングされた色を一括適用
    mapped_colors = cluster_rgb_array[cluster_mapping]
    result_colors = mapped_colors[line_art_labels]
    
    # 出力配列に結果を設定
    output_array[white_mask] = result_colors
    
    print(f"[FillSpaceV2] Batch processing completed")
    
    return Image.fromarray(output_array.astype(np.uint8))


class FillSpaceV2Node:
    """
    線画の下のピクセルをクラスタ色で塗りつぶすノード
    CIEDE2000色差を使用して最も近いクラスタ色を選択（プログレスバー付き）
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),  # 線画（1枚目）
                "flat_image": ("IMAGE",),    # バケツ塗りされた画像（2枚目）
                "original_image": ("IMAGE",), # 元の塗り画像（3枚目）
                "cluster_info": ("CLUSTER_INFO",),  # Fill Area Simpleからのクラスタ情報
                "invert_binary": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Invert Binary Image"
                }),
                "use_batch_processing": ("BOOLEAN", {
                    "default": False,
                    "display_label": "Use Batch Processing (Faster)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "preview")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, binary_image, flat_image, original_image, cluster_info, 
                invert_binary=True, use_batch_processing=False):
        """
        線画の下のピクセルをクラスタ色で塗りつぶす処理
        
        Args:
            binary_image: バイナリ画像（線画）のテンソル
            flat_image: バケツ塗りされた画像のテンソル
            original_image: 元の塗り画像のテンソル
            cluster_info: クラスタ情報の辞書
            invert_binary: バイナリ画像を反転するか
            use_batch_processing: バッチ処理を使用するか
        
        Returns:
            filled_image: 処理済みの画像
            preview: 処理前後の比較画像
        """
        
        # テンソルをPIL Imageに変換
        binary_pil = tensor_to_pil(binary_image)
        flat_pil = tensor_to_pil(flat_image)
        original_pil = tensor_to_pil(original_image)
        
        # 処理方法を選択
        if use_batch_processing:
            # バッチ処理（高速だがメモリ使用量が多い）
            print("[FillSpaceV2] Using batch processing mode")
            filled_image = process_fill_space_batch_optimized(
                binary_pil, 
                original_pil,
                cluster_info,
                invert_binary,
                flat_pil  # flat_imageをベース画像として渡す
            )
        else:
            # 通常処理（プログレスバー付き）
            print("[FillSpaceV2] Using standard processing mode")
            filled_image = process_fill_space_with_clusters_progress(
                binary_pil, 
                original_pil,
                cluster_info,
                invert_binary,
                None,  # プログレスバーコールバックは一時的に無効化
                flat_pil  # flat_imageをベース画像として渡す
            )
        
        # プレビュー画像の作成（前後比較）
        preview = create_before_after_preview(flat_pil, filled_image)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(filled_image)
        preview_tensor = pil_to_tensor(preview)
        
        return (output_tensor, preview_tensor)


def create_before_after_preview(original, processed):
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


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillSpaceV2": FillSpaceV2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillSpaceV2": "Fill Space V2 (Cluster-based)",
}
