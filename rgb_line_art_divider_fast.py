"""
RGB Line Art Layer Divider - FAST VERSION
高速化版：同じRGB値でグループ化してから統合
ld_utils.pyの実装に準拠した安定版
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_convertor import pil2cv, df2rgba
from pytoshop.enums import BlendMode
import cv2
import pytoshop
from pytoshop.core import PsdFile
from pytoshop import layers
from pytoshop import enums
import random
import string

# パス設定
# ComfyUIの標準出力ディレクトリを使用
try:
    output_dir = folder_paths.get_output_directory()
except:
    # フォールバック: カスタムノードディレクトリ内のoutputを使用
    comfy_path = os.path.dirname(folder_paths.__file__)
    layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-fixableflow'
    output_dir = f"{layer_divider_path}/output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def randomname(n):
    """ld_utils.pyと同じランダムファイル名生成"""
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return ''.join(randlst)


def HWC3(x):
    """画像形式の変換ヘルパー関数"""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def to_comfy_img(np_img):
    """NumPy配列をComfyUI形式の画像に変換"""
    out_imgs = []
    out_imgs.append(HWC3(np_img))
    out_imgs = np.stack(out_imgs)
    out_imgs = torch.from_numpy(out_imgs.astype(np.float32) / 255.)
    return out_imgs


def add_psd(psd, img, name, mode):
    """
    ld_utils.pyのadd_psd関数と完全に同じ実装
    """
    # 確実にuint8型、4チャンネルに変換
    if img.shape[2] == 3:
        # BGRにアルファチャンネルを追加
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
        img = np.concatenate([img, alpha], axis=2)
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # OpenCVのBGRA形式からpytoshop用のRGBA形式に変換
    # imgは既にBGRA形式（OpenCV形式）なので、RGBAに変換
    img_rgba = np.zeros_like(img, dtype=np.uint8)
    img_rgba[:, :, 0] = img[:, :, 2]  # B -> R
    img_rgba[:, :, 1] = img[:, :, 1]  # G -> G
    img_rgba[:, :, 2] = img[:, :, 0]  # R -> B
    img_rgba[:, :, 3] = img[:, :, 3]  # A -> A
    
    # ld_utils.pyと同じ形式でチャンネルデータを作成
    layer_1 = layers.ChannelImageData(image=img_rgba[:, :, 3], compression=1)
    layer0 = layers.ChannelImageData(image=img_rgba[:, :, 0], compression=1)
    layer1 = layers.ChannelImageData(image=img_rgba[:, :, 1], compression=1)
    layer2 = layers.ChannelImageData(image=img_rgba[:, :, 2], compression=1)

    new_layer = layers.LayerRecord(channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2},
                                   top=0, bottom=img.shape[0], left=0, right=img.shape[1],
                                   blend_mode=mode,
                                   name=name,
                                   opacity=255,
                                   )
    psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
    return psd


def extract_color_regions_fast(base_image_cv, tolerance=10, max_colors=50):
    """
    高速版：下塗り画像からRGB値ごとに領域を抽出
    まず同じRGB値でグループ化し、その後tolerance内の色を統合
    
    Args:
        base_image_cv: 下塗り画像（BGRA形式）
        tolerance: 同じ色と判定する許容値
        max_colors: 最大色数（統合後の目標色数）
    
    Returns:
        color_regions: {(R,G,B): mask} の辞書
    """
    # BGRAからRGBに変換
    if base_image_cv.shape[2] == 4:
        bgr_image = base_image_cv[:, :, :3]
        alpha = base_image_cv[:, :, 3]
    else:
        bgr_image = base_image_cv
        alpha = np.ones((base_image_cv.shape[0], base_image_cv.shape[1]), dtype=np.uint8) * 255
    
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    
    # アルファマスクを作成
    valid_mask = alpha > 0
    
    # ステップ1: 同じRGB値のピクセルをグループ化
    print("[Fast] Step 1: Grouping pixels by unique colors...")
    
    # 有効なピクセルのみを処理
    pixel_array = rgb_image.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    
    # ユニーク色とインデックスマップを取得（高速化）
    unique_colors, inverse_indices = np.unique(
        pixel_array[valid_flat], 
        axis=0, 
        return_inverse=True
    )
    
    print(f"[Fast] Found {len(unique_colors)} unique colors")
    
    # 各色のピクセル数を計算
    color_counts = np.bincount(inverse_indices)
    
    # ステップ2: tolerance内の色を統合
    print(f"[Fast] Step 2: Merging colors within tolerance {tolerance}...")
    
    if tolerance > 0 and len(unique_colors) > 1:
        # 色グループを統合
        merged_groups = merge_colors_by_tolerance(unique_colors, color_counts, tolerance, max_colors)
        
        # マスクを作成
        color_regions = {}
        for group in merged_groups:
            # このグループに属する全ピクセルのマスクを作成
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for color_idx in group['indices']:
                # この色に属するピクセルを取得
                pixel_mask = (inverse_indices == color_idx)
                # フルサイズのマスクに展開
                full_mask = np.zeros(len(valid_flat), dtype=bool)
                full_mask[valid_flat] = pixel_mask
                mask = np.maximum(mask, full_mask.reshape(height, width).astype(np.uint8) * 255)
            
            if np.any(mask):
                color_regions[tuple(group['center'])] = mask
    else:
        # tolerance=0の場合、全ユニーク色をそのまま使用
        color_regions = {}
        for i, color in enumerate(unique_colors):
            # この色に属するピクセルのマスクを作成
            pixel_mask = (inverse_indices == i)
            full_mask = np.zeros(len(valid_flat), dtype=bool)
            full_mask[valid_flat] = pixel_mask
            mask = full_mask.reshape(height, width).astype(np.uint8) * 255
            
            if np.any(mask):
                color_regions[tuple(color)] = mask
    
    print(f"[Fast] Final: {len(color_regions)} color regions")
    return color_regions


def merge_colors_by_tolerance(unique_colors, color_counts, tolerance, max_colors):
    """
    tolerance内の色をグループ化して統合
    
    Args:
        unique_colors: ユニーク色の配列
        color_counts: 各色のピクセル数
        tolerance: 許容値
        max_colors: 最大色数
    
    Returns:
        merged_groups: 統合されたグループのリスト
    """
    n_colors = len(unique_colors)
    
    # 色をピクセル数でソート（多い順）
    sorted_indices = np.argsort(color_counts)[::-1]
    
    merged_groups = []
    used = np.zeros(n_colors, dtype=bool)
    
    for idx in sorted_indices:
        if used[idx]:
            continue
        
        # 新しいグループを開始
        group = {
            'indices': [idx],
            'colors': [unique_colors[idx]],
            'counts': [color_counts[idx]],
            'total_count': color_counts[idx]
        }
        used[idx] = True
        
        # tolerance内の色を探して統合
        if tolerance > 0:
            base_color = unique_colors[idx]
            for other_idx in sorted_indices:
                if used[other_idx] or other_idx == idx:
                    continue
                
                # 色差を計算
                color_diff = np.abs(unique_colors[other_idx] - base_color)
                if np.all(color_diff <= tolerance):
                    group['indices'].append(other_idx)
                    group['colors'].append(unique_colors[other_idx])
                    group['counts'].append(color_counts[other_idx])
                    group['total_count'] += color_counts[other_idx]
                    used[other_idx] = True
        
        # グループの中心色を計算（重み付き平均）
        colors = np.array(group['colors'])
        counts = np.array(group['counts'])
        group['center'] = np.average(colors, axis=0, weights=counts).astype(int)
        
        merged_groups.append(group)
        
        # 最大色数に達したら終了
        if len(merged_groups) >= max_colors:
            # 残りの色を最後のグループに統合
            for other_idx in sorted_indices:
                if not used[other_idx]:
                    merged_groups[-1]['indices'].append(other_idx)
                    used[other_idx] = True
            break
    
    print(f"[Fast] Merged {n_colors} colors into {len(merged_groups)} groups")
    return merged_groups


def merge_small_regions_fast(color_regions, min_region_size=100):
    """
    高速版：小さい領域を効率的にマージ
    """
    filtered_regions = {}
    small_region_mask = None
    
    for color, mask in color_regions.items():
        # 領域サイズを計算
        region_size = np.sum(mask > 0)
        
        if region_size >= min_region_size:
            filtered_regions[color] = mask
        else:
            # 小さい領域を統合
            if small_region_mask is None:
                small_region_mask = mask.copy()
            else:
                small_region_mask = np.maximum(small_region_mask, mask)
    
    # 小さい領域を「その他」として追加
    if small_region_mask is not None and np.any(small_region_mask > 0):
        filtered_regions[(128, 128, 128)] = small_region_mask
    
    return filtered_regions


def create_region_layers(base_image_cv, color_regions):
    """
    色領域ごとにレイヤーを作成（ld_utils.pyの形式に準拠）
    """
    layers = []
    names = []
    
    # base_image_cvが確実にBGRA形式であることを確認
    if base_image_cv.shape[2] == 3:
        # アルファチャンネルを追加
        alpha = np.ones((base_image_cv.shape[0], base_image_cv.shape[1], 1), dtype=np.uint8) * 255
        base_image_cv = np.concatenate([base_image_cv, alpha], axis=2)
    
    # base_image_cvの値を確認
    base_image_cv = np.clip(base_image_cv, 0, 255).astype(np.uint8)
    
    for color, mask in color_regions.items():
        # マスクを適用してレイヤーを作成
        # 完全に透明な背景から開始
        layer = np.zeros_like(base_image_cv, dtype=np.uint8)
        
        # マスクを確実にuint8に変換
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        # マスクがある部分だけベース画像をコピー
        mask_bool = mask > 0
        
        # BGRチャンネルをコピー
        layer[:, :, :3][mask_bool] = base_image_cv[:, :, :3][mask_bool]
        
        # アルファチャンネルをマスクから設定
        layer[:, :, 3] = mask
        
        # データの整合性チェック - 確実に0-255の範囲に
        layer = np.clip(layer, 0, 255).astype(np.uint8)
        
        layers.append(layer)
        names.append(f"Color_R{color[0]}_G{color[1]}_B{color[2]}")
    
    return layers, names


def save_psd_fast(base_color_cv, line_art_cv, color_layers, layer_names, 
                  output_dir, line_blend_mode, layer_mode="normal"):
    """
    ld_utils.pyのsave_psd関数と同じ構造でPSDを保存
    
    Args:
        base_color_cv: ベース画像（BGRA形式）
        line_art_cv: 線画（BGRA形式）
        color_layers: 色領域レイヤーのリスト
        layer_names: レイヤー名のリスト
        output_dir: 出力ディレクトリ
        line_blend_mode: 線画のブレンドモード
        layer_mode: レイヤーモード（"normal" or "composite"）
    
    Returns:
        filename: 保存したファイル名
    """
    # 入力画像のデータ型と範囲を確実に修正
    base_color_cv = np.clip(base_color_cv, 0, 255).astype(np.uint8)
    line_art_cv = np.clip(line_art_cv, 0, 255).astype(np.uint8)
    
    height, width = base_color_cv.shape[:2]
    
    # PSDファイルを作成（ld_utils.pyと同じ初期化）
    psd = pytoshop.core.PsdFile(num_channels=3, height=height, width=width)
    
    # レイヤーを追加する配列を準備（ld_utils.pyのlayers構造を模倣）
    # layers[0]: ベースレイヤー群
    # layers[1]: 色領域レイヤー群
    # layers[2]: 線画レイヤー群
    
    # 背景レイヤーを追加
    psd = add_psd(psd, base_color_cv, "Background", enums.BlendMode.normal)
    
    # 色領域レイヤーを追加
    for idx, (layer_data, base_name) in enumerate(zip(color_layers, layer_names)):
        layer_data = np.clip(layer_data, 0, 255).astype(np.uint8)
        # ld_utils.pyの命名規則に従う
        layer_name = f"{base_name}_{idx}" if layer_mode == "normal" else base_name
        psd = add_psd(psd, layer_data, layer_name, enums.BlendMode.normal)
    
    # 線画レイヤーを最上位に追加
    # 線画が黒い場合の特別処理
    if line_art_cv.shape[2] == 4:
        alpha = line_art_cv[:, :, 3]
        # RGBチャンネルの最大値をチェック
        rgb_max = np.max(line_art_cv[:, :, :3])
        
        if rgb_max < 10:  # ほぼ黒の線画
            # アルファチャンネルから線画を生成
            line_intensity = 255 - alpha  # アルファの反転
            line_art_fixed = np.zeros_like(line_art_cv)
            line_art_fixed[:, :, 0] = line_intensity
            line_art_fixed[:, :, 1] = line_intensity
            line_art_fixed[:, :, 2] = line_intensity
            line_art_fixed[:, :, 3] = alpha
            psd = add_psd(psd, line_art_fixed, "Line Art", line_blend_mode)
        else:
            psd = add_psd(psd, line_art_cv, "Line Art", line_blend_mode)
    else:
        psd = add_psd(psd, line_art_cv, "Line Art", line_blend_mode)
    
    # ファイル名生成（ld_utils.pyと同じ形式）
    name = randomname(10)
    filename_only = f"output_rgb_fast_{layer_mode}_{name}.psd"
    filename = os.path.join(output_dir, filename_only)
    
    with open(filename, 'wb') as fd:
        psd.write(fd)
    
    return filename


class RGBLineArtDividerFast:
    """
    高速版：RGB線画と下塗り画像から領域分割PSDを生成するノード
    ld_utils.pyの実装に準拠
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "line_art": ("IMAGE",),
                "base_color": ("IMAGE",),
                "color_tolerance": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "line_blend_mode": (["multiply", "normal", "darken", "overlay"],),
                "merge_small_regions": ("BOOLEAN", {
                    "default": True
                }),
                "min_region_size": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "slider"
                }),
                "max_colors": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("composite", "base_color", "layer_count", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"
    OUTPUT_NODE = True

    def execute(self, line_art, base_color, color_tolerance, line_blend_mode, 
                merge_small_regions, min_region_size, max_colors):
        
        try:
            print("[RGBLineArtDividerFast] Starting execution...")
            
            # 画像をNumPy配列に変換
            print("[RGBLineArtDividerFast] Converting tensors to numpy arrays...")
            line_art_np = line_art.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
            base_color_np = base_color.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
            print(f"[RGBLineArtDividerFast] Image shape: {base_color_np.shape}")
            
            # PIL Imageに変換
            line_art_pil = Image.fromarray(line_art_np)
            base_color_pil = Image.fromarray(base_color_np)
            
            # OpenCV形式（BGRA）に変換
            line_art_cv = pil2cv(line_art_pil)
            base_color_cv = pil2cv(base_color_pil)
            
            # BGRAに変換（アルファチャンネルを追加）
            if line_art_cv.shape[2] == 3:
                line_art_cv = cv2.cvtColor(line_art_cv, cv2.COLOR_BGR2BGRA)
            if base_color_cv.shape[2] == 3:
                base_color_cv = cv2.cvtColor(base_color_cv, cv2.COLOR_BGR2BGRA)
            
            # 色領域を抽出（高速版を使用）
            print(f"[RGBLineArtDividerFast] Extracting color regions...")
            color_regions = extract_color_regions_fast(
                base_color_cv, 
                tolerance=color_tolerance,
                max_colors=max_colors
            )
            print(f"[RGBLineArtDividerFast] Found {len(color_regions)} color regions")
            
            # 小さい領域をマージ
            if merge_small_regions:
                print("[RGBLineArtDividerFast] Merging small regions...")
                color_regions = merge_small_regions_fast(color_regions, min_region_size)
                print(f"[RGBLineArtDividerFast] After merging: {len(color_regions)} regions")
            
            # レイヤーを作成
            print("[RGBLineArtDividerFast] Creating layers...")
            color_layers, layer_names = create_region_layers(base_color_cv, color_regions)
            
            # BlendModeの設定（ld_utils.pyと同じ形式）
            blend_mode_map = {
                "multiply": enums.BlendMode.multiply,
                "normal": enums.BlendMode.normal,
                "darken": enums.BlendMode.darken,
                "overlay": enums.BlendMode.overlay
            }
            
            # PSDファイルを保存（ld_utils.pyの方式を完全に踏襲）
            print("[RGBLineArtDividerFast] Saving PSD file...")
            filename = save_psd_fast(
                base_color_cv,
                line_art_cv,
                color_layers,
                layer_names,
                output_dir,
                blend_mode_map[line_blend_mode],
                layer_mode="normal"  # ld_utils.pyのnormalモードに対応
            )
            
            print(f"[RGBLineArtDividerFast] PSD file saved: {filename}")
            print(f"[RGBLineArtDividerFast] Created {len(color_regions)} color region layers")
            
            # コンポジット画像を作成（プレビュー用）
            print("[RGBLineArtDividerFast] Creating composite image...")
            composite = base_color_cv.copy()
            if line_blend_mode == "multiply":
                # 乗算合成（ベクトル化）
                line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
                composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
                composite[:, :, :3] = (composite_rgb * line_rgb * 255).astype(np.uint8)
            elif line_blend_mode == "normal":
                # アルファブレンディング（ベクトル化）
                if line_art_cv.shape[2] == 4:
                    alpha = line_art_cv[:, :, 3:4].astype(np.float32) / 255.0
                    composite[:, :, :3] = (
                        line_art_cv[:, :, :3] * alpha + 
                        composite[:, :, :3] * (1 - alpha)
                    ).astype(np.uint8)
            
            print("[RGBLineArtDividerFast] Execution completed successfully!")
            
            # 出力
            return (
                to_comfy_img(composite),
                to_comfy_img(base_color_cv),
                len(color_regions),
                filename
            )
        
        except Exception as e:
            print(f"[RGBLineArtDividerFast] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
