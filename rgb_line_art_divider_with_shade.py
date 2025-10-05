"""
RGB Line Art Layer Divider with Shade - Fixed VERSION
shade画像を追加入力として受け取り、各領域ごとにbase/light/shadeレイヤーを生成する拡張版
画像サイズ自動調整とLayerFlagsエラー修正済み
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_convertor import pil2cv
from pytoshop.enums import BlendMode
import cv2
import pytoshop
from pytoshop.core import PsdFile
from pytoshop import layers
from pytoshop import enums
import random
import string
from skimage import color as skcolor

# パス設定
try:
    output_dir = folder_paths.get_output_directory()
except:
    comfy_path = os.path.dirname(folder_paths.__file__)
    layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-fixableflow'
    output_dir = f"{layer_divider_path}/output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 共用ヘルパー関数
def randomname(n):
    """ランダムファイル名生成"""
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
    PSDファイルにレイヤーを追加（RGBLineArtDividerFastと同じ実装）
    """
    # 確実にuint8型、4チャンネルに変換
    if img.shape[2] == 3:
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
        img = np.concatenate([img, alpha], axis=2)
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # OpenCVのBGRA形式からpytoshop用のRGBA形式に変換
    img_rgba = np.zeros_like(img, dtype=np.uint8)
    img_rgba[:, :, 0] = img[:, :, 2]  # B -> R
    img_rgba[:, :, 1] = img[:, :, 1]  # G -> G
    img_rgba[:, :, 2] = img[:, :, 0]  # R -> B
    img_rgba[:, :, 3] = img[:, :, 3]  # A -> A
    
    # チャンネルデータを作成
    layer_1 = layers.ChannelImageData(image=img_rgba[:, :, 3], compression=1)
    layer0 = layers.ChannelImageData(image=img_rgba[:, :, 0], compression=1)
    layer1 = layers.ChannelImageData(image=img_rgba[:, :, 1], compression=1)
    layer2 = layers.ChannelImageData(image=img_rgba[:, :, 2], compression=1)

    new_layer = layers.LayerRecord(
        channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2},
        top=0, bottom=img.shape[0], left=0, right=img.shape[1],
        blend_mode=mode,
        name=name,
        opacity=255,
    )
    
    psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
    return psd

def rgb_to_lab(rgb_image):
    """RGB画像をLAB色空間に変換"""
    # 確実にRGB形式（0-1の範囲）に正規化
    if rgb_image.dtype == np.uint8:
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_normalized = rgb_image
    
    # LAB色空間に変換
    lab_image = skcolor.rgb2lab(rgb_normalized)
    return lab_image

def classify_pixels_by_luminance(base_pixel_rgb, shade_pixel_rgb, luminance_threshold=5.0):
    """
    ベース色とシェード色のLab空間での輝度(L)を比較して分類
    
    Args:
        base_pixel_rgb: ベース色のRGB値 (0-255)
        shade_pixel_rgb: シェード色のRGB値 (0-255)
        luminance_threshold: 輝度差の閾値
    
    Returns:
        'light' or 'shade' or 'base'
    """
    # RGB to LAB変換（単一ピクセル用）
    base_lab = rgb_to_lab(np.array([[base_pixel_rgb]], dtype=np.uint8))[0, 0]
    shade_lab = rgb_to_lab(np.array([[shade_pixel_rgb]], dtype=np.uint8))[0, 0]
    
    # L値（輝度）の差を計算
    l_diff = shade_lab[0] - base_lab[0]
    
    if l_diff > luminance_threshold:
        return 'light'
    elif l_diff < -luminance_threshold:
        return 'shade'
    else:
        return 'base'

def extract_color_regions_fast(base_image_cv, tolerance=10, max_colors=50):
    """
    下塗り画像からRGB値ごとに領域を抽出（RGBLineArtDividerFastから流用）
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
    
    print("[WithShade] Step 1: Grouping pixels by unique colors...")
    
    pixel_array = rgb_image.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    
    unique_colors, inverse_indices = np.unique(
        pixel_array[valid_flat], 
        axis=0, 
        return_inverse=True
    )
    
    print(f"[WithShade] Found {len(unique_colors)} unique colors")
    
    color_counts = np.bincount(inverse_indices)
    
    print(f"[WithShade] Step 2: Merging colors within tolerance {tolerance}...")
    
    if tolerance > 0 and len(unique_colors) > 1:
        merged_groups = merge_colors_by_tolerance(unique_colors, color_counts, tolerance, max_colors)
        
        color_regions = {}
        for group in merged_groups:
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for color_idx in group['indices']:
                pixel_mask = (inverse_indices == color_idx)
                full_mask = np.zeros(len(valid_flat), dtype=bool)
                full_mask[valid_flat] = pixel_mask
                mask = np.maximum(mask, full_mask.reshape(height, width).astype(np.uint8) * 255)
            
            if np.any(mask):
                color_regions[tuple(group['center'])] = mask
    else:
        color_regions = {}
        for i, color in enumerate(unique_colors):
            pixel_mask = (inverse_indices == i)
            full_mask = np.zeros(len(valid_flat), dtype=bool)
            full_mask[valid_flat] = pixel_mask
            mask = full_mask.reshape(height, width).astype(np.uint8) * 255
            
            if np.any(mask):
                color_regions[tuple(color)] = mask
    
    print(f"[WithShade] Final: {len(color_regions)} color regions")
    return color_regions

def merge_colors_by_tolerance(unique_colors, color_counts, tolerance, max_colors):
    """
    tolerance内の色をグループ化して統合（RGBLineArtDividerFastから流用）
    """
    n_colors = len(unique_colors)
    
    sorted_indices = np.argsort(color_counts)[::-1]
    
    merged_groups = []
    used = np.zeros(n_colors, dtype=bool)
    
    for idx in sorted_indices:
        if used[idx]:
            continue
        
        group = {
            'indices': [idx],
            'colors': [unique_colors[idx]],
            'counts': [color_counts[idx]],
            'total_count': color_counts[idx]
        }
        used[idx] = True
        
        if tolerance > 0:
            base_color = unique_colors[idx]
            for other_idx in sorted_indices:
                if used[other_idx] or other_idx == idx:
                    continue
                
                color_diff = np.abs(unique_colors[other_idx] - base_color)
                if np.all(color_diff <= tolerance):
                    group['indices'].append(other_idx)
                    group['colors'].append(unique_colors[other_idx])
                    group['counts'].append(color_counts[other_idx])
                    group['total_count'] += color_counts[other_idx]
                    used[other_idx] = True
        
        colors = np.array(group['colors'])
        counts = np.array(group['counts'])
        group['center'] = np.average(colors, axis=0, weights=counts).astype(int)
        
        merged_groups.append(group)
        
        if len(merged_groups) >= max_colors:
            for other_idx in sorted_indices:
                if not used[other_idx]:
                    merged_groups[-1]['indices'].append(other_idx)
                    used[other_idx] = True
            break
    
    print(f"[WithShade] Merged {n_colors} colors into {len(merged_groups)} groups")
    return merged_groups

def create_shade_layers(base_image_cv, shade_image_cv, color_regions, luminance_threshold=5.0):
    """
    各色領域ごとにbase/light/shadeレイヤーを作成
    画像サイズを自動調整
    
    Returns:
        region_layers: {color: {'base': layer, 'light': layer, 'shade': layer}}の辞書
    """
    region_layers = {}
    
    # 画像サイズを統一（shadeをbase_colorのサイズにリサイズ）
    base_height, base_width = base_image_cv.shape[:2]
    shade_height, shade_width = shade_image_cv.shape[:2]
    
    if (shade_height != base_height) or (shade_width != base_width):
        print(f"[WithShade] Resizing shade image from {shade_width}x{shade_height} to {base_width}x{base_height}")
        shade_image_cv = cv2.resize(shade_image_cv, (base_width, base_height), interpolation=cv2.INTER_LINEAR)
    
    # shadeが確実にBGRA形式であることを確認
    if shade_image_cv.shape[2] == 3:
        alpha = np.ones((shade_image_cv.shape[0], shade_image_cv.shape[1], 1), dtype=np.uint8) * 255
        shade_image_cv = np.concatenate([shade_image_cv, alpha], axis=2)
    
    # baseも確実にBGRA形式に
    if base_image_cv.shape[2] == 3:
        alpha = np.ones((base_image_cv.shape[0], base_image_cv.shape[1], 1), dtype=np.uint8) * 255
        base_image_cv = np.concatenate([base_image_cv, alpha], axis=2)
    
    print(f"[WithShade] Creating shade layers for {len(color_regions)} regions...")
    
    for color_rgb, mask in color_regions.items():
        # この領域のレイヤーを初期化
        base_layer = np.zeros_like(base_image_cv, dtype=np.uint8)
        light_layer = np.zeros_like(base_image_cv, dtype=np.uint8)  # shade_image_cvではなくbase_image_cvのサイズで統一
        shade_layer = np.zeros_like(base_image_cv, dtype=np.uint8)
        
        # マスクを確実にuint8に変換
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        mask_bool = mask > 0
        
        # BGRからRGBに変換（比較用）
        base_rgb = cv2.cvtColor(base_image_cv[:, :, :3], cv2.COLOR_BGR2RGB)
        shade_rgb = cv2.cvtColor(shade_image_cv[:, :, :3], cv2.COLOR_BGR2RGB)
        
        # マスク内の各ピクセルを処理
        y_indices, x_indices = np.where(mask_bool)
        
        for y, x in zip(y_indices, x_indices):
            base_pixel = base_rgb[y, x]
            shade_pixel = shade_rgb[y, x]
            
            # ピクセルを分類
            pixel_type = classify_pixels_by_luminance(base_pixel, shade_pixel, luminance_threshold)
            
            if pixel_type == 'light':
                # lightレイヤーに追加（shade画像から）
                light_layer[y, x] = shade_image_cv[y, x]
            elif pixel_type == 'shade':
                # shadeレイヤーに追加（shade画像から）
                shade_layer[y, x] = shade_image_cv[y, x]
            else:
                # baseレイヤーに追加（base_color画像から）
                base_layer[y, x] = base_image_cv[y, x]
        
        # 各レイヤーに共通のアルファチャンネルを設定
        base_layer[:, :, 3] = mask
        light_layer[:, :, 3] = mask
        shade_layer[:, :, 3] = mask
        
        region_layers[color_rgb] = {
            'base': base_layer,
            'light': light_layer,
            'shade': shade_layer
        }
    
    print(f"[WithShade] Created {len(region_layers)} region groups with base/light/shade layers")
    return region_layers

def save_psd_with_shade(base_color_cv, shade_cv, line_art_cv, region_layers, 
                        output_dir, line_blend_mode):
    """
    シンプルな方式でPSDを保存（フォルダ化なし、LayerFlags不使用）
    """
    height, width = base_color_cv.shape[:2]
    
    # PSDファイルを作成
    psd = pytoshop.core.PsdFile(num_channels=3, height=height, width=width)
    
    # 背景レイヤーを追加
    background = np.ones((height, width, 4), dtype=np.uint8) * 255
    psd = add_psd(psd, background, "Background", enums.BlendMode.normal)
    
    # 各色領域のレイヤーを追加（base -> light -> shade の順）
    for idx, (color_rgb, layers_dict) in enumerate(region_layers.items()):
        region_name = f"R{color_rgb[0]}_G{color_rgb[1]}_B{color_rgb[2]}"
        
        # base layer
        psd = add_psd(psd, layers_dict['base'], f"{region_name}_base", enums.BlendMode.normal)
        
        # light layer  
        psd = add_psd(psd, layers_dict['light'], f"{region_name}_light", enums.BlendMode.normal)
        
        # shade layer
        psd = add_psd(psd, layers_dict['shade'], f"{region_name}_shade", enums.BlendMode.multiply)
    
    # 線画レイヤーを最上位に追加
    if line_art_cv.shape[2] == 4:
        alpha = line_art_cv[:, :, 3]
        rgb_max = np.max(line_art_cv[:, :, :3])
        
        if rgb_max < 10:  # ほぼ黒の線画
            line_intensity = 255 - alpha
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
    
    # ファイル名生成
    name = randomname(10)
    filename_only = f"output_rgb_shade_{name}.psd"
    filename = os.path.join(output_dir, filename_only)
    
    with open(filename, 'wb') as fd:
        psd.write(fd)
    
    return filename


class RGBLineArtDividerWithShade:
    """
    shade画像を追加入力として受け取り、光と影のレイヤーを生成する拡張版
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "line_art": ("IMAGE",),
                "base_color": ("IMAGE",),
                "shade": ("IMAGE",),  # 新規追加: shade画像
                "color_tolerance": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "luminance_threshold": ("FLOAT", {  # 新規追加: 輝度差の閾値
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "line_blend_mode": ([
                    "multiply", 
                    "normal", 
                    "darken", 
                    "overlay"
                ],),
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
    RETURN_NAMES = ("composite", "base_color", "region_count", "psd_filename")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"
    OUTPUT_NODE = False

    def execute(self, line_art, base_color, shade, color_tolerance, 
                luminance_threshold, line_blend_mode, max_colors):
        
        try:
            print("[RGBLineArtDividerWithShade] Starting execution...")
            
            # 画像をNumPy配列に変換
            print("[WithShade] Converting tensors to numpy arrays...")
            line_art_np = line_art.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
            base_color_np = base_color.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
            shade_np = shade.cpu().detach().numpy().__mul__(255.).astype(np.uint8)[0]
            
            print(f"[WithShade] Base color shape: {base_color_np.shape}")
            print(f"[WithShade] Shade shape: {shade_np.shape}")
            
            # PIL Imageに変換
            line_art_pil = Image.fromarray(line_art_np)
            base_color_pil = Image.fromarray(base_color_np)
            shade_pil = Image.fromarray(shade_np)
            
            # OpenCV形式（BGRA）に変換
            line_art_cv = pil2cv(line_art_pil)
            base_color_cv = pil2cv(base_color_pil)
            shade_cv = pil2cv(shade_pil)
            
            # BGRAに変換（アルファチャンネルを追加）
            if line_art_cv.shape[2] == 3:
                line_art_cv = cv2.cvtColor(line_art_cv, cv2.COLOR_BGR2BGRA)
            if base_color_cv.shape[2] == 3:
                base_color_cv = cv2.cvtColor(base_color_cv, cv2.COLOR_BGR2BGRA)
            if shade_cv.shape[2] == 3:
                shade_cv = cv2.cvtColor(shade_cv, cv2.COLOR_BGR2BGRA)
            
            # 色領域を抽出
            print(f"[WithShade] Extracting color regions...")
            color_regions = extract_color_regions_fast(
                base_color_cv, 
                tolerance=color_tolerance,
                max_colors=max_colors
            )
            print(f"[WithShade] Found {len(color_regions)} color regions")
            
            # 各領域でbase/light/shadeレイヤーを作成
            print("[WithShade] Creating shade layers...")
            region_layers = create_shade_layers(
                base_color_cv, 
                shade_cv, 
                color_regions, 
                luminance_threshold
            )
            
            # BlendModeの設定
            blend_mode_map = {
                "multiply": enums.BlendMode.multiply,
                "normal": enums.BlendMode.normal,
                "darken": enums.BlendMode.darken,
                "overlay": enums.BlendMode.overlay
            }
            
            # PSDファイルを保存（シンプルな方式）
            print("[WithShade] Saving PSD file...")
            filename = save_psd_with_shade(
                base_color_cv,
                shade_cv,
                line_art_cv,
                region_layers,
                output_dir,
                blend_mode_map[line_blend_mode]
            )
            
            print(f"[WithShade] PSD file saved: {filename}")
            
            # ログファイルに最新のPSDパスを保存
            log_file = os.path.join(output_dir, "fixableflow_savepath.log")
            try:
                with open(log_file, 'w') as f:
                    f.write(os.path.basename(filename))
                print(f"[WithShade] Log file updated: {log_file}")
            except Exception as e:
                print(f"[WithShade] Warning: Could not write log file: {e}")
            
            # コンポジット画像を作成（プレビュー用: base_color + shade + line_art）
            print("[WithShade] Creating composite image...")
            composite = shade_cv.copy()  # shadeを基準にする
            
            if line_blend_mode == "multiply":
                line_rgb = line_art_cv[:, :, :3].astype(np.float32) / 255.0
                composite_rgb = composite[:, :, :3].astype(np.float32) / 255.0
                composite[:, :, :3] = (composite_rgb * line_rgb * 255).astype(np.uint8)
            elif line_blend_mode == "normal":
                if line_art_cv.shape[2] == 4:
                    alpha = line_art_cv[:, :, 3:4].astype(np.float32) / 255.0
                    composite[:, :, :3] = (
                        line_art_cv[:, :, :3] * alpha + 
                        composite[:, :, :3] * (1 - alpha)
                    ).astype(np.uint8)
            
            print("[WithShade] Execution completed successfully!")
            
            filename_only = os.path.basename(filename)
            
            # 出力
            return (
                to_comfy_img(composite),
                to_comfy_img(base_color_cv),
                len(color_regions),
                filename_only
            )
        
        except Exception as e:
            print(f"[WithShade] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# ノードマッピング用
NODE_CLASS_MAPPINGS = {
    "RGBLineArtDividerWithShade": RGBLineArtDividerWithShade
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBLineArtDividerWithShade": "RGB Line Art Divider (With Shade)"
}
