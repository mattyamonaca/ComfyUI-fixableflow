"""
Split Area Node for ComfyUI
線画の線を太くして領域を分割し、各領域を異なる色で塗り分けるノード
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label, binary_fill_holes, binary_closing, binary_opening, binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt
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
    
    Args:
        base_image: ベース画像
        lineart: 線画画像
        thickness: 線の太さ
        new_color: 新しい色 (R, G, B)
    
    Returns:
        合成された画像
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


def smooth_regions(labeled_array, num_features, smoothing_iterations=3, 
                   fill_holes=True, min_area=50, connect_fragments=True):
    """
    検出された領域をスムース化する
    
    Args:
        labeled_array: ラベル付けされた領域配列
        num_features: 領域数
        smoothing_iterations: スムージング処理の反復回数
        fill_holes: 穴埋め処理を行うか
        min_area: 最小領域サイズ（これより小さい領域は削除）
        connect_fragments: 分断された領域を接続するか
    
    Returns:
        スムース化されたラベル配列
    """
    print(f"[DEBUG] Starting smooth_regions: num_features={num_features}, iterations={smoothing_iterations}")
    # 元の配列をコピーして作業（元のラベルを保持）
    smoothed_array = np.copy(labeled_array)
    processed_regions = 0
    
    for region_id in range(1, num_features + 1):
        # 各領域を個別に処理
        region_mask = (labeled_array == region_id)
        original_pixels = np.sum(region_mask)
        
        # 領域が存在しない場合はスキップ
        if original_pixels == 0:
            print(f"[DEBUG] Region {region_id}: No pixels found")
            continue
        
        # 小さすぎる領域でも、元のままで保持（削除しない）
        if original_pixels < min_area:
            print(f"[DEBUG] Region {region_id}: Small region kept as-is ({original_pixels} pixels)")
            # 元のラベルをそのまま保持
            continue
        
        print(f"[DEBUG] Processing region {region_id}: original pixels = {original_pixels}")
        
        # uint8に変換（0 or 1）
        region_binary = region_mask.astype(np.uint8)
        
        # 1. 穴埋め処理を最初に行う（内部の穴を埋める）
        if fill_holes:
            try:
                region_binary = binary_fill_holes(region_binary).astype(np.uint8)
            except:
                print(f"[DEBUG] Region {region_id}: Failed fill_holes, using original")
                region_binary = region_mask.astype(np.uint8)
        
        # 2. メディアンフィルタでノイズ除去（小さいカーネルサイズ）
        if region_binary.max() > 0:  # 領域が残っている場合のみ
            region_binary = cv2.medianBlur(region_binary, 3)
        
        # 3. Closing処理で小さな隙間を埋める
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        region_binary = cv2.morphologyEx(region_binary, cv2.MORPH_CLOSE, kernel_small)
        
        # 4. 軽いスムージング（反復回数を制限）
        for i in range(min(smoothing_iterations, 3)):  # 最大3回に制限
            if region_binary.max() == 0:  # 領域が消えた場合は処理を中止
                print(f"[DEBUG] Region {region_id}: Lost during smoothing iteration {i}")
                region_binary = region_mask.astype(np.uint8)
                break
            
            # ガウシアンブラーは軽めに（カーネルサイズを小さく）
            blurred = cv2.GaussianBlur(region_binary.astype(np.float32), (5, 5), 1.0)
            # 閾値を調整（0.3でより多くのピクセルを保持）
            region_binary = (blurred > 0.3).astype(np.uint8)
        
        # 5. 最終的な領域チェック
        final_pixels = np.sum(region_binary)
        if final_pixels == 0:
            print(f"[DEBUG] Region {region_id}: Lost during processing, restoring original")
            # 元の領域を復元
            smoothed_array[labeled_array == region_id] = region_id
        else:
            # スムース化された領域で更新
            # まず該当領域をクリア
            smoothed_array[labeled_array == region_id] = 0
            # 新しい領域を設定
            smoothed_array[region_binary == 1] = region_id
            processed_regions += 1
            print(f"[DEBUG] Region {region_id}: Smoothed successfully (final pixels = {final_pixels})")
    
    print(f"[DEBUG] Processed {processed_regions}/{num_features} regions")
    print(f"[DEBUG] Final smoothed_array unique values: {np.unique(smoothed_array)}")
    return smoothed_array


def process_split_area(lineart_image, fill_image=None, thickness=1, threshold=128, 
                       use_fill_colors=False, random_seed=None, 
                       enable_smoothing=True, smoothing_iterations=3,
                       fill_holes=True, min_area=50):
    """
    線画を分割して各領域を色分けする処理
    
    Args:
        lineart_image: 線画画像
        fill_image: 塗り画像（オプション）
        thickness: 線の太さ
        threshold: 二値化の閾値
        use_fill_colors: 塗り画像の色を使用するか
        random_seed: ランダムシード
        enable_smoothing: 領域のスムージングを有効にするか
        smoothing_iterations: スムージング処理の反復回数
        fill_holes: 穴埋め処理を行うか
        min_area: 最小領域サイズ
    
    Returns:
        処理済みの画像、バイナリ画像、スムース化されたラベル配列
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
    print(f"[DEBUG] Before smoothing: num_features={num_features}, unique labels={np.unique(labeled_array)}")
    
    # 領域のスムージング処理
    if enable_smoothing:
        print(f"[DEBUG] Smoothing enabled with iterations={smoothing_iterations}, fill_holes={fill_holes}, min_area={min_area}")
        labeled_array_smoothed = smooth_regions(
            labeled_array, 
            num_features, 
            smoothing_iterations=smoothing_iterations,
            fill_holes=fill_holes,
            min_area=min_area,
            connect_fragments=True
        )
        # スムース化後も元のラベルが残っていない場合は元の配列を使用
        if np.max(labeled_array_smoothed) > 0:
            labeled_array = labeled_array_smoothed
        else:
            print("[DEBUG] Warning: Smoothing resulted in empty array, using original")
    
    print(f"[DEBUG] After smoothing: unique labels={np.unique(labeled_array)}")
    
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


class SplitAreaNode:
    """
    線画の領域を分割して色分けするノード
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
                    "display_label": "Enable Region Smoothing"
                }),
                "smoothing_iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Smoothing Iterations"
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Fill Holes in Regions"
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
                enable_smoothing=True, smoothing_iterations=3,
                fill_holes=True, min_area=50,
                fill_image=None, use_fill_colors=False, random_seed=-1):
        """
        線画領域分割処理を実行
        
        Args:
            lineart_image: 線画画像のテンソル
            thickness: 線の太さ
            threshold: 二値化閾値
            fill_image: 塗り画像（オプション）
            use_fill_colors: 塗り画像の色を使用
            random_seed: ランダムシード
        
        Returns:
            split_image: 色分けされた画像
            binary_image: 二値化画像
            region_mask: 領域マスク
        """
        
        # テンソルをPIL Imageに変換
        lineart_pil = tensor_to_pil(lineart_image)
        fill_pil = tensor_to_pil(fill_image) if fill_image is not None else None
        
        # ランダムシードの設定
        seed = None if random_seed == -1 else random_seed
        
        # 処理実行
        colored_image, binary_image, labeled_array = process_split_area(
            lineart_pil,
            fill_pil,
            thickness,
            threshold,
            use_fill_colors,
            seed,
            enable_smoothing,
            smoothing_iterations,
            fill_holes,
            min_area
        )
        region_mask = (labeled_array > 0).astype(np.float32)
        region_mask = np.expand_dims(region_mask, axis=0)
        region_mask = np.expand_dims(region_mask, axis=-1)
        region_mask_tensor = torch.from_numpy(region_mask)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(colored_image)
        binary_tensor = pil_to_tensor(binary_image.convert('RGB'))
        
        return (output_tensor, binary_tensor, region_mask_tensor)


class SplitAreaAdvancedNode:
    """
    高度な領域分割ノード
    より詳細なコントロールと可視化機能付き
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
                    "max": 20,
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
                "dilation_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Dilation Iterations"
                }),
                "color_mode": (["random", "gradient", "pastel", "vivid", "from_fill"],),
                "output_mode": (["colored", "regions", "overlay", "comparison"],),
            },
            "optional": {
                "fill_image": ("IMAGE",),
                "random_seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "display_label": "Random Seed (-1 for random)"
                }),
                "preserve_lines": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Preserve Original Lines"
                }),
                "enable_smoothing": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Enable Region Smoothing"
                }),
                "smoothing_strength": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "display": "slider",
                    "display_label": "Smoothing Strength"
                }),
                "fill_region_holes": ("BOOLEAN", {
                    "default": True,
                    "display_label": "Fill Region Holes"
                }),
                "min_region_size": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "slider",
                    "display_label": "Minimum Region Size"
                }),
                "line_color": ("STRING", {
                    "default": "0,0,0",
                    "display_label": "Line Color (R,G,B)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "INT")
    RETURN_NAMES = ("split_image", "visualization", "binary_image", "region_mask", "num_regions")
    
    FUNCTION = "execute"
    
    CATEGORY = "LayerDivider"
    
    def execute(self, lineart_image, thickness=1, threshold=128, dilation_iterations=1,
                color_mode="random", output_mode="colored", fill_image=None,
                random_seed=-1, preserve_lines=True, line_color="0,0,0",
                enable_smoothing=True, smoothing_strength=5, 
                fill_region_holes=True, min_region_size=100):
        """
        高度な線画領域分割処理
        
        Args:
            lineart_image: 線画画像
            thickness: 線の太さ
            threshold: 二値化閾値
            dilation_iterations: 膨張処理の回数
            color_mode: 色付けモード
            output_mode: 出力モード
            fill_image: 塗り画像（オプション）
            random_seed: ランダムシード
            preserve_lines: 元の線を保持
            line_color: 線の色
        
        Returns:
            split_image: 色分けされた画像
            visualization: 可視化画像
            binary_image: 二値化画像
            region_mask: 領域マスク
            num_regions: 領域数
        """
        
        # テンソルをPIL Imageに変換
        lineart_pil = tensor_to_pil(lineart_image)
        fill_pil = tensor_to_pil(fill_image) if fill_image is not None else None
        
        # ランダムシードの設定
        if random_seed != -1:
            np.random.seed(random_seed)
        
        # 線の色を解析
        try:
            line_rgb = tuple(map(int, line_color.split(',')))
        except:
            line_rgb = (0, 0, 0)
        
        # 線を太くして処理
        new_color = tuple(np.random.randint(0, 256, size=3))
        split_area_line = thicken_and_recolor_lines(
            lineart_pil, lineart_pil, 
            thickness=thickness, 
            new_color=new_color
        )
        
        # 太くした線のみを抽出
        tmp = get_binary_image(split_area_line, new_color)
        
        # グレースケール化と二値化
        gray_image = ImageOps.grayscale(tmp)
        binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, '1')
        
        # 追加の膨張処理
        if dilation_iterations > 1:
            binary_cv = np.array(binary_image, dtype=np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            binary_cv = cv2.dilate(binary_cv, kernel, iterations=dilation_iterations-1)
            binary_image = Image.fromarray(binary_cv)
        
        # 輪郭検出
        labeled_array, num_features = find_contours(binary_image)
        
        # 領域のスムージング処理（Advanced版用の強化版）
        if enable_smoothing:
            labeled_array = smooth_regions_advanced(
                labeled_array, 
                num_features, 
                smoothing_strength=smoothing_strength,
                fill_holes=fill_region_holes,
                min_area=min_region_size
            )
        
        # カラーパレットの生成
        colors = generate_color_palette(num_features, color_mode, fill_pil)
        
        # 出力画像の作成
        split_image = np.array(lineart_pil.convert("RGBA"))
        
        # 各領域を色付け
        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            if color_mode == "from_fill" and fill_pil is not None:
                # 塗り画像から色を取得
                color = get_color_from_fill(fill_pil, region_mask)
            else:
                color = colors[region_id - 1]
            split_image[region_mask] = color
        
        # 元の線を保持
        if preserve_lines:
            lineart_array = np.array(lineart_pil.convert("RGBA"))
            # 線の部分を検出（アルファチャンネルまたは暗い部分）
            if lineart_array.shape[2] == 4:
                line_mask = lineart_array[:, :, 3] > 128
            else:
                gray_lineart = cv2.cvtColor(lineart_array[:, :, :3], cv2.COLOR_RGB2GRAY)
                line_mask = gray_lineart < 128
            
            # 線を上書き
            split_image[line_mask] = list(line_rgb) + [255]
        
        colored_image = Image.fromarray(split_image.astype(np.uint8))
        
        # 可視化の作成
        if output_mode == "regions":
            visualization = visualize_regions(labeled_array, num_features)
        elif output_mode == "overlay":
            visualization = create_overlay(lineart_pil, colored_image)
        elif output_mode == "comparison":
            visualization = create_split_comparison(lineart_pil, colored_image, binary_image)
        else:  # "colored"
            visualization = colored_image
        
        # 領域マスクの作成
        region_mask = (labeled_array > 0).astype(np.float32)
        region_mask = np.expand_dims(region_mask, axis=0)
        region_mask = np.expand_dims(region_mask, axis=-1)
        region_mask_tensor = torch.from_numpy(region_mask)
        
        # ComfyUIのテンソル形式に変換
        output_tensor = pil_to_tensor(colored_image)
        visualization_tensor = pil_to_tensor(visualization)
        binary_tensor = pil_to_tensor(binary_image.convert('RGB'))
        
        # 領域数をIntとして返す
        num_regions_int = int(num_features)
        
        return (output_tensor, visualization_tensor, binary_tensor, 
                region_mask_tensor, num_regions_int)


def smooth_regions_advanced(labeled_array, num_features, smoothing_strength=5, 
                           fill_holes=True, min_area=100):
    """
    Advanced版用の強化されたスムージング処理
    
    Args:
        labeled_array: ラベル付けされた領域配列
        num_features: 領域数
        smoothing_strength: スムージングの強度
        fill_holes: 穴埋め処理を行うか
        min_area: 最小領域サイズ
    
    Returns:
        スムース化されたラベル配列
    """
    smoothed_array = np.zeros_like(labeled_array)
    
    # カーネルサイズをスムージング強度に応じて調整
    kernel_sizes = {
        'small': max(3, smoothing_strength // 2),
        'medium': smoothing_strength,
        'large': smoothing_strength + 2
    }
    
    for region_id in range(1, num_features + 1):
        # 各領域を個別に処理
        region_mask = (labeled_array == region_id).astype(np.uint8)
        
        # 小さすぎる領域はスキップ
        if np.sum(region_mask) < min_area:
            continue
        
        # 1. 初期のノイズ除去
        kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_sizes['small'], kernel_sizes['small'])
        )
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 2. メディアンフィルタで飛び点を除去
        region_mask = cv2.medianBlur(region_mask, kernel_sizes['small'])
        
        # 3. 穴埋め処理（第1段階）
        if fill_holes:
            region_mask = binary_fill_holes(region_mask).astype(np.uint8)
        
        # 4. 距離変換を使用した領域の拡張と収縮
        dist_transform = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
        
        # 閾値を動的に設定（領域の大きさに応じて）
        threshold = np.percentile(dist_transform[dist_transform > 0], 20) if np.any(dist_transform > 0) else 1
        region_mask = (dist_transform > threshold * 0.3).astype(np.uint8)
        
        # 5. モルフォロジー勾配でエッジを滑らかに
        kernel_medium = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_sizes['medium'], kernel_sizes['medium'])
        )
        gradient = cv2.morphologyEx(region_mask, cv2.MORPH_GRADIENT, kernel_medium)
        
        # 6. クロージング処理で隙間を埋める
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # 7. バイラテラルフィルタでエッジを保持しながらスムージング
        region_float = region_mask.astype(np.float32)
        for _ in range(smoothing_strength // 3):
            region_float = cv2.bilateralFilter(region_float, 9, 75, 75)
        
        # 8. 適応的閾値処理で二値化
        region_mask = (region_float > 0.5).astype(np.uint8)
        
        # 9. 最終的なモルフォロジー処理
        kernel_large = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_sizes['large'], kernel_sizes['large'])
        )
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_large)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 10. 最終的な穴埋め
        if fill_holes:
            region_mask = binary_fill_holes(region_mask).astype(np.uint8)
            
            # 内部の小さな穴も埋める
            contours, _ = cv2.findContours(1 - region_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    cv2.drawContours(region_mask, [contour], -1, 1, -1)
        
        # 11. エッジの最終スムージング
        region_mask = cv2.GaussianBlur(region_mask.astype(np.float32), 
                                       (kernel_sizes['large'], kernel_sizes['large']), 
                                       smoothing_strength / 3)
        region_mask = (region_mask > 0.5).astype(np.uint8)
        
        # スムース化された領域を結果に追加
        smoothed_array[region_mask == 1] = region_id
    
    return smoothed_array


def generate_color_palette(num_colors, mode="random", fill_image=None):
    """カラーパレットを生成"""
    colors = []
    
    if mode == "gradient":
        # グラデーションカラー
        for i in range(num_colors):
            hue = (i / num_colors) * 360
            rgb = hsv_to_rgb(hue, 0.7, 0.9)
            colors.append(list(rgb) + [255])
    
    elif mode == "pastel":
        # パステルカラー
        for i in range(num_colors):
            hue = np.random.randint(0, 360)
            rgb = hsv_to_rgb(hue, 0.3, 0.95)
            colors.append(list(rgb) + [255])
    
    elif mode == "vivid":
        # ビビッドカラー
        for i in range(num_colors):
            hue = np.random.randint(0, 360)
            rgb = hsv_to_rgb(hue, 0.9, 0.95)
            colors.append(list(rgb) + [255])
    
    else:  # "random"
        # ランダムカラー
        for i in range(num_colors):
            color = np.random.randint(0, 256, size=3).tolist() + [255]
            colors.append(color)
    
    return colors


def hsv_to_rgb(h, s, v):
    """HSVからRGBへ変換"""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
    return (int(r*255), int(g*255), int(b*255))


def get_color_from_fill(fill_image, region_mask):
    """塗り画像から領域の代表色を取得"""
    fill_array = np.array(fill_image.convert("RGBA"))
    
    # 領域の中心点を見つける
    y_coords, x_coords = np.where(region_mask)
    if len(y_coords) > 0:
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # 範囲内チェック
        if center_y < fill_array.shape[0] and center_x < fill_array.shape[1]:
            color = fill_array[center_y, center_x].tolist()
            if len(color) == 3:
                color.append(255)
            return color
    
    # デフォルト色
    return np.random.randint(0, 256, size=3).tolist() + [255]


def visualize_regions(labeled_array, num_features):
    """領域の可視化（各領域を異なる色で表示）"""
    # カラーマップの作成
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_features + 1, 3))
    colors[0] = [0, 0, 0]  # 背景は黒
    
    # ラベル画像をカラー画像に変換
    result = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    for label_id in range(num_features + 1):
        mask = labeled_array == label_id
        result[mask] = colors[label_id]
    
    return Image.fromarray(result)


def create_overlay(original, colored):
    """元画像と色分け画像のオーバーレイ"""
    original_rgba = original.convert("RGBA")
    colored_rgba = colored.convert("RGBA")
    
    # 半透明で重ねる
    colored_rgba.putalpha(128)
    overlay = Image.alpha_composite(original_rgba, colored_rgba)
    
    return overlay


def create_split_comparison(original, colored, binary):
    """3つの画像を並べて比較"""
    width = original.width
    height = original.height
    
    # 3つの画像を横に並べる
    comparison = Image.new('RGB', (width * 3, height))
    comparison.paste(original.convert('RGB'), (0, 0))
    comparison.paste(binary.convert('RGB'), (width, 0))
    comparison.paste(colored.convert('RGB'), (width * 2, 0))
    
    return comparison


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "SplitArea": SplitAreaNode,
    "SplitArea Advanced": SplitAreaAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitArea": "Split Area",
    "SplitArea Advanced": "Split Area (Advanced)",
}
