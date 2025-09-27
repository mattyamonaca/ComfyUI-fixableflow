"""
LayerDivider Simplified - カラーベース専用の整理版
セグメントマスク関連を完全に削除し、必要な機能のみを残した版
"""

from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_utils import save_psd, divide_folder
from .ldivider.ld_convertor import pil2cv, df2bgra
from .ldivider.ld_processor import get_base, get_normal_layer, get_composite_layer
from pytoshop.enums import BlendMode
from .extract_lineart_node import ExtractLineArtNode, ExtractLineArtAdvancedNode
from .fill_area_node import FillAreaNode, FillAreaAdvancedNode
from .fill_area_simple_node import FillAreaSimpleNode, FillAreaSimpleVisualizeNode
from .split_area_node import SplitAreaNode
from .fill_space_node import FillSpaceNode, FillSpaceAdvancedNode

import cv2

# パス設定
comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'
output_dir = f"{layer_divider_path}/output"
input_dir = f"{layer_divider_path}/input"

if not os.path.exists(f'{output_dir}'):
    os.makedirs(f'{output_dir}')


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


def to_comfy_imgs(np_imgs):
    """複数のNumPy配列をComfyUI形式の画像バッチに変換"""
    out_imgs = []
    for np_img in np_imgs:
        out_imgs.append(HWC3(np_img))
    out_imgs = np.stack(out_imgs)
    out_imgs = torch.from_numpy(out_imgs.astype(np.float32) / 255.)
    return out_imgs


class LayerDividerSimple:
    """
    カラーベース処理とレイヤー分割を1つのノードに統合
    シンプルな使用感を提供
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layer_mode": (["composite", "normal"],),
                "loops": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "init_cluster": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "ciede_threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "blur_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("original", "base", "bright", "shadow", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, image, layer_mode, loops, init_cluster, ciede_threshold, blur_size):
        # Step 1: カラーベース処理
        split_bg = False
        h_split = -1
        v_split = -1
        n_cluster = -1
        alpha = -1
        th_rate = 0

        img_batch_np = image.cpu().detach().numpy().__mul__(255.).astype(np.uint8)
        input_image = Image.fromarray(img_batch_np[0])
        
        image_cv = pil2cv(input_image)
        input_image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGBA)

        df = get_base(
            input_image_cv, loops, init_cluster, ciede_threshold, blur_size,
            h_split, v_split, n_cluster, alpha, th_rate, split_bg, False
        )

        # Step 2: レイヤー分割処理
        divide_mode = "color_base"
        
        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = (
                get_composite_layer(input_image_cv, df)
            )
            
            filename = save_psd(
                input_image_cv,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
                divide_mode
            )

        else:  # normal mode
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image_cv, df)
            
            filename = save_psd(
                input_image_cv,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
                divide_mode
            )

        print(f"PSD file saved: {filename}")
        divide_folder(filename, input_dir, layer_mode)

        # 出力画像の準備
        return (
            to_comfy_img(input_image_cv),
            to_comfy_imgs(base_layer_list),
            to_comfy_imgs(bright_layer_list),
            to_comfy_imgs(shadow_layer_list),
            filename
        )


class LayerDividerAdvanced:
    """
    詳細設定可能なカラーベース処理
    2段階処理が必要な場合に使用
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "loops": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "init_cluster": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "ciede_threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "blur_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("LD_DATA",)
    RETURN_NAMES = ("layer_data",)
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, image, loops, init_cluster, ciede_threshold, blur_size):
        split_bg = False
        h_split = -1
        v_split = -1
        n_cluster = -1
        alpha = -1
        th_rate = 0

        img_batch_np = image.cpu().detach().numpy().__mul__(255.).astype(np.uint8)
        input_image = Image.fromarray(img_batch_np[0])
        
        image_cv = pil2cv(input_image)
        input_image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGBA)

        df = get_base(
            input_image_cv, loops, init_cluster, ciede_threshold, blur_size,
            h_split, v_split, n_cluster, alpha, th_rate, split_bg, False
        )

        # データをタプルで返す（後でDivideノードで使用）
        return ((input_image_cv, df),)


class LayerDividerDivide:
    """
    レイヤー分割専用ノード
    LayerDividerAdvancedの出力を受け取る
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layer_data": ("LD_DATA",),
                "layer_mode": (["composite", "normal"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("original", "base", "bright", "shadow", "psd_path")
    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, layer_data, layer_mode):
        input_image_cv, df = layer_data
        divide_mode = "color_base"
        
        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = (
                get_composite_layer(input_image_cv, df)
            )
            
            filename = save_psd(
                input_image_cv,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
                divide_mode
            )

        else:  # normal mode
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image_cv, df)
            
            filename = save_psd(
                input_image_cv,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
                divide_mode
            )

        print(f"PSD file saved: {filename}")
        divide_folder(filename, input_dir, layer_mode)

        return (
            to_comfy_img(input_image_cv),
            to_comfy_imgs(base_layer_list),
            to_comfy_imgs(bright_layer_list),
            to_comfy_imgs(shadow_layer_list),
            filename
        )


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    # メインノード（統合版）
    "LayerDivider": LayerDividerSimple,
    
    # 詳細設定版（2段階処理）
    "LayerDivider - Advanced": LayerDividerAdvanced,
    "LayerDivider - Divide": LayerDividerDivide,
    
    # 補助ノード
    "LayerDivider - Extract Line Art": ExtractLineArtNode,
    "LayerDivider - Extract Line Art Advanced": ExtractLineArtAdvancedNode,
    "LayerDivider - Fill Area": FillAreaNode,
    "LayerDivider - Fill Area Advanced": FillAreaAdvancedNode,
    "LayerDivider - Fill Area Simple": FillAreaSimpleNode,
    "LayerDivider - Fill Area Simple Visualize": FillAreaSimpleVisualizeNode,
    "LayerDivider - Split Area": SplitAreaNode,
    "LayerDivider - Fill Space": FillSpaceNode,
    "LayerDivider - Fill Space Advanced": FillSpaceAdvancedNode
}

# 表示名マッピング
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerDivider": "Layer Divider (All-in-One)",
    "LayerDivider - Advanced": "Layer Divider Advanced",
    "LayerDivider - Divide": "Layer Divider Divide",
    "LayerDivider - Extract Line Art": "Extract Line Art",
    "LayerDivider - Extract Line Art Advanced": "Extract Line Art (Advanced)",
    "LayerDivider - Fill Area": "Fill Area",
    "LayerDivider - Fill Area Advanced": "Fill Area (Advanced)",
    "LayerDivider - Fill Area Simple": "Fill Area Simple",
    "LayerDivider - Fill Area Simple Visualize": "Fill Area Simple (Visualize)",
    "LayerDivider - Split Area": "Split Area",
    "LayerDivider - Fill Space": "Fill Space",
    "LayerDivider - Fill Space Advanced": "Fill Space (Advanced)"
}
