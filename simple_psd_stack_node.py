"""
Simple PSD Layer Stack Node
複数の画像を受け取り、順番に重ねてPSDファイルとして出力するシンプルなノード
"""

import torch
import numpy as np
import os
from datetime import datetime
import folder_paths
from pytoshop.user import nested_layers
from pytoshop import enums


class SimplePSDStackNode:
    """
    複数の画像を受け取り、順番にレイヤーとして重ねてPSD出力するノード
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 画像のバッチ
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "layered",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("psd_path",)
    FUNCTION = "create_psd"
    CATEGORY = "image/psd"
    OUTPUT_NODE = True
    
    def create_psd(self, images, filename_prefix="layered"):
        """
        複数の画像をレイヤーとして重ねたPSDファイルを作成
        
        Args:
            images: 画像のバッチ (ComfyUI形式: torch.Tensor [B, H, W, C])
            filename_prefix: ファイル名のプレフィックス
        
        Returns:
            PSDファイルのパス
        """
        # 出力ディレクトリの設定
        comfy_path = os.path.dirname(folder_paths.__file__)
        output_dir = os.path.join(comfy_path, 'output')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.psd")
        
        # 画像の情報を取得
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        
        print(f"Creating PSD with {batch_size} layers, size: {width}x{height}")
        print(f"Layer order: Index 0 (bottom) → Index {batch_size-1} (top)")
        
        # レイヤーリストを作成
        layers_list = []
        
        # 各画像をレイヤーとして追加（下から上へ）
        # バッチの index 0 = 一番下のレイヤー（背景）
        # バッチの index N-1 = 一番上のレイヤー（前景）
        for i in range(batch_size):
            # テンソルをNumPy配列に変換（0-255のuint8）
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            
            # チャンネル数を確認
            has_alpha = img_np.shape[2] == 4
            
            # RGBチャンネルを分離
            if has_alpha:
                channels = [
                    img_np[:, :, 0],  # R
                    img_np[:, :, 1],  # G
                    img_np[:, :, 2],  # B
                    img_np[:, :, 3]   # A
                ]
            else:
                channels = [
                    img_np[:, :, 0],  # R
                    img_np[:, :, 1],  # G
                    img_np[:, :, 2]   # B
                ]
            
            # レイヤー名を生成（下から上への順序を明示）
            layer_name = f"Layer {i + 1} (Index {i})"
            
            # レイヤーを作成
            layer = nested_layers.Image(
                name=layer_name,
                visible=True,
                opacity=255,
                group_id=0,
                blend_mode=enums.BlendMode.normal,
                top=0,
                left=0,
                channels=channels,
                metadata=None,
                layer_color=0,
                color_mode=None
            )
            
            layers_list.append(layer)
        
        # PSDファイルを保存
        with open(filename, 'wb') as f:
            nested_layers.export(
                layers_list,
                f,
                width=width,
                height=height,
                depth=8,
                color_mode=enums.ColorMode.rgb
            )
        
        # ログファイルにパスを保存（ダウンロードボタン用）
        log_path = os.path.join(output_dir, 'simple_psd_stack_savepath.log')
        with open(log_path, 'w') as log_file:
            # ファイル名のみを保存（パスなし）
            log_file.write(os.path.basename(filename))
        
        print(f"PSD file saved: {filename}")
        print(f"Total layers: {batch_size}")
        print(f"Log file updated: {log_path}")
        
        return (filename,)


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    "SimplePSDStackNode": SimplePSDStackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplePSDStackNode": "Simple PSD Stack"
}
