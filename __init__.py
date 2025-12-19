# RGB Line Art Dividerを使用 - Updated to force reload
from .rgb_line_art_divider import RGB_NODE_CLASS_MAPPINGS, RGB_NODE_DISPLAY_NAME_MAPPINGS

# RGBLineArtDividerFastの最新版をインポート
from .rgb_line_art_divider_fast import RGBLineArtDividerFast

from .layer_divider_simplified import NODE_CLASS_MAPPINGS as LEGACY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LEGACY_DISPLAY_MAPPINGS

# デバッグノードをインポート
from .fill_space_debug_node import NODE_CLASS_MAPPINGS as DEBUG_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DEBUG_DISPLAY_MAPPINGS

# 拡張版塗りつぶしノードをインポート
from .fill_area_enhanced_node import NODE_CLASS_MAPPINGS as ENHANCED_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ENHANCED_DISPLAY_MAPPINGS

# Fill Space V2ノードをインポート
from .fill_space_node_v2 import NODE_CLASS_MAPPINGS as FILL_SPACE_V2_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FILL_SPACE_V2_DISPLAY_MAPPINGS

# モルフォロジー演算ノードをインポート
from .morphology_node import NODE_CLASS_MAPPINGS as MORPHOLOGY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MORPHOLOGY_DISPLAY_MAPPINGS

# RGB Line Art Divider with Shade ノードをインポート
from .rgb_line_art_divider_with_shade import NODE_CLASS_MAPPINGS as SHADE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SHADE_DISPLAY_MAPPINGS

# Overlay Images ノードをインポート
from .overlay_images_node import NODE_CLASS_MAPPINGS as OVERLAY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OVERLAY_DISPLAY_MAPPINGS

# ノードマッピングを統合
NODE_CLASS_MAPPINGS = {
    **RGB_NODE_CLASS_MAPPINGS,  # 新しいRGBノードを優先
    "RGBLineArtDividerFast": RGBLineArtDividerFast,  # 高速版を追加（更新版）
    **LEGACY_MAPPINGS,  # 既存のノードも保持
    **DEBUG_MAPPINGS,  # デバッグノードを追加
    **ENHANCED_MAPPINGS,  # 拡張版塗りつぶしノードを追加
    **FILL_SPACE_V2_MAPPINGS,  # Fill Space V2ノードを追加
    **MORPHOLOGY_MAPPINGS,  # モルフォロジー演算ノードを追加
    **SHADE_MAPPINGS,  # RGB Line Art Divider with Shade ノードを追加
    **OVERLAY_MAPPINGS  # Overlay Images ノードを追加
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RGB_NODE_DISPLAY_NAME_MAPPINGS,  # 新しいRGBノードを優先
    "RGBLineArtDividerFast": "RGB Line Art Divider (Fast)",  # 高速版を追加
    **LEGACY_DISPLAY_MAPPINGS,  # 既存のノードも保持
    **DEBUG_DISPLAY_MAPPINGS,  # デバッグノードを追加
    **ENHANCED_DISPLAY_MAPPINGS,  # 拡張版塗りつぶしノードを追加
    **FILL_SPACE_V2_DISPLAY_MAPPINGS,  # Fill Space V2ノードを追加
    **MORPHOLOGY_DISPLAY_MAPPINGS,  # モルフォロジー演算ノードを追加
    **SHADE_DISPLAY_MAPPINGS,  # RGB Line Art Divider with Shade ノードを追加
    **OVERLAY_DISPLAY_MAPPINGS  # Overlay Images ノードを追加
}

# Web拡張機能の自動読み込みのためのパス設定
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
