# ComfyUI-fixableflow - Minimal Configuration
# Only nodes used in the workflow

# Extract Line Art ノードをインポート
from .extract_lineart_node import (
    NODE_CLASS_MAPPINGS as EXTRACT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as EXTRACT_DISPLAY_MAPPINGS,
)

# Morphology Operation ノードをインポート
from .morphology_node import (
    NODE_CLASS_MAPPINGS as MORPHOLOGY_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MORPHOLOGY_DISPLAY_MAPPINGS,
)

# Overlay Images ノードをインポート
from .overlay_images_node import (
    NODE_CLASS_MAPPINGS as OVERLAY_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as OVERLAY_DISPLAY_MAPPINGS,
)

# Shadow Extract ノードをインポート
from .shadow_extract_node import (
    NODE_CLASS_MAPPINGS as SHADOW_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SHADOW_DISPLAY_MAPPINGS,
)

# Highlight Extract ノードをインポート（ファイル名: extract_highlight_node.py）
from .extract_highlight_node import (
    NODE_CLASS_MAPPINGS as HIGHLIGHT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as HIGHLIGHT_DISPLAY_MAPPINGS,
)

# Simple PSD Stack ノードをインポート
from .simple_psd_stack_node import (
    NODE_CLASS_MAPPINGS as PSD_STACK_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PSD_STACK_DISPLAY_MAPPINGS,
)

# ノードマッピングを統合
NODE_CLASS_MAPPINGS = {
    **EXTRACT_MAPPINGS,          # Extract Line Art ノード
    **MORPHOLOGY_MAPPINGS,       # Morphology Operation ノード
    **OVERLAY_MAPPINGS,          # Overlay Images ノード
    **SHADOW_MAPPINGS,           # Shadow Extract ノード
    **HIGHLIGHT_MAPPINGS,        # Highlight Extract ノード
    **PSD_STACK_MAPPINGS,        # Simple PSD Stack ノード
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **EXTRACT_DISPLAY_MAPPINGS,      # Extract Line Art ノード
    **MORPHOLOGY_DISPLAY_MAPPINGS,   # Morphology Operation ノード
    **OVERLAY_DISPLAY_MAPPINGS,      # Overlay Images ノード
    **SHADOW_DISPLAY_MAPPINGS,       # Shadow Extract ノード
    **HIGHLIGHT_DISPLAY_MAPPINGS,    # Highlight Extract ノード
    **PSD_STACK_DISPLAY_MAPPINGS,    # Simple PSD Stack ノード
}

# Web拡張機能の自動読み込みのためのパス設定
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

