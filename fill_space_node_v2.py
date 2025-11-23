"""
Fill Space Node (Nearest Color BFS version) for ComfyUI
線画の部分を、周囲の最も近いベタ色で埋めて「線の下の色」を推定するノード。

特徴:
- 入力: binary_image（線画マスク）、flat_image（バケツ塗りベタ画像）
- binary_image の線部分を、flat_image の周囲ピクセルの色で埋めるだけ
- K-means / クラスタリングは一切行わない（パレットを壊さない）
- 4近傍の multi-source BFS で最近傍の非線画ピクセルを求める（O(H*W)）
"""

import os
from collections import deque

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
import folder_paths

# 出力パス（使わなくても良いが、一応元コードに合わせて用意）
comfy_path = os.path.dirname(folder_paths.__file__)
layer_divider_path = f"{comfy_path}/custom_nodes/ComfyUI-fixableflow"
output_dir = f"{layer_divider_path}/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ---------- 共通ユーティリティ ----------

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """ComfyUIのIMAGEテンソルを PIL Image に変換"""
    image_np = tensor.cpu().detach().numpy()
    if image_np.ndim == 4:
        image_np = image_np[0]  # バッチ先頭だけ使う
    image_np = (image_np * 255).astype(np.uint8)

    if image_np.shape[2] == 3:
        mode = "RGB"
    elif image_np.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "L"

    return Image.fromarray(image_np, mode=mode)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image を ComfyUIのIMAGEテンソル形式に変換"""
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)


def create_before_after_preview(original: Image.Image, processed: Image.Image) -> Image.Image:
    """処理前後の比較画像を左右に並べて作成"""
    width, height = original.width, original.height
    comparison = Image.new("RGB", (width * 2, height))

    comparison.paste(original.convert("RGB"), (0, 0))
    comparison.paste(processed.convert("RGB"), (width, 0))

    draw = ImageDraw.Draw(comparison)
    # 真ん中の区切り線
    draw.line([(width, 0), (width, height)], fill=(255, 0, 0), width=2)
    # ラベル
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((width + 10, 10), "Processed", fill=(255, 255, 255))

    return comparison


# ---------- 線の下を埋める中核ロジック ----------

def fill_lines_with_nearest_color(binary_pil: Image.Image,
                                  flat_pil: Image.Image,
                                  invert_binary: bool = True) -> Image.Image:
    """
    binary_pil で指定された線画部分を、flat_pil の最も近い非線画ピクセルの色で埋める。

    想定する binary_pil:
    - 通常: 線が黒(0)、それ以外が白(255)
    - invert_binary=True の場合、内部で白黒を反転してから処理する

    処理フロー:
    1. binary を L グレースケールにして、必要なら反転
    2. 「線ピクセル(line_mask)」と「非線ピクセル(non_line)」を二値マスクとして取得
    3. non_line から multi-source BFS を行い、各ピクセルに「最寄り non_line 座標」を伝播
    4. line_mask 上のピクセルは、その最寄り non_line 座標の色で埋める
    """
    # 1. バイナリ画像をグレースケールに変換
    if binary_pil.mode != "L":
        binary_gray = binary_pil.convert("L")
    else:
        binary_gray = binary_pil

    # 必要に応じて反転
    if invert_binary:
        binary_gray = ImageOps.invert(binary_gray)

    mask_array = np.array(binary_gray)  # 0〜255

    # ここでは「線 = 255近辺」を line_mask とする
    # しきい値は少し下げて、アンチエイリアスのグレーも線として扱う
    line_mask = mask_array >= 200  # True: 線／埋める対象
    non_line_mask = ~line_mask     # True: ベタ塗り領域（色のソース）

    height, width = line_mask.shape

    # 2. ベース画像（flat）の RGB 配列
    base_array = np.array(flat_pil.convert("RGB"))
    output_array = base_array.copy()

    # 全ピクセルが線の場合は何もできないので、そのまま返す
    if not np.any(non_line_mask):
        return flat_pil

    # 3. multi-source BFS で最近傍 non_line の座標を埋める
    # 最近傍の非線画ピクセルの座標を格納する配列
    nearest_y = -np.ones((height, width), dtype=np.int32)
    nearest_x = -np.ones((height, width), dtype=np.int32)

    q = deque()

    # non_line の座標を全て BFS の起点として追加
    ys, xs = np.where(non_line_mask)
    for y, x in zip(ys, xs):
        nearest_y[y, x] = y
        nearest_x[y, x] = x
        q.append((y, x))

    # 4近傍
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS: 各ピクセルに「一番近い non_line の座標」を伝播させる
    while q:
        cy, cx = q.popleft()
        ny0, nx0 = nearest_y[cy, cx], nearest_x[cy, cx]

        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < height and 0 <= nx < width:
                # まだ最近傍が決まっていないピクセルにのみ伝播
                if nearest_y[ny, nx] == -1:
                    nearest_y[ny, nx] = ny0
                    nearest_x[ny, nx] = nx0
                    q.append((ny, nx))

    # 4. line_mask 上のピクセルを最近傍 non_line の色で埋める
    line_ys, line_xs = np.where(line_mask)
    src_ys = nearest_y[line_ys, line_xs]
    src_xs = nearest_x[line_ys, line_xs]

    # 念のため valid な位置だけ使う
    valid = (src_ys >= 0) & (src_xs >= 0)
    line_ys = line_ys[valid]
    line_xs = line_xs[valid]
    src_ys = src_ys[valid]
    src_xs = src_xs[valid]

    output_array[line_ys, line_xs] = base_array[src_ys, src_xs]

    return Image.fromarray(output_array.astype(np.uint8))


# ---------- ComfyUI ノード本体 ----------

class FillSpaceNearestNode:
    """
    線画の下のピクセルを周囲の最も近いベタ色で埋めるノード（最近傍 BFS 版）

    使い方:
        binary_image : 線画マスク（通常は「線=黒, それ以外=白」の画像）
        flat_image   : 線なしバケツ塗り画像（ベタ塗り）
        invert_binary: True の場合、binary_image を先に反転してから処理
                       （線=白, それ以外=黒 の形式で扱いたい場合用）

    出力:
        filled_image : 線がベタ色で埋められた画像（RGB）
        preview      : flat_image と filled_image を横に並べた比較画像
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),  # 線画マスク
                "flat_image": ("IMAGE",),    # バケツ塗りベタ画像
                "invert_binary": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "display_label": "Invert Binary Image",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "preview")

    FUNCTION = "execute"
    CATEGORY = "LayerDivider"

    def execute(self, binary_image, flat_image, invert_binary=True):
        # テンソル → PIL
        binary_pil = tensor_to_pil(binary_image)
        flat_pil = tensor_to_pil(flat_image)

        # 線の下を最近傍色で埋める
        filled_pil = fill_lines_with_nearest_color(
            binary_pil,
            flat_pil,
            invert_binary=invert_binary,
        )

        # プレビュー作成
        preview_pil = create_before_after_preview(flat_pil, filled_pil)

        # PIL → テンソル
        filled_tensor = pil_to_tensor(filled_pil)
        preview_tensor = pil_to_tensor(preview_pil)

        return (filled_tensor, preview_tensor)


# ノードクラスのマッピング
NODE_CLASS_MAPPINGS = {
    "FillSpaceNearest": FillSpaceNearestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillSpaceNearest": "Fill Space (Nearest Color BFS)",
}
