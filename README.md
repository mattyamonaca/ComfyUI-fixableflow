[JP](README.md)
# FixableFlow
![G8k8fr_bMAMe68E](https://github.com/user-attachments/assets/b1ff33c1-0ceb-42e4-94d5-d5ae1d05d705)


FixableFlowは編集可能なイラスト生成AIを目指して作成された、画像生成AIを複数組み合わせて作られるワークフローの枠組みです。
人がイラストを制作する工程を参考にし、工程毎に画像を生成することで生成された画像（イラスト）を任意の工程で編集可能にします。
具体的には、イラストの制作工程を①線画、②バケツ塗り、③1影、④2影、⑤ハイライト、⑥仕上げ効果という6つ工程に分離し、各工程毎に画像を出力する事を目指します。
v0.1では①~③及び⑤の工程に対応しており、順次④、⑥の工程についても対応していく予定です。

また各工程後に出力された画像は、重ね合わせることで一枚のイラストとして出力できるように設計されており、PSDファイル形式での出力が可能となっています。
<img width="933" height="579" alt="スクリーンショット 2025-12-21 17 54 18" src="https://github.com/user-attachments/assets/90bb7591-f897-4fd5-b56c-733ea460a07f" />

<img width="933" height="579" alt="スクリーンショット 2025-12-21 17 54 13" src="https://github.com/user-attachments/assets/732566fb-629e-49be-bfd6-0e28120e1cde" />

<img width="933" height="579" alt="スクリーンショット 2025-12-21 17 54 07" src="https://github.com/user-attachments/assets/c09ef9c0-f921-4c5e-a540-711ecd5be752" />

<img width="933" height="579" alt="スクリーンショット 2025-12-21 17 54 02" src="https://github.com/user-attachments/assets/ffb2b773-8653-48e7-aa34-e1ba95bc9b98" />


## v0.1.1
ハイライト付与に対応しました。
以下のワークフローからハイライト付与版を実行可能です。

![ハイライト対応版Workflow](https://github.com/mattyamonaca/ComfyUI-fixableflow/blob/main/workflows/fixable-workflow-comfyui-highlight.json)

## 使用しているモデル
v0.1におけるワークフローでは、

| タスク名           | モデル名                          | 追加モデル名                         |
|--------------------|-----------------------------------|--------------------------------------|
| 線画生成           | cagliostrolab/animagine-xl-3.1    |  2vXpSwA7/iroiro-lora(image2flat_V1_1024_dim4-000040.safetensors)         |
| バケツ塗り(準備)     | cagliostrolab/animagine-xl-3.1    | kataragi/ControlNet-LineartXL|
| バケツ塗り(フラット化） | maybleMyers/framepack_h1111   |2vXpSwA7/iroiro-lora(image2flat_V1_1024_dim4-000040.safetensors) |
| 1影生成   | maybleMyers/framepack_h1111   |mattyamonaca/framepack-shade-adder_lora |
| ハイライト生成   | maybleMyers/framepack_h1111   |mattyamonaca/fixableflow (fpack_highlight_lora.safetensors) |

上記のモデルにはついては、バケツ塗り（フラット化）及び（1影生成）以外のモデルを使用いただいても問題ありません。
白背景の線画及び、生成した線画を参照して画像を生成できるモデルであれば基本的には全て互換性があります。

また、線画及びバケツ塗り画像を自分で（手描きなどの手段を用いて）用意できる場合は、画像を生成する代わりに手元の線画、もしくはバケツ塗り画像を入力することで1影生成が可能となります。

## Vast.aiを用いた環境構築
Comfy UIを用いてworkflowを実行できますが、PSD化を行うツールの環境設定がうまくいかないことがあるため、再現性のある環境設定方法としてVast.aiを用いた環境構築方法を提示します。
基本的にはローカルで動かす場合もVast.aiで動かす場合も手順は同じなので、ローカルで動かす場合であっても下記文章を参考にしていただければと思います。

<details>

  
  以下のURLからVast.aiに移動します。
  
  [vast.ai](https://cloud.vast.ai/?ref_id=250641)

  
（アカウントを持っていない場合はアカウントの作成をお願いします）


左上のハンバーガーボタンをクリックしてサイドメニューを開き、Templatesを選択します。


<img width="1049" height="608" alt="スクリーンショット 2025-12-21 11 04 30" src="https://github.com/user-attachments/assets/eacfe7e1-fdd9-45a6-904a-de3d99c876b7" />

Templatesの中からComfyUIを選択し、RTX5090をレンタルしてインスタンスを起動します。
（5090でなくても構いませんが、VRAM32GB以上を推奨します）
また、コンテナのVolumeサイズは100GB程度確保していくことを推奨します。


<img width="1293" height="479" alt="スクリーンショット 2025-12-21 11 19 17" src="https://github.com/user-attachments/assets/c820fc5c-ffc2-4b54-afd4-76e8e374a9f1" />
<img width="1177" height="443" alt="スクリーンショット 2025-12-21 11 25 16" src="https://github.com/user-attachments/assets/3f01970a-4a0b-4ecc-a5c9-ef951c72938b" />

インスタンスが起動すると「Open」ボタンが表示されるので、ボタンを押してインスタン内部に入ります。


<img width="905" height="233" alt="スクリーンショット 2025-12-21 11 27 11" src="https://github.com/user-attachments/assets/019d38f5-d2d0-489a-b99e-a54015133ffc" />

インスタンス内部に入ると以下のような画面が表示されるので、まずはJupyter Terminalを開きます。


<img width="1285" height="719" alt="スクリーンショット 2025-12-21 11 39 59" src="https://github.com/user-attachments/assets/d133ea9f-92f0-4601-bf03-6b8ee6e7f217" />


Terminalを開いたら、以下のコマンドを入力します。



```bash
cd /workspace/ComfyUI/custom_nodes/
git clone https://github.com/mattyamonaca/ComfyUI-fixableflow.git
cd /workspace/ComfyUI/custom_nodes/ComfyUI-fixableflow

pip install -r requirements.txt

pip install cython
pip install pytoshop -I --no-cache-dir
pip install psd-tools

cd /workspace/ComfyUI/models/checkpoints/
wget https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors

cd /workspace/ComfyUI/models/loras/
wget https://huggingface.co/2vXpSwA7/iroiro-lora/resolve/main/sdxl/sdxl-lineart_05.safetensors
wget https://huggingface.co/2vXpSwA7/iroiro-lora/resolve/main/sdxl/sdxl-flat.safetensors
wget https://huggingface.co/tori29umai/FramePack_LoRA/resolve/main/image2flat_V1_1024_dim4-000040.safetensors

cd /workspace/ComfyUI/models/controlnet
wget https://huggingface.co/kataragi/ControlNet-LineartXL/resolve/main/Katarag_lineartXL-fp16.safetensors

cd /workspace/ComfyUI/models/vae
wget https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
wget https://huggingface.co/hunyuanvideo-community/HunyuanVideo/resolve/main/vae/diffusion_pytorch_model.safetensors

cd /workspace/ComfyUI/models/clip
wget https://huggingface.co/maybleMyers/framepack_h1111/resolve/main/clip_l.safetensors
wget https://huggingface.co/maybleMyers/framepack_h1111/resolve/main/llava_llama3_fp16.safetensors

cd /workspace/ComfyUI/models/diffusion_models
wget https://huggingface.co/maybleMyers/framepack_h1111/resolve/main/FramePackI2V_HY_bf16.safetensors

cd /workspace/ComfyUI/models/loras
wget https://huggingface.co/mattyamonaca/framepack-shade-adder_lora/resolve/main/shade-adder-lora.safetensors
```

全ての処理が終わったら、再びインスタンス内部に入った時に表示された最初の画面を開き、ComfyUIを開きます。
ComfyUIを開いたら、以下のリンク先にあるjsonファイル（fixable-workflow.json）をダウンロードし、ダウンロード後にドラッグ&ドロップしてworkflowを読み込ませます。

[https://github.com/mattyamonaca/ComfyUI-fixableflow](https://github.com/mattyamonaca/ComfyUI-fixableflow/tree/main/workflows)

<img width="1285" height="616" alt="スクリーンショット 2025-12-21 13 08 21" src="https://github.com/user-attachments/assets/1ce2ef87-6ca3-4a20-b55e-f00ce9f08703" />

workflowを読み込むと上記のようなエラーが出ると思うので、不足しているノードを読み込んでいきます。
ComfyUI Managerを開き、「Install Missing Custom Nodes」を選択します。

下記のような画面が出てくるので、表示されているノードをすべて最新版でインストールします。
<img width="1073" height="578" alt="スクリーンショット 2025-12-21 13 09 30" src="https://github.com/user-attachments/assets/8d264e03-95a3-453d-8985-ecb7c4e4f1ff" />

全てのノードがインストールできたら、一度Comfy UIを再起動します。
再起動後にブラウザをリロードし、赤枠で囲われたノードがなくなったら準備完了です。

<img width="1232" height="579" alt="スクリーンショット 2025-12-21 13 12 21" src="https://github.com/user-attachments/assets/69d76614-4d82-447c-ab65-1f35f9e86bcd" />

</details>

## 今後の予定
FixableFlowでは以下の工程全てに対応する事を目標としています。
現在対応していない工程については、随時実装を進めていきます。


| 対応状況           | 対応タスク                          |
|--------------------|-----------------------------------|
|✅ |線画生成|
|✅ |下塗り(バケツ塗り)生成|
|✅ |1影生成|
|◻️ |2影生成|
|✅ |ハイライト生成|
|◻️ |仕上げ効果生成|
|◻️ |パーツ分割　　　　|






