# pNSFWMedia

Twitter/X の pNSFWMedia を参考にした NSFW 画像分類モデルの再現実装。

## 概要

このプロジェクトは、画像のNSFW（Not Safe For Work）判定を行う2段階構成のモデルを実装しています：

- **Stage A**: CLIP による画像特徴抽出（256次元埋め込み）
- **Stage B**: MLP による二値分類（NSFW確率を出力）

## アーキテクチャ

```
[画像] → [CLIP ViT-B/32] → [Linear Projection] → [256-dim 埋め込み]
                                                          ↓
[256-dim 埋め込み] → [BatchNorm + Dense(tanh/gelu)] × 1-2 → [Dense(sigmoid)] → [NSFW確率]
```

### Stage A: CLIP 埋め込み

- OpenAI CLIP (ViT-B/32) を使用
- 512次元出力を線形射影で256次元に変換
- L2正規化を適用
- 推論時のみ使用（凍結）

### Stage B: 分類器

- TensorFlow/Keras Sequential モデル
- BatchNormalization + Dense ブロック（1-2層）
- 活性化関数: tanh または gelu
- 出力層: Dense(1, sigmoid)
- 損失関数: BinaryCrossentropy

## インストール

### 基本インストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# CLIP のインストール
pip install git+https://github.com/openai/CLIP.git
```

### CUDA/GPU サポート（推奨）

GPU を使用して学習を高速化するには、CUDA 対応の TensorFlow と PyTorch をインストールしてください。

```bash
# CUDA 対応 TensorFlow のインストール
pip install tensorflow[and-cuda]

# CUDA 対応 PyTorch のインストール（CLIP 用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 環境の確認

```bash
# TensorFlow の GPU 確認
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# PyTorch の CUDA 確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## ディレクトリ構造

```
pNSFWMedia/
├── dataset/
│   ├── images/           # 元画像（Stage A 用）
│   │   ├── sfw/
│   │   └── nsfw/
│   └── embeddings/       # 埋め込み（Stage B 用）
│       ├── sfw/
│       └── nsfw/
├── models/               # 学習済みモデル
├── logs/                 # TensorBoard ログ
├── results/              # 評価結果
└── src/                  # ソースコード
    ├── extract_embeddings.py   # Stage A: 埋め込み抽出
    ├── train_classifier.py     # Stage B: 分類器学習
    └── inference.py            # 推論スクリプト
```

## 使い方

### 1. データセットの準備

SFW/NSFW 画像を直接配置：

```
dataset/images/
├── sfw/
│   └── *.jpg
└── nsfw/
    └── *.jpg
```

### 2. 埋め込みの抽出 (Stage A)

CUDA が利用可能な場合は自動的に GPU を使用します。

```bash
python src/extract_embeddings.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --batch-size 32
```

GPU を明示的に指定する場合：

```bash
python src/extract_embeddings.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --device cuda \
    --batch-size 64
```

### 3. 分類器の学習 (Stage B)

train/val は自動で分割されます（デフォルト: train 85% / val 15%）。
CUDA が利用可能な場合は自動的に GPU を使用します。

```bash
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --batch-size 64 \
    --epochs 40 \
    --use-class-weight
```

実行時に以下のようなメッセージが表示されます：

```
[GPU] CUDA is available. Found 1 GPU(s):
  [0] /physical_device:GPU:0
[GPU] Training will use CUDA acceleration
```

または：

```
[GPU] CUDA is not available. Using CPU for training.
[CPU] Training will use CPU
```

### 4. ハイパーパラメータチューニング（オプション）

```bash
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --tune \
    --max-trials 30 \
    --epochs 100
```

### 5. TensorBoard で学習を監視

```bash
tensorboard --logdir logs
```

### 6. 推論

`--threshold` で分類閾値を指定できます（デフォルト: 0.5）。
NSFW確率が閾値以上の場合に NSFW と判定されます。

#### 埋め込みファイルから推論

```bash
# 単一ファイル
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --embedding path/to/embedding.npy

# 閾値を変更して推論（厳密な判定）
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --embedding path/to/embedding.npy \
    --threshold 0.7

# ディレクトリ全体
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --embedding-dir dataset/embeddings \
    --output results/predictions.json
```

#### 画像から直接推論

```bash
# 単一画像
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image path/to/image.jpg

# 閾値を下げて推論（敏感な判定）
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image path/to/image.jpg \
    --threshold 0.3

# ディレクトリ全体
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image-dir path/to/images \
    --output results/predictions.json \
    --threshold 0.6
```

#### 出力例

```
Threshold: 0.5

Prediction for: path/to/image.jpg
  NSFW Probability: 0.8234
  Threshold:        0.5
  Classification:   nsfw
```

## 学習設定

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| batch_size | 64 | バッチサイズ |
| epochs | 40 | 最大エポック数 |
| learning_rate | 1e-3 | 学習率 |
| patience | 5 | Early stopping の patience |
| units | 256 | 隠れ層のユニット数 |
| num_layers | 1 | 隠れ層の数 (1-2) |
| activation | tanh | 活性化関数 (tanh/gelu) |
| dropout | 0.0 | ドロップアウト率 |
| val_ratio | 0.15 | 検証データの割合（15%） |

## メトリクス

- **PR-AUC**: Precision-Recall 曲線下面積
- **ROC-AUC**: ROC 曲線下面積
- **Precision@Recall=0.9**: Recall=0.9 での Precision

## 出力物

- `models/pnsfwmedia_classifier.keras`: 学習済み分類器
- `models/clip_projection.pt`: CLIP 射影層の重み
- `logs/pnsfwmedia/`: TensorBoard ログ
- `results/`: 評価結果とグラフ

## 参考

- 元実装: `nsfw_media.py` (Twitter/X pNSFWMedia)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- KerasTuner: [keras-team/keras-tuner](https://github.com/keras-team/keras-tuner)
