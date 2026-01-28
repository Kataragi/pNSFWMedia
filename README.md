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

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# CLIP のインストール
pip install git+https://github.com/openai/CLIP.git
```

## ディレクトリ構造

```
pNSFWMedia/
├── dataset/
│   ├── images/           # 元画像（Stage A 用）
│   │   ├── train/
│   │   │   ├── sfw/
│   │   │   └── nsfw/
│   │   └── val/
│   │       ├── sfw/
│   │       └── nsfw/
│   └── embeddings/       # 埋め込み（Stage B 用）
│       ├── train/
│       │   ├── sfw/
│       │   └── nsfw/
│       └── val/
│           ├── sfw/
│           └── nsfw/
├── models/               # 学習済みモデル
├── logs/                 # TensorBoard ログ
├── results/              # 評価結果
└── src/                  # ソースコード
    ├── extract_embeddings.py   # Stage A: 埋め込み抽出
    ├── train_classifier.py     # Stage B: 分類器学習
    ├── inference.py            # 推論スクリプト
    └── prepare_dataset.py      # データセット準備
```

## 使い方

### 方法1: 自動分割モード（推奨）

train/valを自動で分割（デフォルト: train 85% / val 15%）

#### 1. データセットの準備

SFW/NSFW 画像を直接配置：

```
dataset/images/
├── sfw/
│   └── *.jpg
└── nsfw/
    └── *.jpg
```

#### 2. 埋め込みの抽出 (Stage A)

```bash
python src/extract_embeddings.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --flat \
    --batch-size 32
```

#### 3. 分類器の学習 (Stage B) - 自動分割

```bash
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --auto-split \
    --val-ratio 0.15 \
    --batch-size 64 \
    --epochs 40 \
    --use-class-weight
```

### 方法2: 手動分割モード

#### 1. データセットの準備

SFW/NSFW 画像を用意し、train/val に分割：

```bash
python src/prepare_dataset.py organize \
    --sfw-dir /path/to/sfw/images \
    --nsfw-dir /path/to/nsfw/images \
    --output-dir dataset/images \
    --train-ratio 0.85
```

データセット構造の確認：

```bash
python src/prepare_dataset.py verify --dataset-dir dataset/images
```

#### 2. 埋め込みの抽出 (Stage A)

```bash
python src/extract_embeddings.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --model ViT-B/32 \
    --batch-size 32
```

#### 3. 分類器の学習 (Stage B)

```bash
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --batch-size 64 \
    --epochs 40 \
    --learning-rate 1e-3 \
    --use-class-weight
```

### ハイパーパラメータチューニング

```bash
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --auto-split \
    --tune \
    --max-trials 30 \
    --epochs 100
```

### 4. TensorBoard で学習を監視

```bash
tensorboard --logdir logs
```

### 5. 推論

#### 埋め込みファイルから推論

```bash
# 単一ファイル
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --embedding path/to/embedding.npy

# ディレクトリ全体
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --embedding-dir dataset/embeddings/val \
    --output results/predictions.json
```

#### 画像から直接推論

```bash
# 単一画像
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image path/to/image.jpg

# ディレクトリ全体
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image-dir path/to/images \
    --output results/predictions.json
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
| auto_split | False | 自動でtrain/val分割 |
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
