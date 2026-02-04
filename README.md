# pNSFWMedia

Twitter/X の pNSFWMedia を参考にした NSFW 画像分類モデルの再現実装。

## 概要

このプロジェクトは、画像のNSFW（Not Safe For Work）判定を行う2段階構成のモデルを実装しています：

- **Stage A**: 画像特徴抽出（256次元埋め込み） — NudeNet YOLOv8 backbone + GAP + 直交射影
- **Stage B**: MLP による二値分類（NSFW確率を出力）
- **Adversarial**: Semantic Feature Migration (SFM) による敵対的摂動生成

## アーキテクチャ

### 分類パイプライン

```
[画像] → [NudeNet YOLOv8 Backbone] → [GAP + Orthogonal Projection] → [256-dim 埋め込み]
                                                                              ↓
[256-dim 埋め込み] → [BatchNorm + Dense(tanh/gelu)] × 1-2 → [Dense(sigmoid)] → [NSFW確率]
```

### Stage A: 埋め込み抽出（NudeNet）

- NudeNet の YOLOv8 ONNX モデルを使用
- ONNX 中間ノードからバックボーン特徴マップを抽出
- Global Average Pooling + ランダム直交射影で256次元に変換
- L2正規化を適用
- CUDA/CPU 自動検出（onnxruntime）

### Stage B: 分類器

- TensorFlow/Keras Sequential モデル
- BatchNormalization + Dense ブロック（1-2層）
- 活性化関数: tanh または gelu
- 出力層: Dense(1, sigmoid)
- 損失関数: BinaryCrossentropy

## インストール

### 基本インストール

```bash
pip install -r requirements.txt
```

### 敵対的摂動学習の追加依存

SFM 敵対的摂動の学習には CLIP と PyTorch が必要です：

```bash
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CUDA/GPU サポート（推奨）

```bash
# CUDA 対応 TensorFlow
pip install tensorflow[and-cuda]

# CUDA 対応 onnxruntime（NudeNet 用）
pip install onnxruntime-gpu
```

#### CUDA 環境の確認

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## ディレクトリ構造

```
pNSFWMedia/
├── dataset/
│   ├── images/           # 元画像
│   │   ├── sfw/
│   │   └── nsfw/
│   └── embeddings/       # 埋め込み（Stage B 用）
│       ├── sfw/
│       └── nsfw/
├── models/               # 学習済みモデル
│   └── adversarial/      # SFM 敵対的摂動モデル
├── logs/                 # TensorBoard ログ
├── results/              # 評価結果
└── src/
    ├── extract_embeddings_nudenet.py  # Stage A: NudeNet 埋め込み抽出
    ├── extract_embeddings.py          # Stage A: CLIP 埋め込み抽出
    ├── train_classifier.py            # Stage B: 分類器学習
    ├── inference.py                   # 推論スクリプト
    └── adversarial/                   # 敵対的摂動学習
        ├── __init__.py
        ├── models.py                  # Generator / Perceptual / Bridge
        ├── dataset.py                 # NSFW/SFW ペア画像データセット
        ├── losses.py                  # SFM 複合損失関数
        └── train.py                   # 学習スクリプト
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

NudeNet の YOLOv8 バックボーンから256次元埋め込みを抽出します。
CUDA が利用可能な場合は自動的に GPU を使用します。

```bash
python src/extract_embeddings_nudenet.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --batch-size 32
```

**オプション:**

```bash
# 射影行列のパスを指定
python src/extract_embeddings_nudenet.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --projection-path models/nudenet_projection.npy

# ONNX モデルのノードを確認（デバッグ用）
python src/extract_embeddings_nudenet.py --list-nodes

# 特徴抽出ノードを手動指定
python src/extract_embeddings_nudenet.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --feature-node "/model.10/cv2/cv2.2/Conv_output_0"
```

### 3. 分類器の学習 (Stage B)

train/val は自動で分割されます（デフォルト: train 85% / val 15%）。

```bash
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --batch-size 64 \
    --epochs 40 \
    --use-class-weight
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

```bash
# 画像から直接推論
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image path/to/image.jpg

# 閾値を変更（敏感な判定）
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image path/to/image.jpg \
    --threshold 0.3

# ディレクトリ全体
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image-dir path/to/images \
    --output results/predictions.json
```

#### 出力例

```
Threshold: 0.5

Prediction for: path/to/image.jpg
  NSFW Probability: 0.8234
  Threshold:        0.5
  Classification:   nsfw
```

---

## 敵対的摂動学習 (Semantic Feature Migration)

### 概要

NSFW 画像に対して人間には知覚困難な微小ノイズ（摂動）を加え、pNSFWMedia 分類器に SFW と誤認させる **摂動生成ネットワーク** を学習します。

本手法は FGSM・PGD・Carlini & Wagner・DeepFool のいずれとも異なり、フィードフォワード型の生成モデルが一回のフォワードパスで摂動を出力する独自設計です。

### SFM アーキテクチャ

```
[NSFW画像] ─┬──────────────────────────────── [元画像]
            │                                     │
            ↓                                     ↓ (視覚比較)
    [CLIP Encoder (frozen)]                [PerceptualSimilarityNet]
            ↓                                     ↑
    [NSFW 埋め込み]                          [摂動後画像]
            │                                     ↑
            ↓                              [元画像 + δ (clamp)]
  [FeatureDifferenceConditioner]                  ↑
    (SFW centroid - NSFW embed)         [PerturbationGenerator]
            │                              ↑          ↑
            ↓                           [NSFW画像] [cond vector]
       [cond vector] ──────────────────────────────┘
```

学習される全体の流れ：

1. NSFW / SFW 画像ペアを CLIP で埋め込み → セントロイドを EMA で追跡
2. `sfw_centroid - nsfw_embed` の差分ベクトルを Generator に FiLM 注入
3. Generator が元画像に対する摂動 δ を出力（`tanh × ε` で有界）
4. 摂動後画像を再び CLIP + 分類器に通し、4成分損失で Generator を更新

### 損失関数

```
L_total = λ_cls  × L_classification       # BCE(pred, 0.05): SFW 確信への誘導
        + λ_feat × L_feature_migration     # 埋め込みを SFW セントロイドへ接近
        + λ_perc × L_perceptual            # VGG 多スケール知覚的類似性
        + λ_mag  × L_magnitude             # 摂動 δ の MSE ノルム抑制
```

| 損失 | 目的 | デフォルト重み |
|------|------|---------------|
| L_classification | 分類器出力を SFW 側 (0.05) へ誤誘導 | λ=1.0 |
| L_feature_migration | 埋め込み空間で SFW 分布へ意味的に移動 | λ=2.0 |
| L_perceptual | 人間の視覚に近い多層 VGG 特徴比較で視覚的忠実性を保持 | λ=1.0 |
| L_magnitude | 摂動の絶対量を抑制し知覚不可能に維持 | λ=10.0 |

### データセットの準備

敵対的摂動学習には **NSFW 画像と SFW 画像の両方** が必要です。分類器学習（Stage B）用の埋め込みではなく、**元画像ファイル** を使います。

```
dataset/images/
├── sfw/          # SFW 画像（特徴分布の学習に使用）
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
└── nsfw/         # NSFW 画像（摂動対象）
    ├── img001.jpg
    ├── img002.png
    └── ...
```

- **対応フォーマット**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- **推奨画像数**: NSFW / SFW 各 500 枚以上（多いほど特徴分布の学習精度が向上）
- **画像サイズ**: 任意（学習時に自動で 224×224 にリサイズ）
- SFW 画像は摂動対象ではなく、**NSFW/SFW 間の特徴差分を学習する参照** として使用されます

### 前提条件

学習開始前に以下が必要です：

1. **学習済み分類器** `models/pnsfwmedia_classifier.keras`（Stage B で生成）
2. **CLIP 射影層** `models/clip_projection.pt`（`src/extract_embeddings.py` で生成）
3. **NSFW/SFW 画像** が `dataset/images/` に配置済み

分類器と射影層がまだ無い場合は、先に Stage A → B を完了してください：

```bash
# Stage A: 埋め込み抽出
python src/extract_embeddings_nudenet.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings

# Stage B: 分類器学習
python src/train_classifier.py \
    --embeddings-dir dataset/embeddings \
    --epochs 40

# CLIP 射影層の生成（SFM 学習に必要）
python src/extract_embeddings.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings_clip
```

### 学習の実行

```bash
python src/adversarial/train.py \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras \
    --projection-path models/clip_projection.pt \
    --epochs 50 \
    --batch-size 8 \
    --lr 2e-4 \
    --epsilon 0.03
```

### 学習パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--epsilon` | 0.03 | 摂動の最大 L∞ 強度（[0,1] スケール） |
| `--lr` | 2e-4 | Generator の学習率 |
| `--batch-size` | 8 | バッチサイズ（VRAM に応じて調整） |
| `--epochs` | 50 | 学習エポック数 |
| `--lambda-cls` | 1.0 | 分類損失の重み |
| `--lambda-feat` | 2.0 | 特徴移動損失の重み |
| `--lambda-perc` | 1.0 | 知覚的類似性損失の重み |
| `--lambda-mag` | 10.0 | 摂動量損失の重み |
| `--sfw-target` | 0.05 | 目標 NSFW 確率（低いほど強い誘導） |
| `--image-size` | 224 | 学習時の画像リサイズ先 |
| `--clip-model` | ViT-B/32 | CLIP モデル種別 |

### CUDA 互換性問題のデバッグ

PyTorch / ONNX / onnx2torch 変換や CUDA 実行時にエラーが発生した場合、以下の環境変数を指定して同期実行モードで原因を特定します：

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python src/adversarial/train.py \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras \
    --epochs 50
```

- `CUDA_VISIBLE_DEVICES=0`: 使用する GPU を明示的に指定
- `CUDA_LAUNCH_BLOCKING=1`: CUDA カーネルを同期実行し、エラー発生行を正確に特定

CPU のみで学習する場合：

```bash
python src/adversarial/train.py --cpu \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras
```

### チェックポイントと再開

学習中のチェックポイントは `models/adversarial/` に保存されます：

```
models/adversarial/
├── sfm_best.pt              # 最高 ASR のチェックポイント
├── sfm_epoch004.pt           # 5エポックごとの定期保存
├── sfm_final.pt              # 最終エポック
└── train_config.json         # 学習設定の記録
```

中断した学習を再開するには：

```bash
python src/adversarial/train.py \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras \
    --resume models/adversarial/sfm_epoch004.pt
```

### 学習の監視

```bash
tensorboard --logdir logs/adversarial
```

TensorBoard で確認できるメトリクス：

- **Attack Success Rate (ASR)**: 摂動後に SFW と判定される割合
- **loss_cls / loss_feat / loss_perc / loss_mag**: 各損失成分の推移

---

## 分類器 学習設定

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
- `models/nudenet_projection.npy`: NudeNet 射影行列
- `models/adversarial/sfm_best.pt`: 最良の敵対的摂動生成モデル
- `logs/`: TensorBoard ログ
- `results/`: 評価結果とグラフ

## 参考

- 元実装: `nsfw_media.py` (Twitter/X pNSFWMedia)
- NudeNet: [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- KerasTuner: [keras-team/keras-tuner](https://github.com/keras-team/keras-tuner)
