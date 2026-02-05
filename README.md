# pNSFWMedia

Twitter/X の pNSFWMedia を参考にした NSFW 画像分類モデルの再現実装。

## 概要

このプロジェクトは、画像のNSFW（Not Safe For Work）判定を行う2段階構成のモデルを実装しています：

- **Stage A**: 画像特徴抽出（256次元埋め込み） — NudeNet YOLOv8 backbone + GAP + 直交射影
- **Stage B**: MLP による二値分類（NSFW確率を出力）
- **Adversarial**: Semantic Feature Migration (SFM) による敵対的摂動生成

## インストール

### Windows（推奨）

`setup.bat` を実行すると、仮想環境の作成・依存パッケージのインストール・ディレクトリ構造の作成が自動で行われます。

```cmd
git clone https://github.com/Kataragi/pNSFWMedia.git
cd pNSFWMedia
setup.bat
```

セットアップ完了後、仮想環境を有効化：

```cmd
venv\Scripts\activate
```

### Linux / macOS

```bash
git clone https://github.com/Kataragi/pNSFWMedia.git
cd pNSFWMedia
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### CUDA/GPU サポート（オプション）

GPU を使用して学習・推論を高速化するには、CUDA 対応パッケージを追加インストールしてください。

```bash
# CUDA 対応 PyTorch（Windows / Linux）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 対応 TensorFlow
pip install tensorflow[and-cuda]

# CUDA 対応 onnxruntime（NudeNet 用）
pip install onnxruntime-gpu
```

#### CUDA 環境の確認

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### 学習済みモデルのダウンロード

仮想環境を有効化した状態で `download_models.bat` を実行すると、HuggingFace から学習済みモデルをダウンロードします。

#### Windows

```cmd
venv\Scripts\activate
download_models.bat
```

#### Linux / macOS

```bash
source venv/bin/activate
pip install huggingface_hub
huggingface-cli download kataragi/adversarial --local-dir models/adversarial --local-dir-use-symlinks False
```

ダウンロードされるモデル（[HuggingFace リポジトリ](https://huggingface.co/kataragi/adversarial)）：

| ファイル | 説明 |
|---------|------|
| `sfm_final.pt` | 敵対的摂動生成モデル（最終エポック） |
| `sfm_best.pt` | 敵対的摂動生成モデル（最高 ASR） |
| `pnsfwmedia_classifier.keras` | NSFW 分類器 |
| `clip_projection.pt` | CLIP 射影層の重み |

---

## 敵対的摂動の適用（クイックスタート）

学習済みの敵対的摂動モデルを使い、NSFW 画像に知覚困難なノイズを加えて SFW に誤認させます。
出力画像は入力画像の **オリジナル解像度** で保存されます。

### 必要なファイル

以下のファイルが必要です（`download_models.bat` で自動ダウンロード可能）：

- `models/adversarial/sfm_final.pt` — 学習済み摂動生成モデル
- `models/adversarial/pnsfwmedia_classifier.keras` — NSFW 分類器
- `models/adversarial/clip_projection.pt` — CLIP 射影層の重み

### 単一画像に適用

```bash
python src/adversarial/apply.py \
    --checkpoint models/adversarial/sfm_final.pt \
    --classifier-path models/adversarial/pnsfwmedia_classifier.keras \
    --projection-path models/adversarial/clip_projection.pt \
    --image path/to/image.jpg \
    --output-dir output/adversarial
```

### ディレクトリ内の画像を一括処理

```bash
python src/adversarial/apply.py \
    --checkpoint models/adversarial/sfm_final.pt \
    --classifier-path models/adversarial/pnsfwmedia_classifier.keras \
    --projection-path models/adversarial/clip_projection.pt \
    --image-dir path/to/images/ \
    --output-dir output/adversarial
```

### 出力例

コンソールには加工前後の NSFW 確率と判定結果が表示されます：

```
============================================================
Image                           Before      After      Result
------------------------------------------------------------
  photo001.jpg                  0.9312 NSFW  0.1247 SFW   FLIPPED
  photo002.jpg                  0.8876 NSFW  0.0983 SFW   FLIPPED
  photo003.jpg                  0.7654 NSFW  0.4312 SFW   FLIPPED

============================================================
Summary
============================================================
  Total images processed : 3
  NSFW -> SFW flipped    : 3 / 3  (100.0%)
  Avg NSFW prob (before)  : 0.8614
  Avg NSFW prob (after)   : 0.2181
  Output directory        : output/adversarial/
  Report saved to         : output/adversarial/results.json
```

### 推論パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--checkpoint` | (必須) | 学習済み SFM チェックポイント (.pt) |
| `--classifier-path` | `models/pnsfwmedia_classifier.keras` | pNSFWMedia 分類器 |
| `--projection-path` | `models/clip_projection.pt` | CLIP 射影層の重み |
| `--clip-model` | ViT-B/32 | CLIP モデル種別 |
| `--image` | — | 単一画像パス（`--image-dir` と排他） |
| `--image-dir` | — | 画像ディレクトリ（`--image` と排他） |
| `--output-dir` | `output/adversarial` | 出力先ディレクトリ |
| `--suffix` | `_perturbed` | 出力ファイル名に付加するサフィックス |
| `--threshold` | 0.5 | NSFW 分類閾値 |
| `--cpu` | — | CPU モードを強制 |

### JSON レポート

処理結果は `output/adversarial/results.json` に自動保存されます：

```json
{
  "checkpoint": "models/adversarial/sfm_final.pt",
  "classifier": "models/pnsfwmedia_classifier.keras",
  "threshold": 0.5,
  "epsilon": 0.03,
  "summary": {
    "total": 3,
    "flipped": 3,
    "flip_rate": 1.0,
    "avg_prob_before": 0.8614,
    "avg_prob_after": 0.2181
  },
  "images": [
    {
      "input": "path/to/photo001.jpg",
      "output": "output/adversarial/photo001_perturbed.jpg",
      "original_size": [1920, 1080],
      "prob_before": 0.9312,
      "prob_after": 0.1247,
      "label_before": "NSFW",
      "label_after": "SFW",
      "flipped": true
    }
  ]
}
```

---

## ディレクトリ構造

```
pNSFWMedia/
├── setup.bat             # Windows セットアップスクリプト
├── download_models.bat   # 学習済みモデルのダウンロード
├── requirements.txt      # 依存パッケージ
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
    └── adversarial/                   # 敵対的摂動
        ├── __init__.py
        ├── models.py                  # Generator / Perceptual / Bridge
        ├── dataset.py                 # NSFW/SFW ペア画像データセット
        ├── losses.py                  # SFM 複合損失関数
        ├── train.py                   # 学習スクリプト
        └── apply.py                   # 摂動適用・推論スクリプト
```

---

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

### SFM 敵対的摂動アーキテクチャ

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

---

## 分類器の学習

### 1. データセットの準備

SFW/NSFW 画像を配置：

```
dataset/images/
├── sfw/
│   └── *.jpg
└── nsfw/
    └── *.jpg
```

### 2. 埋め込みの抽出 (Stage A)

```bash
python src/extract_embeddings_nudenet.py \
    --input-dir dataset/images \
    --output-dir dataset/embeddings \
    --batch-size 32
```

### 3. 分類器の学習 (Stage B)

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

### 5. 推論

```bash
python src/inference.py \
    --model-path models/pnsfwmedia_classifier.keras \
    --image path/to/image.jpg
```

### 分類器 学習パラメータ

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

---

## 敵対的摂動モデルの学習

NSFW 画像に対して人間には知覚困難な微小ノイズ（摂動）を加え、pNSFWMedia 分類器に SFW と誤認させる **摂動生成ネットワーク** を学習します。

本手法は FGSM・PGD・Carlini & Wagner・DeepFool のいずれとも異なり、フィードフォワード型の生成モデルが一回のフォワードパスで摂動を出力する独自設計です。

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

### 前提条件

学習開始前に以下が必要です：

1. **学習済み分類器** `models/pnsfwmedia_classifier.keras`（Stage B で生成）
2. **CLIP 射影層** `models/clip_projection.pt`（`src/extract_embeddings.py` で生成）
3. **NSFW/SFW 画像** が `dataset/images/` に配置済み

分類器と射影層がまだ無い場合は、先に Stage A → B を完了し、CLIP 射影層を生成してください：

```bash
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
    --epochs 30 \
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
| `--output-dir` | models/adversarial | モデル出力先 |

### ノイズ量の調整

視覚的に見えないレベルまでノイズを抑えたい場合：

```bash
python src/adversarial/train.py \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras \
    --epsilon 0.012 \
    --lambda-cls 3.0 \
    --lambda-feat 4.0 \
    --lambda-perc 5.0 \
    --lambda-mag 50.0 \
    --sfw-target 0.10 \
    --epochs 30
```

### チェックポイント

学習中のチェックポイントは `models/adversarial/` に保存されます：

```
models/adversarial/
├── sfm_best.pt              # 最高 ASR のチェックポイント
├── sfm_epoch004.pt          # 5エポックごとの定期保存
├── sfm_final.pt             # 最終エポック
└── train_config.json        # 学習設定の記録
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

### CUDA 互換性問題のデバッグ

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python src/adversarial/train.py \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras
```

CPU のみで学習する場合：

```bash
python src/adversarial/train.py --cpu \
    --image-dir dataset/images \
    --classifier-path models/pnsfwmedia_classifier.keras
```

---

## メトリクス

- **PR-AUC**: Precision-Recall 曲線下面積
- **ROC-AUC**: ROC 曲線下面積
- **Precision@Recall=0.9**: Recall=0.9 での Precision
- **ASR (Attack Success Rate)**: 敵対的摂動で SFW に誤分類された割合

## 出力物

- `models/pnsfwmedia_classifier.keras`: 学習済み分類器
- `models/nudenet_projection.npy`: NudeNet 射影行列
- `models/clip_projection.pt`: CLIP 射影層の重み
- `models/adversarial/sfm_best.pt`: 最良の敵対的摂動生成モデル
- `logs/`: TensorBoard ログ
- `results/`: 評価結果とグラフ

## 参考

- 元実装: `nsfw_media.py` (Twitter/X pNSFWMedia)
- NudeNet: [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- KerasTuner: [keras-team/keras-tuner](https://github.com/keras-team/keras-tuner)
