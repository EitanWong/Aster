<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **本番対応 Apple Silicon ローカル LLM 推論ランタイム**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster は、長いコンテキストと OpenClaw スタイルのエージェント ワークロード向けに最適化された、本番対応の Apple Silicon ローカル LLM 推論ランタイムです。

## Aster を選ぶ理由

Aster は以下のシナリオに最適化されています：

- 巨大なプロンプトと繰り返される長いプレフィックス
- ツール集約型エージェント プロンプト
- 長い会話
- 継続的なローカル バックグラウンド サービス
- ベンチマーク検証済みのランタイム ポリシー選択
- Apple Silicon + MLX デプロイメント

OpenAI 互換 API を公開し、高度な最適化を教義ではなく候補戦略として扱います。推測デコーディング、プレフィックス キャッシング、バッチ処理、スケジューリング、ストリーミング ケイデンスはすべてベンチマークされ、測定されたローカル パフォーマンスと安定性に基づいて選択されます。

## コア アイデア

- ストリーミングおよび非ストリーミング エンドポイント付き OpenAI 互換 API
- 明示的なプリフィル/デコード分割
- キュー認識バッチ処理を備えた適応型スケジューラー
- ページング KV マネージャー抽象化
- 決定論的ハッシング付き自動プレフィックス キャッシュ
- 自動無効化フォールバック付き推測デコーディング コントローラー
- 最速の安定したプロファイルを保持するベンチマーク/自動チューニング サブシステム
- 構造化ログ、メトリクス、監督、および準備/ヘルス レポート

## クイック スタート

```bash
cd /Users/eitan/Documents/Projects/Python/Aster

# 仮想環境を作成
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# 依存関係をインストール（ASR/TTS 用 mlx-audio を含む）
python -m pip install -r requirements.txt

# モデルをダウンロード（ASR、LLM、TTS）
bash scripts/setup/download_models.sh

# サーバーを起動
python -m aster --config configs/config.yaml
```

API は `http://127.0.0.1:8080` で利用可能になります

### インストールの確認

```bash
# ヘルス チェック
curl http://127.0.0.1:8080/health

# LLM 推論をテスト
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "max_tokens": 100
  }'

# ASR（音声テキスト変換）をテスト
python scripts/test_audio_cli.py --tts "こんにちは世界" --output test.wav
python scripts/test_audio_cli.py --asr test.wav

# エンドツーエンド パイプラインをテスト
python scripts/test_audio_cli.py --pipeline "これはテストです"
```

## Python バージョン

Aster は最新の Python を対象としており、Python 3.13.x（利用可能な場合）で実行する必要があります（最小 3.12+）。macOS システム Python はこのプロジェクトではサポートされていません。

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` — LLM チャット推論
- `POST /v1/completions` — LLM テキスト完成
- `POST /v1/audio/transcriptions` — ASR（音声テキスト変換）
- `POST /v1/audio/speech` — TTS（テキスト音声変換）

互換性に関する注記：
- Aster のデフォルト互換性契約とオプトイン デバッグ拡張については、`docs/api/OPENAI_COMPAT.md` を参照してください。

## オーディオ サービス（ASR & TTS）

Aster には、Qwen3 モデルによって駆動される統合音声認識と合成が含まれています：

### ASR（音声テキスト変換）
- モデル：Qwen3-ASR-0.6B（0.66GB）
- 複数言語をサポート
- 高速ローカル転写

### TTS（テキスト音声変換）
- ベース モデル：Qwen3-TTS-0.6B（1.59GB）
- CustomVoice モデル：Qwen3-TTS-CustomVoice-0.6B（オプション、音声クローン用）
- 調整可能な音声速度
- 参照音声を使用した音声クローン

### オーディオ API の例

**TTS（テキスト音声変換）：**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "こんにちは、これはテストです",
    "voice": "default",
    "speed": 1.0
  }' \
  --output output.wav
```

**ASR（音声テキスト変換）：**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

### オーディオ テスト

提供されている CLI テスト ツールを使用します：
```bash
# TTS をテスト
python scripts/test_audio_cli.py --tts "こんにちは世界" --output output.wav

# ASR をテスト
python scripts/test_audio_cli.py --asr output.wav

# エンドツーエンド パイプラインをテスト（TTS -> ASR）
python scripts/test_audio_cli.py --pipeline "テスト メッセージ"

# 完全なテスト スイートを実行
pytest tests/test_audio_services.py -v -s
```

詳細なオーディオ サービス ドキュメントについては、`docs/guides/DEPLOYMENT.md` を参照してください。

## ベンチマーク哲学

スタートアップ自動チューニングは、短いウォームアップ ベンチマークを実行して、最速の安定したポリシーを選択できます。ベンチマーク サブシステムは以下を比較します：

- 推測デコーディング オン/オフ
- ドラフト トークン数
- プレフィックス キャッシュ オン/オフ
- バッチ ウィンドウ
- バッチ キャップ
- ページ サイズ
- スケジューリング モード
- ストリーミング フラッシュ ケイデンス

プロファイルは保持され、後続の起動時に使用されます。

## Apple Silicon チューニング ノート

- 繰り返される動的割り当てよりも事前割り当てとページ プールを優先する
- 統一メモリ スラッシュを避けるために MLX モデル常駐を慎重に使用する
- マシンごとにプレフィックス キャッシングと推測デコーディングをベンチマークする
- Python ホット パスを小さく保つ。調整を安定したループに移動する
- 長いプロンプト下での一貫した最初のトークン レイテンシを優先する

## 動的最適化哲学

Aster は、ローカル マシンで有益であることが証明された最適化のみを有効にします：

- 推測デコーディングはグローバルに無効にすることも、リクエスト クラスごとに無効にすることもできます
- ヒット率が低い場合またはメモリ圧力が上昇した場合、プレフィックス キャッシュを削減または無効にできます
- レイテンシが上昇すると、バッチ ウィンドウが自動的に縮小します
- 不安定性または回帰が検出されると、フォールバック プロファイルが選択されます

## モデル セットアップ

hfd + aria2 アクセラレーション付きのワンクリック モデル ダウンロード：

```bash
# すべての必要なモデルをダウンロード（ASR、LLM、TTS）
bash scripts/setup/download_models.sh

# または Python を直接使用してより多くの制御を行う
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

詳細な手順については、`scripts/setup/README-model-download.md` を参照してください。

## モデル パス

`model.path` と `model.draft_path` は以下のいずれかです：
- MLX 変換モデル ディレクトリへの絶対ローカル パス
- `mlx-lm` で読み込み可能な互換 Hugging Face リポジトリ ID

本番環境では、ローカル MLX 変換ディレクトリを優先します。`configs/config.yaml` を更新します：

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## OpenClaw 統合

OpenClaw を Aster の OpenAI 互換ベース URL とモデル ID にポイントします。Aster は繰り返されるシステム/ツール プレフィックスと長期エージェント セッション用に構築されているため、安定したスキャフォールディングと長いコンテキスト再利用を備えたワークロードから特に利益を得るはずです。

## プロジェクト ガイダンス ドキュメント

- `docs/guides/QUICK_START_MODELS.md` — モデル ダウンロード クイック ガイド
- `docs/reference/MODEL_SETUP.md` — 詳細なセットアップとトラブルシューティング
- `docs/development/MODEL_DOWNLOAD_ARCHITECTURE.md` — システム設計
- `docs/reference/ROADMAP.md` — 長期アーキテクチャ進化計画
- `docs/api/OPENAI_COMPAT.md` — 互換性の境界とデバッグ拡張
- `docs/development/DEBUGGING.md` — オペレーター デバッグ ガイド
- `docs/operations/OPERATIONS.md` — 日常的なサービス運用
- `docs/guides/BENCHMARK_GUIDE.md` — パフォーマンス ベンチマーク ガイド
- `docs/guides/BACKGROUND_SERVICE_SETUP.md` — バックグラウンド サービス セットアップ
- `DOCS.md` — 完全なドキュメント ナビゲーション
