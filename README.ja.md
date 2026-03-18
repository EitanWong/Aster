# Aster

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)

Asterは、長いコンテキストとOpenClawスタイルのエージェントワークロード向けに最適化されたApple Silicon用のローカルLLM推論ランタイムです。

## Asterを選ぶ理由

Asterは以下のシナリオに最適化されています：

- 超長いプロンプトと繰り返される長いプレフィックス
- ツール集約的なエージェントプロンプト
- 長い会話
- 継続的なローカルバックグラウンドサービス
- ベンチマーク検証されたランタイムポリシー選択
- Apple Silicon + MLXデプロイメント

OpenAI互換のAPIを提供し、高度な最適化を教義ではなく候補戦略として扱います。推測デコーディング、プレフィックスキャッシング、バッチ処理、スケジューリング、ストリーミングレートはすべてベンチマークされ、測定されたローカルパフォーマンスと安定性に基づいて選択されます。

## コア機能

- OpenAI互換のAPI（ストリーミングおよび非ストリーミングエンドポイント）
- 明示的なprefill/decode分離
- キュー認識型の適応スケジューラ
- ページングKVマネージャー抽象化
- 決定論的ハッシング付き自動プレフィックスキャッシュ
- 自動無効化フォールバック付き推測デコーディングコントローラー
- ベンチマーク/自動チューニングサブシステム
- 構造化ログ、メトリクス、監視、および準備/ヘルスレポート

## クイックスタート

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python -m aster --config configs/config.yaml
```

## Pythonバージョン

Asterは最新のPythonを対象としており、Python 3.13.x（利用可能な場合）またはPython 3.12以上で実行する必要があります。macOSシステムPythonはこのプロジェクトではサポートされていません。

## APIエンドポイント

- `GET /health` - ヘルスチェック
- `GET /ready` - 準備状態チェック
- `GET /metrics` - Prometheusメトリクス
- `GET /v1/models` - モデルリスト
- `POST /v1/chat/completions` - チャット完成
- `POST /v1/completions` - テキスト完成

互換性に関する注記：
- `docs/OPENAI_COMPAT.md`を参照して、Asterのデフォルト互換性契約とオプトインデバッグ拡張を確認してください。

## ベンチマーク理念

起動時の自動チューニングは、短い予熱ベンチマークを実行して最速の安定ポリシーを選択できます。ベンチマークサブシステムは以下を比較します：

- 推測デコーディングのオン/オフ
- ドラフトトークン数
- プレフィックスキャッシュのオン/オフ
- バッチウィンドウ
- バッチキャップ
- ページサイズ
- スケジューリングモード
- ストリーミングフラッシュレート

プロファイルは保存され、後続の起動時に使用されます。

## Apple Silicon チューニングノート

- 繰り返される動的割り当てよりも事前割り当てとページプールを優先する
- 統一メモリスラッシングを避けるためにMLXモデルの常駐を慎重に使用する
- マシンごとにプレフィックスキャッシングと推測デコーディングをベンチマークする
- Pythonホットパスを小さく保つ；調整を安定ループに移動する
- 長いプロンプトの下での一貫した最初のトークンレイテンシを優先する

## 動的最適化理念

Asterは、ローカルマシンで有益であることが証明された最適化のみを有効にします：

- 推測デコーディングはグローバルに無効にするか、リクエストクラスごとに無効にできます
- ヒット率が低いか、メモリ圧力が上昇した場合、プレフィックスキャッシュを削減または無効にできます
- レイテンシが上昇すると、バッチウィンドウが自動的に縮小します
- 不安定性または回帰が検出されると、フォールバックプロファイルが選択されます

## モデルパス

`model.path`と`model.draft_path`は以下のいずれかです：
- MLX変換モデルディレクトリへの絶対ローカルパス
- `mlx-lm`でロード可能な互換性のあるHugging Faceリポジトリ ID

意図された本番設定では、9Bターゲットと0.8Bドラフトモデルの両方にローカルMLX変換ディレクトリを優先してください。

有用なセットアップと検証コマンド：

```bash
bash scripts/setup/download_models.sh
# または、より耐性のあるダウンロードパス：
USE_HFD=1 bash scripts/setup/download_models.sh
source .venv/bin/activate
python scripts/dev/model_smoke.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

## OpenClaw統合

OpenClawをAsterのOpenAI互換ベースURLとモデルIDに指定します。Asterは繰り返されるシステム/ツールプレフィックスと長期エージェントセッション用に構築されているため、安定したスキャフォルディングと長いコンテキスト再利用を備えたワークロードから特に利益を得るはずです。

## プロジェクトドキュメント

- `docs/ROADMAP.md` — 長期的なアーキテクチャ進化計画
- `docs/OPENAI_COMPAT.md` — 互換性の境界とデバッグ拡張ルール
- `docs/DEBUGGING.md` — オペレーターデバッグガイド
- `docs/OPERATIONS.md` — 日常的なサービス運用
- `docs/DEVELOPMENT.md` — 開発ガイド

## ライセンス

MIT License - [LICENSE](LICENSE)を参照してください

## 貢献

貢献を歓迎します！[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)を参照して、貢献ガイドラインを確認してください。
