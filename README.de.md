<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **Produktionsorientierte Apple Silicon lokale LLM-Inferenzlaufzeit**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster ist eine produktionsorientierte Apple Silicon lokale LLM-Inferenzlaufzeit, die für langkontextuelle, OpenClaw-ähnliche Agent-Workloads optimiert ist.

## Warum Aster

Aster ist optimiert für:

- riesige Eingabeaufforderungen und wiederholte lange Präfixe
- werkzeugintensive Agent-Eingabeaufforderungen
- lange Gespräche
- kontinuierliche lokale Hintergrundverwaltung
- durch Benchmarking validierte Laufzeit-Richtlinienauswahl
- Apple Silicon + MLX-Bereitstellung

Es stellt eine OpenAI-kompatible API bereit und behandelt erweiterte Optimierungen als Kandidatenstrategien, nicht als Dogma. Spekulatives Dekodieren, Präfix-Caching, Batch-Verarbeitung, Planung und Streaming-Kadenz werden alle verglichen und basierend auf gemessenem lokalem Leistung und Stabilität ausgewählt.

## Kernideen

- OpenAI-kompatible API mit Streaming- und Nicht-Streaming-Endpunkten
- explizite Prefill/Decode-Aufteilung
- adaptiver Scheduler mit warteschlangen-bewusstem Batch-Processing
- Paged-KV-Manager-Abstraktion
- automatisches Präfix-Caching mit deterministischem Hashing
- spekulativer Dekodierungs-Controller mit automatischem Deaktivierungs-Fallback
- Benchmark/Autotuning-Subsystem, das das schnellste stabile Profil beibehält
- strukturierte Protokolle, Metriken, Überwachung und Bereitschafts-/Gesundheitsberichte

## Schnellstart

```bash
cd /Users/eitan/Documents/Projects/Python/Aster

# Virtuelle Umgebung erstellen
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren (einschließlich mlx-audio für ASR/TTS)
python -m pip install -r requirements.txt

# Modelle herunterladen (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# Server starten
python -m aster --config configs/config.yaml
```

Die API ist unter `http://127.0.0.1:8080` verfügbar

### Installation überprüfen

```bash
# Gesundheit überprüfen
curl http://127.0.0.1:8080/health

# LLM-Inferenz testen
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "Hallo"}],
    "max_tokens": 100
  }'

# ASR testen (Sprache-zu-Text)
python scripts/test_audio_cli.py --tts "Hallo Welt" --output test.wav
python scripts/test_audio_cli.py --asr test.wav

# End-to-End-Pipeline testen
python scripts/test_audio_cli.py --pipeline "Dies ist ein Test"
```

## Python-Version

Aster zielt auf modernes Python ab und sollte auf Python 3.13.x ausgeführt werden, wenn verfügbar (mindestens 3.12+). Das macOS-System-Python wird für dieses Projekt als nicht unterstützt betrachtet.

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` — LLM-Chat-Inferenz
- `POST /v1/completions` — LLM-Textvervollständigung
- `POST /v1/audio/transcriptions` — ASR (Sprache-zu-Text)
- `POST /v1/audio/speech` — TTS (Text-zu-Sprache)

Kompatibilitätshinweise:
- Siehe `docs/api/OPENAI_COMPAT.md` für Asters Standard-Kompatibilitätsvertrag und optionale Debug-Erweiterungen.

## Audio-Dienste (ASR & TTS)

Aster umfasst integrierte Spracherkennung und Synthese, die von Qwen3-Modellen angetrieben werden:

### ASR (Sprache-zu-Text)
- Modell: Qwen3-ASR-0.6B (0,66 GB)
- Unterstützt mehrere Sprachen
- Schnelle lokale Transkription

### TTS (Text-zu-Sprache)
- Basismodell: Qwen3-TTS-0.6B (1,59 GB)
- CustomVoice-Modell: Qwen3-TTS-CustomVoice-0.6B (optional, für Sprachklonen)
- Einstellbare Sprechgeschwindigkeit
- Sprachklonen mit Referenzaudio

### Audio-API-Beispiele

**TTS (Text-zu-Sprache):**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "Hallo, dies ist ein Test",
    "voice": "default",
    "speed": 1.0
  }' \
  --output output.wav
```

**ASR (Sprache-zu-Text):**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

### Audio-Test

Verwenden Sie das bereitgestellte CLI-Test-Tool:
```bash
# TTS testen
python scripts/test_audio_cli.py --tts "Hallo Welt" --output output.wav

# ASR testen
python scripts/test_audio_cli.py --asr output.wav

# End-to-End-Pipeline testen (TTS -> ASR)
python scripts/test_audio_cli.py --pipeline "Testnachricht"

# Vollständige Test-Suite ausführen
pytest tests/test_audio_services.py -v -s
```

Siehe `docs/guides/DEPLOYMENT.md` für detaillierte Audio-Service-Dokumentation.

## Benchmark-Philosophie

Das Autotuning beim Start kann einen kurzen Warm-up-Benchmark ausführen, um die schnellste stabile Richtlinie auszuwählen. Das Benchmark-Subsystem vergleicht:

- spekulatives Dekodieren an/aus
- Anzahl der Entwurfs-Token
- Präfix-Caching an/aus
- Batch-Fenster
- Batch-Obergrenzen
- Seitengröße
- Planungsmodi
- Streaming-Flush-Kadenz

Profile werden beibehalten und bei nachfolgenden Starts verwendet.

## Apple Silicon Tuning-Hinweise

- Präallokation und Seiten-Pools gegenüber wiederholten dynamischen Zuordnungen bevorzugen
- MLX-Modell-Residenz sorgfältig verwenden, um Unified-Memory-Thrashing zu vermeiden
- Präfix-Caching und spekulatives Dekodieren pro Maschine benchmarken
- Python-Hot-Paths klein halten; Koordination in stabile Schleifen verschieben
- konsistente First-Token-Latenz unter langen Eingabeaufforderungen priorisieren

## Dynamische Optimierungsphilosophie

Aster aktiviert nur Optimierungen, die sich auf der lokalen Maschine als vorteilhaft erweisen:

- spekulatives Dekodieren kann global oder pro Anfrageklasse deaktiviert werden
- Präfix-Cache kann reduziert oder deaktiviert werden, wenn die Hit-Rate niedrig ist oder der Speicherdruck steigt
- Batch-Fenster schrumpfen automatisch, wenn die Latenz steigt
- Fallback-Profile werden ausgewählt, wenn Instabilität oder Regressionen erkannt werden

## Modell-Setup

One-Click-Modell-Download mit hfd + aria2-Beschleunigung:

```bash
# Alle erforderlichen Modelle herunterladen (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# Oder Python direkt für mehr Kontrolle verwenden
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

Siehe `scripts/setup/README-model-download.md` für detaillierte Anweisungen.

## Modell-Pfade

`model.path` und `model.draft_path` können sein:
- absolute lokale Pfade zu MLX-konvertierten Modellverzeichnissen
- kompatible Hugging Face Repo-IDs, die von `mlx-lm` geladen werden können

Für die Produktion bevorzugen Sie lokale MLX-konvertierte Verzeichnisse. Aktualisieren Sie `configs/config.yaml`:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## OpenClaw-Integration

Zeigen Sie OpenClaw auf Asters OpenAI-kompatible Basis-URL und Modell-ID. Aster ist für wiederholte System-/Tool-Präfixe und langlebige Agent-Sitzungen konzipiert, daher sollte es besonders von Workloads mit stabilen Gerüsten und langem Kontext-Reuse profitieren.

## Projekt-Leitfäden

- `docs/guides/QUICK_START_MODELS.md` — Schnellstart-Leitfaden für Modell-Download
- `docs/reference/MODEL_SETUP.md` — Detaillierte Einrichtung und Fehlerbehebung
- `docs/development/MODEL_DOWNLOAD_ARCHITECTURE.md` — Systemdesign
- `docs/reference/ROADMAP.md` — Langfristiger Architektur-Evolutionsplan
- `docs/api/OPENAI_COMPAT.md` — Kompatibilitätsgrenze und Debug-Erweiterungen
- `docs/development/DEBUGGING.md` — Operator-Debugging-Leitfaden
- `docs/operations/OPERATIONS.md` — Tägliche Service-Operationen
- `docs/guides/BENCHMARK_GUIDE.md` — Performance-Benchmark-Leitfaden
- `docs/guides/BACKGROUND_SERVICE_SETUP.md` — Hintergrund-Service-Einrichtung
- `DOCS.md` — Vollständige Dokumentationsnavigation
