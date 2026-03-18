# Aster

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)

Aster ist eine produktionsorientierte lokale LLM-Inferenzlaufzeit für Apple Silicon, die für Arbeitslasten mit langem Kontext und OpenClaw-ähnliche Agent-Workloads optimiert ist.

## Warum Aster

Aster ist optimiert für:

- Riesige Eingabeaufforderungen und wiederholte lange Präfixe
- Werkzeugintensive Agent-Eingabeaufforderungen
- Lange Gespräche
- Kontinuierliche lokale Hintergrundverwaltung
- Benchmark-validierte Laufzeit-Richtlinienauswahl
- Apple Silicon + MLX-Bereitstellung

Es stellt eine OpenAI-kompatible API bereit und behandelt fortgeschrittene Optimierungen als Kandidatenstrategien, nicht als Dogma. Spekulatives Dekodieren, Präfix-Caching, Batch-Verarbeitung, Planung und Streaming-Kadenz werden alle verglichen und basierend auf gemessener lokaler Leistung und Stabilität ausgewählt.

## Kernfunktionen

- OpenAI-kompatible API mit Streaming- und Nicht-Streaming-Endpunkten
- Explizite Prefill/Decode-Trennung
- Adaptiver Scheduler mit warteschlangen-bewusster Batch-Verarbeitung
- Abstraktion des paginierten KV-Managers
- Automatisches Präfix-Caching mit deterministischem Hashing
- Spekulativer Dekodierungscontroller mit automatischem Deaktivierungs-Fallback
- Benchmark-/Autotuning-Subsystem, das das schnellste stabile Profil beibehält
- Strukturierte Protokolle, Metriken, Überwachung und Verfügbarkeits-/Gesundheitsberichte

## Schnellstart

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python -m aster --config configs/config.yaml
```

## Python-Version

Aster zielt auf modernes Python ab und sollte auf Python 3.13.x (falls verfügbar) oder mindestens 3.12+ ausgeführt werden. Das macOS-System-Python wird für dieses Projekt nicht unterstützt.

## API-Endpunkte

- `GET /health` - Gesundheitsprüfung
- `GET /ready` - Bereitschaftsprüfung
- `GET /metrics` - Prometheus-Metriken
- `GET /v1/models` - Modelliste
- `POST /v1/chat/completions` - Chat-Vervollständigung
- `POST /v1/completions` - Text-Vervollständigung

Kompatibilitätshinweise:
- Siehe `docs/OPENAI_COMPAT.md` für Asters Standard-Kompatibilitätsvertrag und optionale Debug-Erweiterungen.

## Benchmark-Philosophie

Das Autotuning beim Start kann einen kurzen Aufwärm-Benchmark ausführen, um die schnellste stabile Richtlinie auszuwählen. Das Benchmark-Subsystem vergleicht:

- Spekulatives Dekodieren ein/aus
- Entwurfs-Token-Anzahl
- Präfix-Caching ein/aus
- Batch-Fenster
- Batch-Limits
- Seitengröße
- Planungsmodi
- Streaming-Flush-Kadenz

Profile werden beibehalten und bei nachfolgenden Starts verwendet.

## Apple Silicon Tuning-Hinweise

- Bevorzugen Sie Vorallokation und Seiten-Pools gegenüber wiederholten dynamischen Zuordnungen
- Verwenden Sie MLX-Modell-Residenz sorgfältig, um Unified-Memory-Thrashing zu vermeiden
- Benchmark-Präfix-Caching und spekulatives Dekodieren pro Maschine
- Halten Sie Python-Hot-Paths klein; verschieben Sie die Koordination in stabile Schleifen
- Priorisieren Sie konsistente First-Token-Latenz unter langen Eingabeaufforderungen

## Dynamische Optimierungsphilosophie

Aster aktiviert nur Optimierungen, die sich auf der lokalen Maschine als vorteilhaft erweisen:

- Spekulatives Dekodieren kann global oder pro Request-Klasse deaktiviert werden
- Präfix-Caching kann reduziert oder deaktiviert werden, wenn die Hit-Rate niedrig ist oder der Speicherdruck steigt
- Batch-Fenster schrumpfen automatisch, wenn die Latenz steigt
- Fallback-Profile werden ausgewählt, wenn Instabilität oder Regression erkannt wird

## Modellpfade

`model.path` und `model.draft_path` können sein:
- Absolute lokale Pfade zu MLX-konvertierten Modellverzeichnissen
- Kompatible Hugging Face-Repository-IDs, die von `mlx-lm` geladen werden können

Für die beabsichtigte Produktionseinrichtung bevorzugen Sie lokal MLX-konvertierte Verzeichnisse für sowohl das 9B-Zielmodell als auch das 0.8B-Entwurfsmodell.

Nützliche Setup- und Validierungsbefehle:

```bash
bash scripts/setup/download_models.sh
# oder für einen Download-resistenteren Pfad:
USE_HFD=1 bash scripts/setup/download_models.sh
source .venv/bin/activate
python scripts/dev/model_smoke.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

## OpenClaw-Integration

Zeigen Sie OpenClaw auf Asters OpenAI-kompatible Basis-URL und Modell-ID. Aster ist für wiederholte System-/Tool-Präfixe und langlebige Agent-Sitzungen konzipiert, daher sollte es besonders von Arbeitslasten mit stabiler Gerüstung und Langkontext-Wiederverwendung profitieren.

## Projektdokumentation

- `docs/ROADMAP.md` — Langfristiger Architektur-Evolutionsplan
- `docs/OPENAI_COMPAT.md` — Kompatibilitätsgrenze und Debug-Erweiterungsregeln
- `docs/DEBUGGING.md` — Operator-Debugging-Anleitung
- `docs/OPERATIONS.md` — Tägliche Serviceverwaltung
- `docs/DEVELOPMENT.md` — Entwicklungsanleitung

## Lizenz

MIT License - Siehe [LICENSE](LICENSE)

## Beitragen

Beiträge sind willkommen! Siehe [CONTRIBUTING.de.md](CONTRIBUTING.de.md) für Beitragsrichtlinien.
