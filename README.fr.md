<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **Environnement d'exécution d'inférence LLM local Apple Silicon orienté vers la production**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster est un environnement d'exécution d'inférence LLM local Apple Silicon orienté vers la production, optimisé pour les charges de travail d'agents de contexte long et de style OpenClaw.

## Pourquoi Aster

Aster est optimisé pour :

- les invites énormes et les préfixes longs répétés
- les invites d'agents intensives en outils
- les longues conversations
- le service de fond local continu
- la sélection de politique d'exécution validée par benchmark
- le déploiement Apple Silicon + MLX

Il expose une API compatible OpenAI et traite les optimisations avancées comme des stratégies candidates, pas comme un dogme. Le décodage spéculatif, la mise en cache des préfixes, le traitement par lots, la planification et la cadence de diffusion en continu sont tous comparés et sélectionnés en fonction des performances locales mesurées et de la stabilité.

## Idées principales

- API compatible OpenAI avec points de terminaison de diffusion en continu et non-diffusion en continu
- division explicite préfill/décodage
- planificateur adaptatif avec traitement par lots conscient de la file d'attente
- abstraction du gestionnaire KV paginé
- cache de préfixe automatique avec hachage déterministe
- contrôleur de décodage spéculatif avec secours de désactivation automatique
- sous-système de benchmark/autoréglage qui persiste le profil le plus rapide et stable
- journaux structurés, métriques, supervision et rapports de disponibilité/santé

## Démarrage rapide

```bash
cd /Users/eitan/Documents/Projects/Python/Aster

# Créer un environnement virtuel
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# Installer les dépendances (y compris mlx-audio pour ASR/TTS)
python -m pip install -r requirements.txt

# Télécharger les modèles (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# Démarrer le serveur
python -m aster --config configs/config.yaml
```

L'API sera disponible à `http://127.0.0.1:8080`

### Vérifier l'installation

```bash
# Vérifier la santé
curl http://127.0.0.1:8080/health

# Tester l'inférence LLM
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "Bonjour"}],
    "max_tokens": 100
  }'

# Tester ASR (conversion parole-texte)
python scripts/test_audio_cli.py --tts "Bonjour le monde" --output test.wav
python scripts/test_audio_cli.py --asr test.wav

# Tester le pipeline de bout en bout
python scripts/test_audio_cli.py --pipeline "Ceci est un test"
```

## Version Python

Aster cible Python moderne et doit être exécuté sur Python 3.13.x si disponible (minimum 3.12+). Le Python système macOS est considéré comme non pris en charge pour ce projet.

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` — Inférence de chat LLM
- `POST /v1/completions` — Complément de texte LLM
- `POST /v1/audio/transcriptions` — ASR (conversion parole-texte)
- `POST /v1/audio/speech` — TTS (conversion texte-parole)

Notes de compatibilité :
- Voir `docs/api/OPENAI_COMPAT.md` pour le contrat de compatibilité par défaut d'Aster et les extensions de débogage optionnelles.

## Services audio (ASR et TTS)

Aster inclut la reconnaissance vocale et la synthèse intégrées alimentées par les modèles Qwen3 :

### ASR (conversion parole-texte)
- Modèle : Qwen3-ASR-0.6B (0,66 Go)
- Prend en charge plusieurs langues
- Transcription locale rapide

### TTS (conversion texte-parole)
- Modèle de base : Qwen3-TTS-0.6B (1,59 Go)
- Modèle CustomVoice : Qwen3-TTS-CustomVoice-0.6B (optionnel, pour le clonage vocal)
- Vitesse de parole ajustable
- Clonage vocal avec audio de référence

### Exemples d'API audio

**TTS (conversion texte-parole) :**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "Bonjour, ceci est un test",
    "voice": "default",
    "speed": 1.0
  }' \
  --output output.wav
```

**ASR (conversion parole-texte) :**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

### Test audio

Utilisez l'outil de test CLI fourni :
```bash
# Tester TTS
python scripts/test_audio_cli.py --tts "Bonjour le monde" --output output.wav

# Tester ASR
python scripts/test_audio_cli.py --asr output.wav

# Tester le pipeline de bout en bout (TTS -> ASR)
python scripts/test_audio_cli.py --pipeline "Message de test"

# Exécuter la suite de tests complète
pytest tests/test_audio_services.py -v -s
```

Voir `docs/guides/DEPLOYMENT.md` pour la documentation détaillée du service audio.

## Philosophie de benchmarking

L'autoréglage au démarrage peut exécuter un court benchmark de préchauffage pour choisir la politique la plus rapide et stable. Le sous-système de benchmark compare :

- décodage spéculatif activé/désactivé
- nombres de jetons de brouillon
- mise en cache des préfixes activée/désactivée
- fenêtres de traitement par lots
- plafonds de lots
- tailles de page
- modes de planification
- cadence de vidage de diffusion en continu

Les profils sont conservés et utilisés lors des démarrages ultérieurs.

## Notes de réglage Apple Silicon

- favoriser la préallocation et les pools de pages plutôt que les allocations dynamiques répétées
- utiliser avec prudence la résidence du modèle MLX pour éviter le thrashing de mémoire unifiée
- benchmark la mise en cache des préfixes et le décodage spéculatif par machine
- garder les chemins chauds Python petits ; déplacer la coordination dans des boucles stables
- prioriser la latence cohérente du premier jeton sous les invites longues

## Philosophie d'optimisation dynamique

Aster n'active que les optimisations qui s'avèrent bénéfiques sur la machine locale :

- le décodage spéculatif peut être désactivé globalement ou par classe de requête
- le cache de préfixe peut être réduit ou désactivé lorsque le taux de succès est faible ou que la pression mémoire augmente
- les fenêtres de traitement par lots se réduisent automatiquement lorsque la latence augmente
- les profils de secours sont sélectionnés lorsque l'instabilité ou les régressions sont détectées

## Configuration du modèle

Téléchargement de modèle en un clic avec accélération hfd + aria2 :

```bash
# Télécharger tous les modèles requis (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# Ou utiliser Python directement pour plus de contrôle
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

Voir `scripts/setup/README-model-download.md` pour les instructions détaillées.

## Chemins de modèle

`model.path` et `model.draft_path` peuvent être :
- chemins locaux absolus vers les répertoires de modèles convertis MLX
- ID de dépôt Hugging Face compatibles chargeables par `mlx-lm`

Pour la production, préférez les répertoires convertis MLX locaux. Mettre à jour `configs/config.yaml` :

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## Intégration OpenClaw

Pointez OpenClaw vers l'URL de base compatible OpenAI d'Aster et l'ID de modèle. Aster est construit pour les préfixes système/outil répétés et les sessions d'agent de longue durée, il devrait donc bénéficier particulièrement des charges de travail avec des échafaudages stables et la réutilisation de contexte long.

## Documents d'orientation du projet

- `docs/guides/QUICK_START_MODELS.md` — Guide de démarrage rapide du téléchargement de modèles
- `docs/reference/MODEL_SETUP.md` — Configuration détaillée et dépannage
- `docs/development/MODEL_DOWNLOAD_ARCHITECTURE.md` — Conception du système
- `docs/reference/ROADMAP.md` — Plan d'évolution architecturale à long terme
- `docs/api/OPENAI_COMPAT.md` — Limite de compatibilité et extensions de débogage
- `docs/development/DEBUGGING.md` — Guide de débogage de l'opérateur
- `docs/operations/OPERATIONS.md` — Opérations de service quotidiennes
- `docs/guides/BENCHMARK_GUIDE.md` — Guide de benchmark de performance
- `docs/guides/BACKGROUND_SERVICE_SETUP.md` — Configuration du service de fond
- `DOCS.md` — Navigation complète de la documentation
