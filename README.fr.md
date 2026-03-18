<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">
  
  # Aster
  
  **Runtime d'inférence LLM local optimisé pour Apple Silicon**
  
  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster est un runtime d'inférence LLM local optimisé pour Apple Silicon, conçu pour les charges de travail de contexte long et les agents de style OpenClaw.

## Pourquoi Aster

Aster est optimisé pour :

- Les invites énormes et les préfixes longs répétés
- Les invites d'agents intensives en outils
- Les longues conversations
- Le service local continu en arrière-plan
- La sélection de politique d'exécution validée par benchmark
- Le déploiement Apple Silicon + MLX

Il expose une API compatible avec OpenAI et traite les optimisations avancées comme des stratégies candidates, pas comme un dogme. Le décodage spéculatif, la mise en cache des préfixes, le traitement par lots, la planification et la cadence de streaming sont tous comparés et sélectionnés en fonction des performances locales mesurées et de la stabilité.

## Caractéristiques principales

- API compatible avec OpenAI avec points de terminaison de streaming et non-streaming
- Séparation explicite prefill/decode
- Planificateur adaptatif avec traitement par lots conscient de la file d'attente
- Abstraction du gestionnaire KV paginé
- Cache de préfixe automatique avec hachage déterministe
- Contrôleur de décodage spéculatif avec secours de désactivation automatique
- Sous-système de benchmark/autotuning qui persiste le profil le plus rapide et stable
- Journaux structurés, métriques, supervision et rapports de disponibilité/santé

## Démarrage rapide

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python -m aster --config configs/config.yaml
```

## Version Python

Aster cible Python moderne et doit être exécuté sur Python 3.13.x si disponible (3.12+ minimum). Le Python système macOS est considéré comme non pris en charge pour ce projet.

## Points de terminaison API

- `GET /health` - Vérification de santé
- `GET /ready` - Vérification de disponibilité
- `GET /metrics` - Métriques Prometheus
- `GET /v1/models` - Liste des modèles
- `POST /v1/chat/completions` - Complétion de chat
- `POST /v1/completions` - Complétion de texte

Notes de compatibilité :
- Consultez `docs/OPENAI_COMPAT.md` pour le contrat de compatibilité par défaut d'Aster et les extensions de débogage optionnelles.

## Philosophie de benchmarking

L'autotuning au démarrage peut exécuter un court benchmark de préchauffage pour choisir la politique la plus rapide et stable. Le sous-système de benchmark compare :

- Décodage spéculatif activé/désactivé
- Nombres de jetons de brouillon
- Cache de préfixe activé/désactivé
- Fenêtres de traitement par lots
- Limites de traitement par lots
- Tailles de page
- Modes de planification
- Cadence de vidage du streaming

Les profils sont persistés et utilisés lors des démarrages ultérieurs.

## Notes de tuning Apple Silicon

- Favoriser la préallocation et les pools de pages plutôt que les allocations dynamiques répétées
- Utiliser avec prudence la résidence du modèle MLX pour éviter le thrashing de mémoire unifiée
- Benchmarker la mise en cache des préfixes et le décodage spéculatif par machine
- Garder les chemins chauds Python petits ; déplacer la coordination vers des boucles stables
- Prioriser la latence cohérente du premier jeton sous les invites longues

## Philosophie d'optimisation dynamique

Aster n'active que les optimisations qui s'avèrent bénéfiques sur la machine locale :

- Le décodage spéculatif peut être désactivé globalement ou par classe de requête
- Le cache de préfixe peut être réduit ou désactivé lorsque le taux de succès est faible ou que la pression mémoire augmente
- Les fenêtres de traitement par lots se réduisent automatiquement lorsque la latence augmente
- Les profils de secours sont sélectionnés lorsque l'instabilité ou la régression est détectée

## Chemins des modèles

`model.path` et `model.draft_path` peuvent être :
- Chemins locaux absolus vers les répertoires de modèles convertis par MLX
- ID de dépôt Hugging Face compatibles chargeables par `mlx-lm`

Pour la configuration de production prévue, préférez les répertoires convertis par MLX locaux pour le modèle cible 9B et le modèle de brouillon 0.8B.

Commandes utiles de configuration et de validation :

```bash
bash scripts/setup/download_models.sh
# ou, pour un chemin plus résilient aux téléchargements :
USE_HFD=1 bash scripts/setup/download_models.sh
source .venv/bin/activate
python scripts/dev/model_smoke.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

## Intégration OpenClaw

Pointez OpenClaw vers l'URL de base compatible OpenAI d'Aster et l'ID du modèle. Aster est construit pour les préfixes système/outil répétés et les sessions d'agent de longue durée, il devrait donc bénéficier particulièrement des charges de travail avec un échafaudage stable et une réutilisation de contexte long.

## Documentation du projet

- `docs/ROADMAP.md` — Plan d'évolution architecturale à long terme
- `docs/OPENAI_COMPAT.md` — Limite de compatibilité et règles d'extension de débogage
- `docs/DEBUGGING.md` — Guide de débogage de l'opérateur
- `docs/OPERATIONS.md` — Opérations de service quotidiennes
- `docs/DEVELOPMENT.md` — Guide de développement

## Licence

MIT License - Consultez [LICENSE](LICENSE)

## Contribuer

Les contributions sont bienvenues ! Consultez [CONTRIBUTING.fr.md](CONTRIBUTING.fr.md) pour les directives de contribution.
