<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">
  
  # Aster
  
  **Tiempo de ejecución de inferencia LLM local optimizado para Apple Silicon**
  
  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster es un tiempo de ejecución de inferencia LLM local optimizado para Apple Silicon, diseñado para cargas de trabajo de contexto largo y estilo OpenClaw Agent.

## Por qué Aster

Aster está optimizado para:

- Indicaciones enormes y prefijos largos repetidos
- Indicaciones de agentes intensivas en herramientas
- Conversaciones largas
- Servicio local continuo en segundo plano
- Selección de política de tiempo de ejecución validada por puntos de referencia
- Implementación de Apple Silicon + MLX

Expone una API compatible con OpenAI y trata las optimizaciones avanzadas como estrategias candidatas, no como dogma. La decodificación especulativa, el almacenamiento en caché de prefijos, el procesamiento por lotes, la programación y la cadencia de transmisión se comparan y seleccionan en función del rendimiento local medido y la estabilidad.

## Características principales

- API compatible con OpenAI con puntos finales de transmisión y no transmisión
- División explícita de prefill/decode
- Programador adaptativo con procesamiento por lotes consciente de la cola
- Abstracción del administrador de KV paginado
- Caché de prefijo automático con hash determinista
- Controlador de decodificación especulativa con respaldo de deshabilitación automática
- Subsistema de referencia/autoajuste que persiste el perfil más rápido y estable
- Registros estructurados, métricas, supervisión e informes de disponibilidad/salud

## Inicio rápido

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python -m aster --config configs/config.yaml
```

## Versión de Python

Aster se dirige a Python moderno y debe ejecutarse en Python 3.13.x cuando esté disponible (3.12+ mínimo). El Python del sistema macOS se considera no compatible con este proyecto.

## Puntos finales de API

- `GET /health` - Verificación de salud
- `GET /ready` - Verificación de disponibilidad
- `GET /metrics` - Métricas de Prometheus
- `GET /v1/models` - Lista de modelos
- `POST /v1/chat/completions` - Finalización de chat
- `POST /v1/completions` - Finalización de texto

Notas de compatibilidad:
- Consulte `docs/OPENAI_COMPAT.md` para el contrato de compatibilidad predeterminado de Aster y las extensiones de depuración de inclusión.

## Filosofía de evaluación comparativa

El autoajuste de inicio puede ejecutar una evaluación comparativa de calentamiento breve para elegir la política más rápida y estable. El subsistema de evaluación comparativa compara:

- Decodificación especulativa activada/desactivada
- Recuentos de tokens de borrador
- Caché de prefijo activado/desactivado
- Ventanas de lotes
- Límites de lotes
- Tamaños de página
- Modos de programación
- Cadencia de descarga de transmisión

Los perfiles se persisten y se utilizan en inicios posteriores.

## Notas de ajuste de Apple Silicon

- Favorecer la preasignación y grupos de páginas sobre asignaciones dinámicas repetidas
- Usar cuidadosamente la residencia del modelo MLX para evitar trashing de memoria unificada
- Comparar almacenamiento en caché de prefijos y decodificación especulativa por máquina
- Mantener pequeñas las rutas activas de Python; mover la coordinación a bucles estables
- Priorizar la latencia consistente del primer token bajo indicaciones largas

## Filosofía de optimización dinámica

Aster solo habilita optimizaciones que demuestran ser beneficiosas en la máquina local:

- La decodificación especulativa se puede deshabilitar globalmente o por clase de solicitud
- El caché de prefijo se puede reducir o deshabilitar cuando la tasa de aciertos es baja o aumenta la presión de memoria
- Las ventanas de lotes se reducen automáticamente cuando aumenta la latencia
- Los perfiles de respaldo se seleccionan cuando se detecta inestabilidad o regresión

## Rutas de modelo

`model.path` y `model.draft_path` pueden ser:
- Rutas locales absolutas a directorios de modelos convertidos por MLX
- ID de repositorio de Hugging Face compatibles cargables por `mlx-lm`

Para la configuración de producción prevista, prefiera directorios convertidos por MLX locales para el modelo de destino de 9B y el modelo de borrador de 0.8B.

Comandos útiles de configuración y validación:

```bash
bash scripts/setup/download_models.sh
# o, para una ruta más resistente a descargas:
USE_HFD=1 bash scripts/setup/download_models.sh
source .venv/bin/activate
python scripts/dev/model_smoke.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

## Integración de OpenClaw

Apunte OpenClaw a la URL base compatible con OpenAI de Aster e ID de modelo. Aster está construido para prefijos de sistema/herramienta repetidos y sesiones de agente de larga duración, por lo que debería beneficiarse particularmente de cargas de trabajo con andamiaje estable y reutilización de contexto largo.

## Documentación del proyecto

- `docs/ROADMAP.md` — Plan de evolución arquitectónica a largo plazo
- `docs/OPENAI_COMPAT.md` — Límite de compatibilidad y reglas de extensión de depuración
- `docs/DEBUGGING.md` — Guía de depuración del operador
- `docs/OPERATIONS.md` — Operaciones de servicio diarias
- `docs/DEVELOPMENT.md` — Guía de desarrollo

## Licencia

MIT License - Consulte [LICENSE](LICENSE)

## Contribuir

¡Las contribuciones son bienvenidas! Consulte [CONTRIBUTING.es.md](CONTRIBUTING.es.md) para obtener las directrices de contribución.
