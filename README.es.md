<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **Tiempo de ejecución de inferencia LLM local de Apple Silicon orientado a la producción**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster es un tiempo de ejecución de inferencia LLM local de Apple Silicon orientado a la producción, optimizado para cargas de trabajo de agentes de contexto largo y estilo OpenClaw.

## Por qué Aster

Aster está optimizado para:

- indicaciones enormes y prefijos largos repetidos
- indicaciones de agentes intensivas en herramientas
- conversaciones largas
- servicio de fondo local continuo
- selección de política de tiempo de ejecución validada por punto de referencia
- implementación de Apple Silicon + MLX

Expone una API compatible con OpenAI y trata las optimizaciones avanzadas como estrategias candidatas, no como dogma. La decodificación especulativa, el almacenamiento en caché de prefijos, el procesamiento por lotes, la programación y la cadencia de transmisión se comparan y se seleccionan en función del rendimiento local medido y la estabilidad.

## Ideas principales

- API compatible con OpenAI con puntos finales de transmisión y no transmisión
- división explícita de prefill/decodificación
- programador adaptativo con procesamiento por lotes consciente de la cola
- abstracción del administrador de KV paginado
- caché de prefijo automático con hash determinista
- controlador de decodificación especulativa con respaldo de deshabilitación automática
- subsistema de punto de referencia/autoajuste que persiste el perfil más rápido y estable
- registros estructurados, métricas, supervisión e informes de disponibilidad/salud

## Inicio rápido

```bash
cd /Users/eitan/Documents/Projects/Python/Aster

# Crear entorno virtual
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# Instalar dependencias (incluido mlx-audio para ASR/TTS)
python -m pip install -r requirements.txt

# Descargar modelos (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# Iniciar el servidor
python -m aster --config configs/config.yaml
```

La API estará disponible en `http://127.0.0.1:8080`

### Verificar instalación

```bash
# Verificar salud
curl http://127.0.0.1:8080/health

# Probar inferencia LLM
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 100
  }'

# Probar ASR (conversión de voz a texto)
python scripts/test_audio_cli.py --tts "Hola mundo" --output test.wav
python scripts/test_audio_cli.py --asr test.wav

# Probar canalización de extremo a extremo
python scripts/test_audio_cli.py --pipeline "Este es una prueba"
```

## Versión de Python

Aster se dirige a Python moderno y debe ejecutarse en Python 3.13.x cuando esté disponible (mínimo 3.12+). El Python del sistema macOS se considera no compatible con este proyecto.

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` — Inferencia de chat LLM
- `POST /v1/completions` — Finalización de texto LLM
- `POST /v1/audio/transcriptions` — ASR (conversión de voz a texto)
- `POST /v1/audio/speech` — TTS (conversión de texto a voz)

Notas de compatibilidad:
- Consulte `docs/api/OPENAI_COMPAT.md` para el contrato de compatibilidad predeterminado de Aster y extensiones de depuración opcionales.

## Servicios de audio (ASR y TTS)

Aster incluye reconocimiento de voz y síntesis integrados impulsados por modelos Qwen3:

### ASR (conversión de voz a texto)
- Modelo: Qwen3-ASR-0.6B (0,66 GB)
- Admite múltiples idiomas
- Transcripción local rápida

### TTS (conversión de texto a voz)
- Modelo base: Qwen3-TTS-0.6B (1,59 GB)
- Modelo CustomVoice: Qwen3-TTS-CustomVoice-0.6B (opcional, para clonación de voz)
- Velocidad de voz ajustable
- Clonación de voz con audio de referencia

### Ejemplos de API de audio

**TTS (conversión de texto a voz):**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "Hola, esto es una prueba",
    "voice": "default",
    "speed": 1.0
  }' \
  --output output.wav
```

**ASR (conversión de voz a texto):**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

### Prueba de audio

Utilice la herramienta de prueba CLI proporcionada:
```bash
# Probar TTS
python scripts/test_audio_cli.py --tts "Hola mundo" --output output.wav

# Probar ASR
python scripts/test_audio_cli.py --asr output.wav

# Probar canalización de extremo a extremo (TTS -> ASR)
python scripts/test_audio_cli.py --pipeline "Mensaje de prueba"

# Ejecutar suite de pruebas completa
pytest tests/test_audio_services.py -v -s
```

Consulte `docs/guides/DEPLOYMENT.md` para obtener documentación detallada del servicio de audio.

## Filosofía de evaluación comparativa

El autoajuste de inicio puede ejecutar un punto de referencia de calentamiento corto para elegir la política más rápida y estable. El subsistema de punto de referencia compara:

- decodificación especulativa activada/desactivada
- recuentos de tokens de borrador
- almacenamiento en caché de prefijo activado/desactivado
- ventanas de procesamiento por lotes
- límites de lotes
- tamaños de página
- modos de programación
- cadencia de descarga de transmisión

Los perfiles se conservan y se utilizan en inicios posteriores.

## Notas de ajuste de Apple Silicon

- favorecer la preasignación y los grupos de páginas sobre asignaciones dinámicas repetidas
- usar cuidadosamente la residencia del modelo MLX para evitar trashing de memoria unificada
- punto de referencia almacenamiento en caché de prefijo y decodificación especulativa por máquina
- mantener las rutas activas de Python pequeñas; mover la coordinación a bucles estables
- priorizar la latencia consistente del primer token bajo indicaciones largas

## Filosofía de optimización dinámica

Aster solo habilita optimizaciones que demuestran ser beneficiosas en la máquina local:

- la decodificación especulativa se puede deshabilitar globalmente o por clase de solicitud
- el caché de prefijo se puede reducir o deshabilitar cuando la tasa de aciertos es baja o aumenta la presión de memoria
- las ventanas de procesamiento por lotes se reducen automáticamente cuando aumenta la latencia
- los perfiles de respaldo se seleccionan cuando se detecta inestabilidad o regresiones

## Configuración del modelo

Descarga de modelo de un clic con aceleración hfd + aria2:

```bash
# Descargar todos los modelos requeridos (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# O usar Python directamente para más control
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

Consulte `scripts/setup/README-model-download.md` para obtener instrucciones detalladas.

## Rutas de modelo

`model.path` y `model.draft_path` pueden ser:
- rutas locales absolutas a directorios de modelos convertidos a MLX
- ID de repositorio de Hugging Face compatibles cargables por `mlx-lm`

Para producción, prefiera directorios convertidos a MLX locales. Actualizar `configs/config.yaml`:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## Integración de OpenClaw

Apunte OpenClaw a la URL base compatible con OpenAI de Aster e ID de modelo. Aster está construido para prefijos de sistema/herramienta repetidos y sesiones de agentes de larga duración, por lo que debería beneficiarse especialmente de cargas de trabajo con andamios estables y reutilización de contexto largo.

## Documentos de orientación del proyecto

- `docs/guides/QUICK_START_MODELS.md` — Guía rápida de descarga de modelos
- `docs/reference/MODEL_SETUP.md` — Configuración detallada y solución de problemas
- `docs/development/MODEL_DOWNLOAD_ARCHITECTURE.md` — Diseño del sistema
- `docs/reference/ROADMAP.md` — Plan de evolución arquitectónica a largo plazo
- `docs/api/OPENAI_COMPAT.md` — Límite de compatibilidad y extensiones de depuración
- `docs/development/DEBUGGING.md` — Guía de depuración del operador
- `docs/operations/OPERATIONS.md` — Operaciones de servicio diarias
- `docs/guides/BENCHMARK_GUIDE.md` — Guía de punto de referencia de rendimiento
- `docs/guides/BACKGROUND_SERVICE_SETUP.md` — Configuración del servicio de fondo
- `DOCS.md` — Navegación completa de documentación
