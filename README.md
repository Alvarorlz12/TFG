# TFG - Análisis Avanzado de Imágenes Médicas con Deep Learning para Cáncer de Páncreas

Repositorio del Trabajo Fin de Grado.

> [!IMPORTANT]
> El conjunto de datos no es público. No se pueden reproducir los experimentos.

## Características

- Implementación de modelos, funciones de pérdida y métricas para segmentación semántica de imágenes médicas.
- Utilidades para los entrenamientos: Trainer, Evaluator, Notifier...
- Integración con MONAI

## Instalación

```bash
git clone https://github.com/Alvarorlz12/TFG.git
cd TFG/digipanca
pip install -e .
```

## Uso

Usar los scripts de `digipanca/src/scripts/` para realizar entrenamientos, evaluaciones e inferencias,
definiendo previamente las configuraciones en formato `.yaml` tanto para el uso de `src` como para el apoyo con el bundle de MONAI.

Para más información sobre MONAI: https://github.com/Project-MONAI/MONAI

> [!NOTE]
> Para utilizar el Notificador, es necesario disponer de una API key de Google Cloud para Google Sheets y un token de bot de Telegram

## Licencia

Este proyecto está bajo la licencia Apache 2.0.