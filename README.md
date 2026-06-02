# Detector de Señas para Truco

App de escritorio que detecta gestos faciales en tiempo real a través de la webcam para hacer señas en el **Truco argentino** sin necesidad de hablar.

## Gestos detectados

| Gesto | Carta / Seña |
|---|---|
| Ambas cejas levantadas | Macho / 1 de espadas |
| Guiñar (un ojo) | Hembra / 1 de basto |
| Mover comisura derecha | 7 de espadas |
| Mover comisura izquierda | 7 de oro |
| Morder labio inferior | 3 |
| Beso | 2 |
| Abrir la boca | 1 de oro o copas |

## Cómo funciona

- Usa **FaceLandmarker** de **MediaPipe** para detectar 478 puntos faciales.
- Extrae 43 **ARKit blendshapes** por frame para analizar expresiones.
- Las 7 señas se evalúan por frame con histéresis para evitar falsos positivos.
- Ventana redimensionable con el mouse, cierra con `q` o con el botón **X**.

## Requisitos

- Python 3.8 o superior
- Webcam

## Instalación

 En bash:

`pip install -r requirements.txt`

El modelo face_landmarker.task se descarga automáticamente la primera vez que se ejecuta.

## Uso

`python main.py`   
## Configuración

En main.py se puede ajustar:
- CAM_WIDTH / CAM_HEIGHT — resolución deseada (por defecto 1920×1080).
- CAM_INDEX — selección de cámara (también vía argumento).
En gesture_detector.py se pueden modificar los umbrales de detección (brow_threshold, wink_threshold, kiss_threshold, jaw_open_threshold, etc.).
## Controles
- q — Salir
- Arrastrar bordes de la ventana — Redimensionar

## Nota
 El modelo face_landmarker.task está en .gitignore (no se trackea).
