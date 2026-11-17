# Preparar entrenamiento YOLO para terrenos

El análisis multimodal funciona mediante OpenAI. **YOLO todavía no está entrenado para este proyecto**. Ya se descargó un piloto público de residuos con anotaciones de origen TACO; no hay pesos propios ni un dataset local validado que cubra todos los daños. Instalar YOLO, descargar imágenes o cambiar `OPENAI_MODEL` no constituye entrenamiento.

## Piloto público de residuos

El preparador `scripts/prepare_taco_dataset.py` descarga un subconjunto de **300 imágenes** de TACO, preferentemente con tierra o vegetación y sin interiores según etiquetas de escena. Solo cubre **`0: residuos`**; no crea etiquetas de maleza o agua a partir de las apariencias.

Lote descargado y auditado el 16/11/2026:

| División | Imágenes |
| --- | ---: |
| Entrenamiento | 204 |
| Validación | 48 |
| Prueba | 48 |
| **Total** | **300** |

Contiene **781 cajas de residuos**. Se excluyeron tres posibles similitudes por dHash. La carpeta del piloto ocupa unos **40,45 MiB** y las fuentes conservadas unos **43,61 MiB**. La auditoría técnica pasó; la revisión visual completa, los negativos y la evaluación en terrenos locales siguen pendientes.

Destino por defecto:

```text
.data/datasets/terrenos/taco_residuos_piloto_v1/
  dataset.yaml
  manifest.json
  LICENSE-DATASET.txt
  README.md
  revision.html
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

- `revision.html` permite inspeccionar localmente todas las imágenes con sus cajas de origen superpuestas. **No marca las imágenes como revisadas automáticamente.** Comprueba cajas, contexto, datos personales, similitudes y pertinencia para terrenos.
- `manifest.json` conserva versión y URL de anotaciones, IDs y categorías originales, cambios, fuente por imagen, hashes, lotes y estado de revisión.
- El registro público TACO declara CC BY 4.0; se excluyen las imágenes que indican una licencia diferente (por ejemplo OpenLitterMap/ODbL), sin reinterpretarlas como CC BY. En las seleccionadas los metadatos pueden no especificar licencia individual: se registra esa limitación y hay que revisar derechos antes de redistribuir.
- Se normalizan las miniaturas oficiales de hasta 640 px, sin recortes ni giros no comprobados. Objetos muy pequeños pueden necesitar imágenes de mayor resolución.
- Los lotes de origen se separan entre entrenamiento, validación y prueba; el reparto es aproximadamente 70/20/10 por lotes, no necesariamente por número exacto de imágenes. Se excluyen duplicados exactos y posibles similitudes dHash. **TACO no proporciona IDs fiables de terreno/autor para garantizar independencia física**: la revisión de escenas sigue siendo necesaria.
- No se añaden negativos sin revisar, ni se declara que los resultados representan terrenos peruanos. Antes de entrenar, completa negativos revisados y una prueba local independiente.
- Las fuentes originales descargadas se conservan en `.data/datasets/fuentes/taco/`, con anotaciones y registro de licencia. No se descarga el archivo completo de varios GB ni se usan claves de OpenAI.

Para generar un lote nuevo (se rechaza sobrescribir un destino existente):

```powershell
.\.venv\Scripts\python.exe -B scripts\prepare_taco_dataset.py --count 300 --output .data\datasets\terrenos\taco_residuos_piloto_v2
```

Para auditar el lote preparado, sin red ni entrenamiento:

```powershell
.\.venv\Scripts\python.exe -B scripts\verify_dataset.py .data\datasets\terrenos\taco_residuos_piloto_v1
```

La auditoría comprueba pares imagen/etiqueta, coordenadas, hashes y separación por lote. No sustituye la revisión visual ni la de licencias. **No mezcles directamente los IDs de este piloto (`0: residuos`) con los del ejemplo multiclase de abajo (`0: maleza`)**: hay que remapear etiquetas al ampliar las clases.

## Datos que necesitamos

1. Fotos y videos de terrenos con autorización para utilizarlos. Evita personas, documentos y otros datos privados.
2. Cajas delimitadoras **revisadas por una persona** en cada ejemplo de las clases que se vayan a entrenar. No uses las respuestas de OpenAI o de un detector como verdad de referencia sin revisar.
3. Ejemplos variados de cada clase y ejemplos negativos (zonas sin ese problema). No fuerces una etiqueta si el problema no es visible. Agua acumulada, erosión y desniveles son indicios visuales, no diagnósticos del suelo.
4. Divisiones separadas de entrenamiento, validación y prueba. **Un mismo terreno o video debe quedar en una sola división**: dividir sus fotogramas entre entrenamiento y validación produce métricas engañosas.

Clases compatibles con la app:

| ID sugerido | Nombre exacto |
| --- | --- |
| 0 | `maleza` |
| 1 | `residuos` |
| 2 | `agua_acumulada` |
| 3 | `erosion_visible` |
| 4 | `acceso_deteriorado` |
| 5 | `cerco_danado` |
| 6 | `desnivel_visible` |

Se puede empezar con un subconjunto bien anotado, renumerado desde 0. `otro_visible` no es una clase de entrenamiento suficientemente definida.

## Formato local

```text
.data/datasets/terrenos/
  dataset.yaml
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Cada imagen tiene un `.txt` del mismo nombre. Formato YOLO por caja:

```text
id_clase centro_x centro_y ancho alto
```

Las coordenadas se normalizan de 0 a 1 respecto a la imagen. Una imagen negativa tiene un archivo de etiquetas vacío; no representa datos sin revisar.

Ejemplo de `dataset.yaml` (ajusta `path` a la carpeta real dentro de `.data`):

```yaml
path: D:/Tasacion AI/.data/datasets/terrenos
train: images/train
val: images/val
test: images/test
names:
  0: maleza
  1: residuos
  2: agua_acumulada
  3: erosion_visible
  4: acceso_deteriorado
  5: cerco_danado
  6: desnivel_visible
```

No se aporta un dataset ficticio ni se inventan pesos. Primero revisaremos cantidad/calidad de datos, etiquetas, particiones, licencia y CPU/GPU disponible; después fijaremos duración y parámetros del entrenamiento.

## Cuando los datos estén listos

Instala las dependencias opcionales y aporta pesos base `.pt` **de origen confiable** dentro de `.data/models/`. Los archivos `.pt` pueden ejecutar código al cargarse: no uses pesos desconocidos ni los subas desde la web de la app.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe scripts\train_yolo.py --dataset .data\datasets\terrenos\dataset.yaml --weights .data\models\base_confiable.pt --epochs 50 --device cpu
```

Es un ejemplo para una etapa posterior, no se ha ejecutado. Usa `--device 0` solo si tienes GPU compatible. Revisa la licencia de Ultralytics, de los pesos y de las imágenes antes de distribuir o usar comercialmente.

El script guarda resultados en `.data/runs/`. Antes de publicar el modelo, evalúa precision, recall y mAP por clase sobre terrenos reservados y revisa falsos positivos/negativos. **No basta con terminar las épocas**. Publicaremos una copia del `best.pt` elegido en `.data/models/terrenos/best.pt`, sin sobrescribir pesos anteriores sin respaldo. La app rechazará clases genéricas como `person` o `car`.

Los resultados de YOLO solo localizan regiones propuestas. Las medidas reales, causas del daño, intervenciones definitivas y precios siguen requiriendo revisión y mediciones adecuadas.

## Ejecutar el experimento de residuos en este equipo

El equipo detectado tiene un **Ryzen 5 2400G (4 núcleos/8 hilos), 8 GB de RAM y Radeon RX 570**. Se utiliza CPU, no CUDA. El piloto emplea **YOLO11 nano** y transferencia desde pesos base oficiales; no se entrena una red desde cero.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe -B scripts\download_yolo_weights.py
.\.venv\Scripts\python.exe -B scripts\train_yolo.py --dataset .data\datasets\terrenos\taco_residuos_piloto_v1\dataset.yaml --weights .data\models\base\yolo11n.pt --epochs 30 --imgsz 416 --batch 2 --threads 4 --freeze 10 --patience 8 --device cpu --allow-unreviewed-pilot
```

`--allow-unreviewed-pilot` acepta explícitamente un **experimento** con anotaciones públicas pendientes de revisión completa; no afirma que sean verificadas ni elimina la necesidad de negativos y evaluación local. El entrenador actual valida este piloto de clase única; para nuevos conjuntos multiclase se debe ampliar la auditoría antes de entrenarlos.

- Se conservan los pesos base en `.data/models/base/yolo11n.pt`, con URL oficial, licencia y SHA-256 en `yolo11n.source.json`. El asset antiguo no tiene digest público en GitHub: se registra el origen y un hash local, no una firma independiente.
- El entrenamiento tiene como máximo 30 épocas y puede detenerse antes tras 8 sin mejora. Se congelan las primeras 10 capas para reducir cómputo; no es una comparación exhaustiva de arquitecturas o parámetros.
- Se usan 4 hilos, lote 2, imágenes 416 px, `workers=0`, sin caché en RAM ni sincronización cloud. Los tiempos dependen del equipo; no se contrata GPU ni se llama a OpenAI.
- Se crea un experimento nuevo en `.data/runs/residuos_yolo11n_<fecha_hora>/`. Se rechaza sobrescribir uno existente.
- El lote original se mantiene intacto: el entrenamiento usa `dataset_snapshot/` dentro del experimento, donde Ultralytics puede escribir sus cachés.
- `train/weights/best.pt` se elige según validación, no según el test. Después se evalúa una vez sobre las 48 imágenes de prueba reservadas.
- `training_report.json` guarda parámetros, versiones, hashes, épocas efectivas y mAP50/mAP50–95. Los resúmenes de precisión/recall de Ultralytics no se presentan como un umbral fijo.
- Además se calcula precisión/recall/F1 a **confianza 0,35 e IoU de emparejamiento 0,50**, con correspondencia uno a uno, en `test_at_conf_035_iou_050`. `test_predictions.json` conserva resultados por imagen. Ese punto usa NMS IoU 0,70 y máximo 100 detecciones.
- El modelo queda **experimental y sin activación automática en la app**. No se sobrescribe `.data/models/terrenos/best.pt`; primero se revisan resultados, falsos positivos y licencia/uso previsto.

Experimento lanzado en esta sesión: `.data/runs/residuos_yolo11n_piloto_20261116_v1/`. El estado de ejecución y finalización lo indica su `training_report.json`. El manifiesto del dataset conserva el estado del lote al descargarlo y no se usa como indicador dinámico de entrenamientos posteriores.

YOLO11 y Ultralytics tienen condiciones **AGPL-3.0/Enterprise**. Revisar obligaciones antes de redistribuir o utilizar en un servicio comercial; instalar una dependencia no concede una licencia Enterprise.
