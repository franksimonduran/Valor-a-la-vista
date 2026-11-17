# Valor A la Vista · Mejoras de terrenos

Aplicación Python + Streamlit especializada **solo en terrenos**. Usa visión multimodal de **OpenAI (`gpt-6.1-terra`, Responses API)** para proponer problemas visibles, evidencia por foto/fotograma, recomendaciones y un score explicable de condición. Calcula el **costo aproximado de mejora por m² afectado**, no el valor de venta del terreno.

> Prototipo académico: no es tasación, diagnóstico técnico ni aprobación de crédito. La ausencia de un daño detectado no demuestra que no exista. No se ha validado la precisión con un conjunto representativo ni se incorpora un catálogo de precios oficiales comprobados.

## Inicio en Windows / PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe app.py
```

Configura en `config/config.py` o en el entorno (el entorno tiene prioridad):

```python
OPENAI_API_KEY = "TU_CLAVE"
OPENAI_MODEL = "gpt-6.1-terra"
API_BASE_URL = "https://api.openai.com/v1"
```

No compartas la clave. La página `platform.openai.com/api-keys` no es un endpoint de API. Solo se permite la API nativa HTTPS de OpenAI, sin redirecciones. No se necesita el SDK: se usa `requests`. Python recomendado: 3.12. Las versiones comprobadas están en `requirements-lock.txt`.

El modelo debe estar habilitado en tu cuenta y tener cuota de API; una suscripción a ChatGPT no incluye ese consumo. La integración usa `input_image`, `text.format` y salida JSON estricta, sin parámetros antiguos como `max_completion_tokens` ni temperatura forzada. No se hacen pruebas pagadas automáticamente.

## Uso

1. Selecciona **una foto o varias** desde tu dispositivo o cámara. Se agregan inmediatamente a la galería y, desde la primera, puedes pulsar **Analizar terreno**, sin completar vistas guiadas, área, ubicación ni precios. Agregar imágenes no llama a la API. Cada inspección corresponde a un mismo terreno. La galería vive en memoria durante la sesión, no en una base de datos.
2. Asigna vistas: **Panorámica, Acceso, Superficie, Límites**; añade tantos detalles como necesites hasta **16 fotos, 12 MB cada una y 64 MB totales**. Admite JPEG, PNG y WebP de hasta 25 megapíxeles. Puedes quitar cada foto. Las copias exactas se rechazan; las similitudes son solo avisos.
3. Declara ubicación y, opcionalmente, área total como contexto. **No se usa el área total como cantidad dañada**. Las fotos no acreditan propiedad, linderos, área real, suelo estable, servicios ni aptitud para construir.
4. Pulsa **Analizar terreno**: autoriza enviar las imágenes a OpenAI, con posibles cargos. Se eliminan EXIF/metadatos y se normalizan las imágenes antes de enviarlas. Solo se procesan fotos nuevas o fallidas. **Reanalizar todas las fotos** fuerza nuevas consultas. Subir o quitar fotos no dispara consultas.
5. Revisa problemas, foto, ubicación en el encuadre, evidencia, severidad aparente y acción orientativa. Se buscan maleza, residuos, agua acumulada, erosión aparente, acceso, cercos y desniveles. El catálogo guía, pero no garantiza detectar todos los defectos.
6. Opcionalmente **Busca precios en Internet**: utiliza Responses + búsqueda web de OpenAI. Envía ubicación y hallazgos, sin reenviar fotos. El modelo y la cuenta deben admitir búsqueda web; si no la admiten o no hay referencias, no se inventan precios. No hay consultas pagadas automáticas.
7. Las referencias candidatas necesitan URL oficial peruana citada por la búsqueda, documento/página o partida, fecha, ámbito y unidad. **Preparar referencias candidatas para revisión** rellena propuestas sin confirmarlas ni agregar cantidades. Revisa el documento, vigencia y aplicabilidad; una cita o URL oficial no certifica su contenido. Las referencias ya editadas no deben sustituirse.
8. Para cada mejora declara zona física, fases de trabajo, **cantidad afectada medida o declarada**, unidad, rango unitario, documento y alcance. Confirma el contraste y pulsa **Calcular presupuesto**.
9. Descarga el informe HTML (imprimible en PDF) o JSON. Incluye trazabilidad por foto, partidas, cantidades, fuentes y límites. Siempre es un **borrador automático sin revisión humana**; confirmar un precio no equivale a revisar el análisis visual.

## Videos y YOLO

- Selecciona **Añadir desde → Video**, carga MP4, MOV, AVI o WebM de hasta **60 s, 100 MB y 4K**, y se preparan automáticamente los fotogramas, sin requerir otras fotografías ni llamar todavía a la API. OpenCV debe poder decodificar el códec; MP4 H.264 es una opción recomendada.
- Se extraen entre **1 y 12 fotogramas** distribuidos en el video, limitados también por los espacios de la galería (máximo 16 fotos/fotogramas y 64 MB en total). No se analiza audio ni todos los fotogramas: pueden omitirse daños entre capturas. Los tiempos son aproximados, calculados por índice/fps.
- Con al menos un fotograma listo se habilita **Analizar terreno**; revisa o elimina capturas antes de pulsarlo. Solo esas imágenes se envían a OpenAI, no el video ni su audio. La extracción no genera cargos de API y elimina su archivo temporal local. La API puede generar hasta una consulta por imagen nueva. Las capturas repetidas exactas se descartan.
- El informe conserva nombre y SHA-256 del video, índice de fotograma y tiempo aproximado por hallazgo. Varios momentos del mismo daño no son más m²: agrúpalos antes de presupuestar.
- **Detección local con YOLO** es complementaria y opcional: solo se habilita con `requirements-yolo.txt` y pesos propios y confiables en `.data/models/terrenos/best.pt`. No se descargan pesos al iniciar, ni se presenta un YOLO genérico como detector entrenado de daños.
- Sus clases deben coincidir con las de terrenos. Las cajas son coordenadas normalizadas, no superficies reales. Sus confianzas no están calibradas como probabilidad de daño; **no modifican automáticamente score, recomendaciones ni presupuesto**. Una salida vacía no certifica terreno sano.

**Entrenamiento pendiente:** ya se descargó un piloto público de TACO para residuos con anotaciones de origen; no hay pesos de terrenos ni un dataset validado de todos los daños. Véase [Preparar el entrenamiento](docs/entrenamiento_terrenos.md). Entrenar YOLO es distinto de usar la API de OpenAI.

Se ha lanzado el experimento local `residuos_yolo11n_piloto_20261116_v1` (YOLO11 nano, CPU, hasta 30 épocas). Su estado y, al finalizar, sus métricas estarán en `.data/runs/residuos_yolo11n_piloto_20261116_v1/training_report.json`. No activa ni reemplaza automáticamente los pesos de la app. Las dependencias opcionales quedan fijadas en `requirements-yolo-lock.txt`.

### Dataset piloto público

`scripts/prepare_taco_dataset.py` descarga hasta 300 imágenes por defecto, convierte sus cajas COCO a YOLO (`0: residuos`) y guarda imágenes, etiquetas, licencia declarada, atribución y manifiesto en `.data/datasets/terrenos/taco_residuos_piloto_v1/`. Selecciona tierra/vegetación sin interiores según metadatos y excluye fuentes con otra licencia explícita. La herramienta no entrena ni utiliza la API.

Abre `revision.html` del lote para revisar todas las cajas. Es un **piloto pendiente de revisión local**, sin negativos verificados ni etiquetas de maleza/agua. Los lotes se separan entre train/val/test y se filtran similitudes, pero no se puede garantizar que sean terrenos físicamente distintos sin revisar las escenas. Conserva licencia y procedencia por imagen; los permisos individuales no se verifican automáticamente.

**Lote actual:** 300 imágenes, 781 cajas; 204 entrenamiento / 48 validación / 48 prueba. Auditoría técnica correcta. Unos 84 MiB entre piloto y fuentes conservadas; no se descargó el archivo completo de varios GB.

```powershell
.\.venv\Scripts\python.exe -B scripts\verify_dataset.py .data\datasets\terrenos\taco_residuos_piloto_v1
```

## Costo por m²

```text
Costo mínimo de la mejora = m² afectados × precio mínimo S/ por m²
Costo máximo de la mejora = m² afectados × precio máximo S/ por m²
```

Se muestra cada partida por separado. Cercos pueden usar metros lineales; otras partidas, unidades o servicios. **No se convierten artificialmente a m² ni se calcula un promedio entre unidades incompatibles**. Los cálculos usan `Decimal`, redondeo `ROUND_HALF_UP`, hasta 3 decimales para cantidades y 2 para precios.

Varias fotos no representan más superficie: agrupa los problemas de una misma reparación en una única partida y declara su cantidad conjunta. Se bloquean fases de trabajo repetidas en la misma zona declarada. Identificar incorrectamente zonas distintas puede impedir detectar solapamientos. Los posibles daños, severidad alta/no determinable, drenaje, erosión y desniveles requieren evaluación técnica previa declarada, no certificada por el sistema.

Se priorizan MVCS, MTC y entidades públicas competentes. **Índices INEI, valores arancelarios, valores de edificación, tasas TUPA y presupuestos globales no son precios de reparación**. Los contratos públicos no fijan tarifas nacionales. Si falta respaldo aplicable, queda **sin precio comprobado** y el total es parcial. No se agregan automáticamente IGV, transporte, permisos ni imprevistos: hay que declarar lo incluido y excluido.

El score es ilustrativo: bueno 100, regular 60, deteriorado 20; cada uno de los cuatro criterios pesa 25 %. Los no verificables se excluyen del promedio y reducen cobertura. Tener muchas fotos no sustituye las cuatro vistas guiadas, ni tener las vistas asignadas garantiza que los criterios sean visibles. No se activa un simulador de crédito ni una tasación del suelo.

## Comprobaciones sin API

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B scripts\smoke_test.py
.\.venv\Scripts\python.exe -B scripts\verify_env.py
.\.venv\Scripts\python.exe -m pip check
```

Las pruebas usan imágenes sintéticas y respuestas simuladas. `scripts/evaluate_model.py` es distinto: envía las fotos del manifiesto a OpenAI y genera cargos; ejecútalo solo con imágenes autorizadas. Sus métricas no validan uso financiero.

## Archivos principales

- `app.py`: interfaz solo de terrenos.
- `valor_a_la_vista/gallery.py`: galería ampliable y límites.
- `valor_a_la_vista/video.py`: extracción local de fotogramas y trazabilidad.
- `valor_a_la_vista/detector.py`: YOLO opcional con pesos locales de terrenos.
- `valor_a_la_vista/vision.py`: OpenAI, esquema JSON y preparación de imágenes.
- `valor_a_la_vista/inspection.py`: caché por foto/modelo/vista/versión, cobertura y evidencia.
- `valor_a_la_vista/online_prices.py`: búsqueda web opcional y filtro de candidatos citados.
- `valor_a_la_vista/pricing.py`: grupos, fuentes, cantidades y presupuesto.
- `valor_a_la_vista/report.py`: informes HTML/JSON escapados; SHA-256 no es firma digital.
- `data/precios_referenciales.csv`: vacío hasta incorporar precios respaldados. Los ejemplos multi-activo anteriores se conservan en `data/legado/`, fuera del flujo.
- `.data/backups/antes-terrenos-20261116-165630.zip`: respaldo previo a la migración, sin configuración privada.

Los módulos históricos de tarifas comerciales de paredes, presupuesto manual, simulación crediticia y segmentación no se usan en la interfaz. YOLO y segmentación son opcionales, no se descargan ni entrenan modelos al iniciar la app. Consulta `scripts/train_yolo.py --help` y los requisitos opcionales antes de entrenar; revisa licencias y utiliza datos autorizados. `.data/python/3.12` es el Python base local del entorno, no una caché para eliminar.


## Despliegue en Render (Python 3.12)

La rama principal `main` se despliega automáticamente en Render. Este prototipo utiliza Streamlit y no necesita Ollama ni un servidor de modelos local. La detección YOLO es opcional y no está preparada para el plan gratuito.

- **Build:** `python -m pip install --upgrade pip && python -m pip install -r requirements.txt`
- **Start:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --browser.gatherUsageStats false`
- **`PYTHON_VERSION`:** `3.12.10`
- **`OPENAI_API_KEY`:** clave privada (solo en Render → Environment; nunca en GitHub).
- **`OPENAI_MODEL`:** `gpt-6.1-terra`, sujeto a acceso y saldo de la cuenta.
- **Acceso al análisis:** sin contraseña en la interfaz. La API se invoca únicamente al pulsar **Analizar terreno**, **Reanalizar todas las fotos**, **Buscar precios en Internet** o **Actualizar referencias**. Una clave privada de OpenAI sigue siendo imprescindible y las llamadas pueden generar costos. En una web pública, establece cuotas y límites de gasto en la cuenta de OpenAI; para uso general se requiere una protección de acceso o limitación de consumo del lado del servidor.
- **`VAV_MAX_VIDEO_MB`:** `32` en Render Free para reducir el riesgo de agotar memoria; el máximo local predeterminado es 100 MB.

En producción, `opencv-python-headless` evita dependencias de ventanas gráficas que no están disponibles en servidores. Los cambios en `main` pasan por GitHub Actions: pruebas unitarias y arranque HTTP sin consumir la API. Una compilación exitosa **no** demuestra precisión visual, disponibilidad de la API ni validez pericial.

Render Free tiene disco temporal, una sola instancia y suspensión por inactividad. Las fotos viven en sesión y se pierden al reiniciar; descarga los informes antes de salir. Para uso público con usuarios reales deben añadirse autenticación robusta, consentimiento explícito, límites de consumo y almacenamiento seguro.


## Límites de demostración y diagnóstico de errores

- `VAV_DAILY_API_LIMIT` (valor inicial `32`): número máximo de consultas lógicas a OpenAI por día UTC **y proceso**.
- `VAV_SESSION_API_LIMIT` (valor inicial `16`): consultas por sesión y día; cada foto sin analizar, reanálisis forzado y búsqueda de referencias utiliza una unidad. Los aciertos de caché no consumen otra unidad.
- **Estos límites no garantizan un gasto máximo:** reiniciar el servidor los reinicia, otras instancias tienen contadores propios y los reintentos HTTP temporales pueden consumir más de una solicitud. Configura también topes en la cuenta de OpenAI. No hay contraseña pública, pero tampoco autenticación individual.
- En HTTP 429, el sistema distingue cuota/facturación cuando OpenAI envía `insufficient_quota`, y límites temporales cuando envía `rate_limit_exceeded`; solo estos últimos se reintentan brevemente. Un HTTP 429 sin código reconocible no permite deducir la causa exacta.
- La selección de un video extrae fotogramas localmente. Cambiar la cantidad sustituye las capturas de ese mismo video, sin borrar fotos independientes. Si falla, aparece **Reintentar extracción del video**.
- Con una sola foto válida o un fotograma válido se puede comenzar la inspección. Si no hay cobertura completa, el panel indica expresamente que la lectura es parcial.
- Ninguna prueba automática sustituye evaluar la calidad del análisis sobre fotografías reales y contrastarlas con un especialista.
