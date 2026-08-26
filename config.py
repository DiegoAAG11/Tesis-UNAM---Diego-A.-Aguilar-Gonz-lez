# -*- coding: utf-8 -*-
"""
config.py — Configuraciones centralizadas del sistema de análisis de haz de fibra óptica.

Todas las configuraciones se almacenan como dataclasses con valores por defecto.
El inicializador principal (main.py) instancia estos objetos y los pasa a cada módulo.
"""

from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# CALIBRACIÓN DE PÍXEL
# =============================================================================

@dataclass
class CalibrationConfig:
    """
    Parámetros de la calibración de tamaño de píxel (Opción 0, previa a
    casi todas las demás): se fotografía la punta de una fibra de diámetro
    físico conocido y se despeja µm/px a partir del radio detectado.

    `beam_diameter_um` es el ÚNICO patrón de longitud del proyecto — todas
    las magnitudes en µm de todos los análisis dependen en última instancia
    de que este valor corresponda a la fibra realmente fotografiada.
    Consumida por CalibracionLongitudPixel.py.
    """
    beam_diameter_um: float = 125.0       # Diámetro real de la fibra en µm
    exposure_us:      float = 10_000.0    # Exposición de cámara para calibración en µs
    gain_db:          float = 10.0        # Ganancia en dB
    pixel_format:     str   = "Mono8"     # Formato de pixel: Mono8, Mono10, Mono12
    save_figure:      Optional[str] = None
    show_figure:      bool = True


# =============================================================================
# EXPERIMENTO GENERAL
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Identificación y condiciones del experimento. Estos valores no alteran
    ningún cálculo salvo dos excepciones importantes:

    - `device_name` determina el nombre de la carpeta raíz
      (`YYYYMMDD_CaracterizacionHaz_<device_name>`) y el prefijo de todos
      los archivos generados (ver SessionConfig.filename_*).
    - `wavelength` SÍ entra en la física: se parsea a metros en
      CamaraTurbulencia.analizar_video_from_centroids para calcular
      r₀/Cₙ²/σ_R. Por eso su formato se valida estrictamente como "NNN nm"
      (gui/validacion.py::valid_wavelength).

    El resto (temperatura y corriente del láser, condiciones) es
    trazabilidad experimental: se vuelca a metadata_adquisicion.json y al
    reporte para poder reconstruir después en qué condiciones se tomó cada
    medición.
    """
    experiment_id:   str = 'EXPxxx'
    device_name:     str = 'MMI'          # Nombre del dispositivo óptico analizado (usado en nombres de archivos y en la carpeta raíz)
    conditions:      str = 'longitud de la fibra NFC: 11600 um'
    wavelength:      str = '980 nm'
    temperature:     str = '23.01'
    current:         str = '61.33'
    channel:         str = 'CH01'


# =============================================================================
# ACTIVACIÓN DE ETAPAS
# =============================================================================

@dataclass
class StagesConfig:
    """
    Interruptores de las 4 etapas de la adquisición del haz óptico
    (Opción 1). Permiten repetir solo una parte del experimento sin
    volver a capturar todo — p. ej. reactivar únicamente la Etapa 1 si el
    video de estabilidad temporal salió mal, conservando las
    polarizaciones ya adquiridas en otra corrida.

    SetupDialog exige al menos una etapa activa. Consumidas por
    Parte1.py::run_experiment vía el accesor `_stage()`.
    """
    stage0_dark_frame:             bool = True
    stage1_temporal_stability:     bool = True
    stage2_polarization_haz:       bool = True
    stage3_polarization_haz_fibra: bool = True


# =============================================================================
# DARK FRAME
# =============================================================================

@dataclass
class DarkFrameConfig:
    """
    Etapa 0: captura de la línea de base de ruido del sensor con el láser
    apagado. Ese dark frame se resta después, píxel a píxel, en el
    preprocesado (Parte2.py), eliminando la corriente de oscuridad y el
    offset del sensor para que la intensidad medida corresponda a señal
    óptica real y no a fondo electrónico.

    Los ángulos de paletas se fijan igual que en el resto de etapas por
    consistencia mecánica, aunque con el láser apagado la polarización no
    tiene efecto sobre la imagen capturada.
    """
    # Debe coincidir con utils_carpetas.NOMBRE_DARKFRAME -- la creacion real
    # de la carpeta usa esa constante, no este campo (no es configurable
    # por sesion en la practica).
    subfolder_name: str = 'darkframe'
    filename:       str = 'darkframe'     # Siempre "darkframe" para evitar confusiones, sin número ni extensión
    paddle1:        int = 80
    paddle2:        int = 80
    paddle3:        int = 80


# =============================================================================
# ESTABILIDAD TEMPORAL
# =============================================================================

@dataclass
class TemporalStabilityConfig:
    """
    Etapa 1: captura del haz a lo largo del tiempo con la polarización
    FIJA, para caracterizar su estabilidad temporal. Es la etapa que
    produce el dato central de la tesis — el video del que la Opción 3
    extrae beam wander, centelleo, ensanchamiento y correlación espacial.

    Se puede capturar video, fotos espaciadas (`num_images` /
    `time_interval`), o ambos. El video es lo que usa el análisis
    cuantitativo; las fotos son un respaldo con mayor profundidad de bits
    efectiva por imagen.

    `video_duration` fija cuántos segundos se graban, pero NO los fps: el
    fps real se mide empíricamente durante la captura
    (Parte1.py::record_video), porque la cámara no garantiza una tasa
    nominal constante y un eje de tiempo incorrecto sesgaría todas las
    métricas temporales derivadas.
    """
    paddle1:         int   = 80
    paddle2:         int   = 80
    paddle3:         int   = 80
    capture_photos:  bool  = True
    capture_video:   bool  = True
    num_images:      int   = 1
    time_interval:   float = 1.0
    log_all_metadata: bool = False
    video_duration:  int   = 30
    video_codec:     str   = 'mp4v'
    video_format:    str   = 'mp4'
    subfolder_name:  str   = 'estabilidad_temporal'


# =============================================================================
# POLARIZACIONES (Etapas 2 y 3)
# =============================================================================

@dataclass
class PolarizationConfig:
    """
    Etapas 2 y 3: barrido de hasta 10 estados de las paletas del
    controlador de polarización, capturando una imagen del haz en cada
    uno.

    Propósito científico: verificar que el perfil espacial del haz
    estructurado sea ROBUSTO frente a cambios del estado de polarización
    de entrada. Si el patrón se deformara al mover las paletas, la
    caracterización del haz dependería de una variable no controlada en
    los experimentos de turbulencia. Las métricas derivadas (matriz de
    correlación cruzada, mapa de varianza, dispersión del centroide entre
    estados) cuantifican precisamente esa robustez.

    Los 10 tríos de ángulos son valores FIJOS elegidos manualmente por el
    autor para muestrear el espacio de polarización de forma dispersa; no
    provienen de ninguna fórmula. Etapa 2 mide el haz solo; Etapa 3 repite
    el mismo barrido con la fibra insertada en el arreglo, para separar el
    efecto del dispositivo del efecto de la fibra de transporte.
    """
    subfolder_haz:       str = 'diferentes_polarizaciones_haz'
    subfolder_haz_fibra: str = 'diferentes_polarizaciones_haz_fibra'

    # 10 posiciones de polarización: lista con 'paddle1','paddle2','paddle3'
    positions: list = field(default_factory=lambda: [
        {'paddle1':   0, 'paddle2':   0, 'paddle3':   0},
        {'paddle1':  80, 'paddle2':  80, 'paddle3':  80},
        {'paddle1': 160, 'paddle2': 160, 'paddle3': 160},
        {'paddle1': 105, 'paddle2':  30, 'paddle3': 128},
        {'paddle1':  26, 'paddle2':  17, 'paddle3': 154},
        {'paddle1': 139, 'paddle2':  91, 'paddle3': 113},
        {'paddle1':  50, 'paddle2':  44, 'paddle3':  68},
        {'paddle1':  77, 'paddle2':  55, 'paddle3': 130},
        {'paddle1':  71, 'paddle2':   8, 'paddle3':  16},
        {'paddle1':  64, 'paddle2':  25, 'paddle3':  76},
    ])

    # P01 (índice 0) es siempre obligatoria; las demás se pueden deshabilitar.
    # Lista de 10 booleanos: True = habilitada, False = omitir en etapas 2 y 3.
    enabled_positions: list = field(default_factory=lambda: [True] * 10)

    @property
    def active_indices(self) -> list[int]:
        """Retorna los índices 0-based de las posiciones habilitadas."""
        return [i for i, en in enumerate(self.enabled_positions) if en]


# =============================================================================
# SWITCH ÓPTICO
# =============================================================================

@dataclass
class OpticalSwitchConfig:
    """
    Switch óptico 4x1 gobernado por una tarjeta NI-DAQ: selecciona cuál de
    los 4 puertos de entrada de fibra llega al arreglo de medición, sin
    reconectar fibras a mano entre dispositivos.

    Es el único hardware OPCIONAL del pipeline del haz óptico: si no está
    conectado, la adquisición continúa con una advertencia (montajes de un
    solo canal no lo necesitan). `channel` se sincroniza automáticamente
    con el canal elegido en la pestaña General de SetupDialog.
    """
    device_name: str  = 'Dev1'
    port:        int  = 0
    # Líneas digitales NI-DAQ que codifican el canal en binario. El orden
    # exacto de bit (MSB/LSB primero) tiene referencias contradictorias sin
    # verificar entre archivos del proyecto (Parte1.py dice "[MSB,LSB]",
    # gui/setup_dialog.py y docs/BeamCharacterization.md dicen "LSB/MSB")
    # -- pendiente de confirmar con hardware real, no se resuelve aquí.
    lines:       list = field(default_factory=lambda: [0, 1])
    channel:     int  = 1


# =============================================================================
# CÁMARA
# =============================================================================

@dataclass
class CameraConfig:
    """
    Parámetros del sensor de la cámara Allied Vision. Son los que más
    influyen en la validez del dato: exposición y ganancia deben elegirse
    para que el haz ocupe buena parte del rango dinámico SIN saturar —
    un pico saturado aplana el perfil y sesga D4σ, energía encerrada y
    centroide (Parte3.py::_verificar_saturacion avisa si ocurre).

    Regla experimental clave: el dark frame debe capturarse con la MISMA
    exposición y ganancia que las imágenes a las que se le restará; de lo
    contrario la resta no corresponde al mismo nivel de ruido de fondo.

    `pixel_format` fija la profundidad de bits (Mono8/10/12): más bits dan
    mejor resolución de intensidad a costa de fps. `guardar_npz` conserva
    además el dato crudo sin pérdida ni overlays, como respaldo
    reprocesable frente a la compresión del .mp4/.png.
    """
    exposure_time:       int            = 10_000
    gain:                float          = 10.0
    pixel_format:        str            = "Mono8"   # Mono8 | Mono10 | Mono12
    exposure_auto:       str            = 'Off'
    video_exposure_time: Optional[int]  = None   # None → usa exposure_time
    video_gain:          Optional[float] = None  # None → usa gain
    guardar_npz:         bool           = True   # respaldo crudo sin perdida (.npz), ademas de .png/.mp4


# =============================================================================
# POLARIZADOR
# =============================================================================

@dataclass
class PolarizerConfig:
    """
    Número de serie del controlador de polarización Thorlabs MPC320. El
    SDK Kinesis identifica el dispositivo por serial, no por puerto, así
    que este valor debe coincidir con el equipo físico del laboratorio.
    Cambiar de equipo exige actualizarlo aquí (o desde SetupDialog).
    """
    serial: str = "38388714"


# =============================================================================
# ANOTACIÓN EN IMÁGENES
# =============================================================================

@dataclass
class OverlayConfig:
    """
    Texto libre que Parte1.py quema sobre las imágenes/video de
    adquisición (esquina inferior derecha), para que la condición
    experimental quede legible en la propia imagen sin depender de los
    metadatos.

    ⚠ Estos píxeles de texto son datos falsos desde el punto de vista
    fotométrico: si el ROI del preprocesado los incluyera, contaminarían
    centroide, ancho y energía. Por eso Parte2.py verifica explícitamente
    la superposición ROI-vs-overlays antes de recortar. Dejar el texto
    vacío elimina el overlay y el riesgo.
    """
    text: str = 'Longitud de fibra NCF: 11600 um'


# =============================================================================
# BARRA DE ESCALA
# =============================================================================

@dataclass
class ScaleConfig:
    """
    Barra de escala física quemada sobre las imágenes de adquisición
    (esquina inferior izquierda), para que cualquier figura publicada
    lleve su propia referencia de tamaño real.

    `um_per_px`/`px_per_um` NO se editan a mano: los deriva
    `update_from_um_per_px()` a partir de la calibración vigente, invocado
    desde SessionConfig.apply_um_per_px(). Los valores por defecto son
    solo un marcador de posición hasta que se ejecuta la calibración.

    Aplica la misma advertencia que OverlayConfig: la barra es contenido
    sintético dentro de la imagen y no debe caer dentro del ROI analizado.
    """
    enabled:       bool  = True
    bar_length_um: int   = 100
    bar_height_px: int   = 16
    margin:        int   = 35
    font_scale:    float = 0.55
    font_thickness: int  = 1
    color:         tuple = (255, 255, 255)
    outline_color: tuple = (0, 0, 0)

    # Estos se calculan desde um_per_px (proporcionado por calibración o usuario)
    um_per_px:  float = 2.2
    px_per_um:  float = 0.4545

    def update_from_um_per_px(self, um_per_px: float):
        """Actualiza ambas escalas a partir de µm/px."""
        self.um_per_px = um_per_px
        self.px_per_um = 1.0 / um_per_px if um_per_px > 0 else 0.0



# =============================================================================
# CARACTERIZACIÓN DE CÁMARA DE TURBULENCIA
# =============================================================================

@dataclass
class TurbulenceConfig:
    """
    Parámetros de la caracterización de la cámara de turbulencia
    (Opciones 4-6). Definen la geometría óptica que entra directamente en
    el despeje de los parámetros de turbulencia atmosférica.

    Física involucrada: r₀, Cₙ² y σ_R se despejan de la varianza del beam
    wander mediante la relación de Kolmogorov
    ⟨r_c²⟩ = 0.54·L²·(λ/2W₀)²·(2W₀/r₀)^(5/3). Ahí entran `distance_m` (L,
    la longitud del trayecto turbulento) y W₀ (radio del haz colimado).

    W₀ admite dos vías, seleccionables con `metodo_w0`:
      - 'directo'  — se mide W₀ por segundo momento (D4σ/2) sobre el frame
                     de referencia sin turbulencia. No necesita más datos.
      - 'w_fibre'  — se calcula W₀ = f₁·λ/(π·W_fibre) por propagación
                     gaussiana, usando `focal_length_mm` y el radio de
                     campo modal `w_fibre_um` medido con la Opción 7.

    Un error en L o en W₀ se propaga con exponente no lineal a r₀ y Cₙ²,
    así que son los dos parámetros más críticos de esta configuración.
    La polarización se mantiene FIJA durante todo el barrido para que la
    única variable entre casos sea la turbulencia (ΔT y velocidad de aire).
    """
    # ── Parámetros ópticos del haz colimado ───────────────────────────────────
    focal_length_mm:   float = 11.0       # f1: focal de la lente colimadora [mm]
    distance_m:        float = 1.0        # L: distancia de propagacion [m]


    # ── Polarizacion (fija durante todo el experimento) ───────────────────────
    paddle1:           int   = 80
    paddle2:           int   = 80
    paddle3:           int   = 80

    # ── Calculo de W0 ─────────────────────────────────────────────────────────
    metodo_w0:         str   = 'directo'  # 'directo' (D4sigma) o 'w_fibre' (desde focal)
    w_fibre_um:        float = 0.0        # W_fibre [µm] — requerido si metodo_w0='w_fibre'

    # ── Adquisicion ───────────────────────────────────────────────────────────
    video_duration_s:  int   = 30
    video_codec:       str   = 'mp4v'
    video_format:      str   = 'mp4'

# =============================================================================
# ANÁLISIS (Parte 3)
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Configuración del análisis del haz óptico (Opción 3, `Parte3.py`).
    Agrupa tres tipos de ajuste bien distintos:

    1. **Parámetros que SÍ afectan los resultados numéricos**:
       `umbral_intensidad` (define el borde del haz en el método de
       umbral), `ancho_perfil_radial` (nº de líneas promediadas por
       perfil), `fracciones_energia` (qué radios de energía encerrada se
       reportan), `intervalo_frames`/`analizar_todos` (submuestreo
       temporal — analizar menos frames acelera la corrida pero reduce la
       estadística del beam wander). Cambiarlos cambia los números
       reportados en la tesis.

    2. **Parámetros de muestreo angular**: `analizar_todos_angulos` /
       `intervalo_angulos` controlan la resolución del análisis radial de
       360°.

    3. **Interruptores de salida** (`generar_*`, `video_*`): activan o
       desactivan cada figura/video/tabla individualmente. NO alteran los
       cálculos, solo qué archivos se escriben — existen para poder
       repetir rápidamente una corrida cuando solo se necesita reajustar
       una figura concreta. Se aplican mediante
       `Parte3.py::_patch_analysis_functions`.
    """
    # Frames
    analizar_todos:    bool  = False
    intervalo_frames:  int   = 10

    # Ángulos para el análisis radial
    analizar_todos_angulos: bool = True   # True = 0°–359° completo
    intervalo_angulos:      int  = 1      # paso en grados (1 = todos)
    ancho_perfil_radial:    int  = 1      # píxeles de ancho a promediar (impar recomendado)
    radial_mostrar_e2:      bool = True   # mostrar línea indicadora 1/e² en video radial
    radial_mostrar_maximos: bool = True   # mostrar conteo de máximos locales en video radial
    fps_video:         Optional[float] = None
    umbral_intensidad: float = 0.01

    # Vídeo de salida
    generar_videos:  bool  = True
    velocidad_video: float = 1.0

    # Análisis de polarizaciones
    analizar_polarizaciones: bool = True
    desplazamientos_fso:     list = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15, 20])
    fracciones_energia:      list = field(default_factory=lambda: [0.50, 0.86, 0.99])

    # Resultados a generar (activar/desactivar individualmente)
    generar_desplazamiento_centroide: bool = True
    generar_potencia_normalizada:     bool = True
    generar_correlacion_espacial:     bool = True
    generar_ancho_haz:                bool = True
    generar_imagen_referencia:        bool = True
    generar_imagen_normalizada:       bool = True
    # Vídeos individuales
    video_desplazamiento_centroide:   bool = True
    video_potencia_normalizada:       bool = True
    video_correlacion_espacial:       bool = True
    video_ancho_haz_temporal:         bool = True
    video_ancho_haz_frames:           bool = True
    generar_video_resumen:            bool = True   # 07_video_resumen_analisis.mp4 (5 paneles)
    # Análisis radial (polarizaciones)
    video_perfil_radial:              bool = True
    grafica_angulo_intensidad:        bool = True   # 12_energia_angular_integrada.png (estática) -- nombre del campo sin cambiar por compatibilidad con presets guardados

    # Resultados de polarizaciones individuales (pol_01 … pol_08)
    generar_pol_tabla:       bool = True   # pol_01_tabla_metricas.png
    generar_pol_promedio:    bool = True   # pol_02a/b imagen promedio + std
    generar_pol_varianza:    bool = True   # pol_03_mapa_varianza.png
    generar_pol_correlacion: bool = True   # pol_04_matriz_correlacion.png
    generar_pol_centroide:   bool = True   # pol_05_desplazamiento_centroide.png
    generar_pol_energia:     bool = True   # pol_06_curvas_energia_encerrada.png
    generar_pol_img_fso:     bool = True   # 07_imagen_promedio_fso.png
    generar_pol_tabla_fso:   bool = True   # 08_tabla_metricas_fso.png
    generar_pol_sensibilidad: bool = True  # 09_sensibilidad_desalineacion.png
    generar_pol_txt:         bool = True   # 10_resultados_numericos.txt


# =============================================================================
# SESIÓN COMPLETA
# =============================================================================

@dataclass
class SessionConfig:
    """
    Objeto raíz que viaja entre módulos.
    El inicializador (main.py) lo construye tras la GUI y lo pasa a cada etapa.
    """
    um_per_px:          float = 2.2       # Definido por calibración o entrada manual

    calibration:  CalibrationConfig       = field(default_factory=CalibrationConfig)
    experiment:   ExperimentConfig        = field(default_factory=ExperimentConfig)
    stages:       StagesConfig            = field(default_factory=StagesConfig)
    dark_frame:   DarkFrameConfig         = field(default_factory=DarkFrameConfig)
    temporal:     TemporalStabilityConfig = field(default_factory=TemporalStabilityConfig)
    polarization: PolarizationConfig      = field(default_factory=PolarizationConfig)
    switch:       OpticalSwitchConfig     = field(default_factory=OpticalSwitchConfig)
    camera:       CameraConfig            = field(default_factory=CameraConfig)
    polarizer:    PolarizerConfig         = field(default_factory=PolarizerConfig)
    overlay:      OverlayConfig           = field(default_factory=OverlayConfig)
    scale:        ScaleConfig             = field(default_factory=ScaleConfig)
    analysis:     AnalysisConfig          = field(default_factory=AnalysisConfig)
    turbulence:   TurbulenceConfig        = field(default_factory=TurbulenceConfig)

    # Carpeta RAÍZ DEL DISPOSITIVO creada/reutilizada por Parte 1 (se propaga
    # a Parte 2 y 3) — contiene los casos Sin turbulencia/Transitorio/Con
    # turbulencia, cada uno con su propio Adquisicion/Preprocesado/Analisis/
    # darkframe (ver utils_carpetas.py).
    acquisition_folder: Optional[str] = None
    # Ídem, tras el preprocesado (Parte 2 también retorna la raíz del
    # dispositivo, no la carpeta de un caso individual)
    cropped_folder: Optional[str] = None
    # Último caso adquirido en Parte 1 dentro de esta sesión. Solo se usa
    # como preselección de conveniencia en el checkbox de selección de
    # casos de Parte 2/Parte 3 — nunca para saltarse ese diálogo, que
    # siempre se muestra.
    caso_actual: Optional[str] = None

    def apply_um_per_px(self, um_per_px: float):
        """
        Propaga el tamaño de píxel a todos los módulos que lo necesitan
        (barra de escala en adquisición y análisis).
        """
        self.um_per_px = um_per_px
        self.scale.update_from_um_per_px(um_per_px)

    # ── Nombres de archivo según convención nueva ──────────────────────────────

    def filename_haz(self, canal: str, polarization_index: int) -> str:
        """Ejemplo: MMI_Haz_CH01_P01"""
        return f"{self.experiment.device_name}_Haz_{canal}_P{polarization_index:02d}"

    def filename_haz_fibra(self, canal: str, polarization_index: int) -> str:
        """Ejemplo: MMI_Haz_Fibra_CH01_P01"""
        return f"{self.experiment.device_name}_Haz_Fibra_{canal}_P{polarization_index:02d}"

    def filename_estabilidad(self, canal: str, image_index: int) -> str:
        """Ejemplo: MMI_Estabilidad_CH01_T01"""
        return f"{self.experiment.device_name}_Estabilidad_{canal}_T{image_index:02d}"

    def filename_video(self, canal: str) -> str:
        """Ejemplo: MMI_Estabilidad_CH01_Video"""
        return f"{self.experiment.device_name}_Estabilidad_{canal}_Video"
