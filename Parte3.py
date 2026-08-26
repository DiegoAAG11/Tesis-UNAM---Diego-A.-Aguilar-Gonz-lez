# -*- coding: utf-8 -*-
"""
Analizador temporal y de polarizaciones
para un Haz de Fibra Óptica

Descripción:
Analiza los datos ya adquiridos (Parte1.py) y opcionalmente preprocesados
(Parte2.py) de la caracterización del haz óptico: el video de estabilidad
temporal (desplazamiento del centroide, ancho del haz por umbral y D4σ,
potencia normalizada, correlación espacial, perfil radial) y, si están
disponibles, las series de imágenes en diferentes polarizaciones (Etapas 2
y 3) para evaluar sensibilidad a la polarización y acoplamiento a fibra.

Metodología:
1) Pide la carpeta RAÍZ del dispositivo y con cuáles de los casos
   existentes (Sin turbulencia/Transitorio/Con turbulencia) trabajar
   (selección múltiple), y con qué origen de datos (Adquisicion o
   Preprocesado).
2) Para cada caso: carga el video de estabilidad temporal y calcula
   centroide, ancho del haz, potencia normalizada y correlación espacial
   frame a frame; genera las figuras y videos de cada serie temporal.
3) Si hay imágenes de polarización disponibles: calcula métricas por
   imagen (D4σ, energía encerrada, elipticidad), matriz de correlación
   cruzada entre polarizaciones, curvas de energía encerrada y la curva
   de sensibilidad a desalineación.
4) Exporta todos los resultados numéricos en CSV/TXT (Datos_Crudos/),
   además de las figuras y videos de publicación.

Autor: Diego Aguilar
"""

import os
import re
import sys
import glob
import traceback
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import utils_carpetas
import utils_imagenes

_ROOT_DIR_P3 = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR_P3 not in sys.path:
    sys.path.insert(0, _ROOT_DIR_P3)
from gui.eleccion_carpeta_dialog import pedir_eleccion_carpeta
from gui.seleccion_casos_dialog import pedir_casos_multiples
from console_ui import print_banner, print_seccion, print_warn, print_error, print_ok
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    # Formato de publicacion: tesis, papers, congresos
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
    "savefig.dpi":        300,
    # Fuente serif (Times New Roman compatible) para notacion cientifica
    "font.family":        "serif",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.titlesize":   15,
    # Lineas y marcadores
    "lines.linewidth":    1.8,
    "lines.markersize":   7,
    # Cuadricula discreta
    "axes.grid":          True,
    "grid.color":         "#CCCCCC",
    "grid.linestyle":     "--",
    "grid.alpha":         0.5,
    # Sin bordes superior/derecho (estilo publicacion)
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#333333",
    "text.color":         "#111111",
    "axes.labelcolor":    "#111111",
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIGURACIÓN PRINCIPAL  
# =============================================================================

ANALIZAR_TODOS    = True
INTERVALO_FRAMES  = 10
FPS_VIDEO         = None
UMBRAL_INTENSIDAD = 0.01

GENERAR_VIDEOS  = True
VELOCIDAD_VIDEO = 1.0

# ── Análisis de polarizaciones ────────────────────────────────────────────────────
ANALIZAR_POLARIZACIONES = True
FRACCIONES_ENERGIA      = [0.50, 0.86, 0.99]
ANALIZAR_TODOS_ANGULOS  = True
INTERVALO_ANGULOS       = 1
ANCHO_PERFIL_RADIAL     = 1      # lineas paralelas a promediar (1 = linea simple)
RADIAL_MOSTRAR_E2       = True   # mostrar linea indicadora 1/e2 en el video radial
RADIAL_MOSTRAR_MAXIMOS  = True   # mostrar conteo de maximos locales en el video radial

# ── Escala espacial ───────────────────────────────────────────────────────────
TAMANO_PIXEL_UM = 0.22836

# ── Marca de dispositivo/caso para pie de pagina de figuras/videos ───────────
# Se fija una vez por caso analizado en _ejecutar_analisis_caso(); None si no
# se pudo determinar (p. ej. ejecucion standalone sobre una carpeta con un
# nombre atipico) -- en ese caso simplemente no se agrega pie de pagina.
_MARCA_PIE = None

# ── Nombres de subcarpetas ────────────────────────────────────────────────────
NOMBRE_SUBCARPETA_TEMPORAL      = "Estabilidad_Temporal"
NOMBRE_SUBCARPETA_POLARIZACION  = "Polarizaciones"
NOMBRE_SUBCARPETA_DATOS_CRUDOS  = "Datos_Crudos"


# =============================================================================
# SELECCIÓN INTERACTIVA DE CARPETAS
# =============================================================================

def _solicitar_carpeta_raiz_interactivo() -> str | None:
    """
    Pide únicamente la carpeta RAÍZ DEL DISPOSITIVO (contiene los casos
    Sin turbulencia/Transitorio/Con turbulencia) — nunca hay que navegar
    manualmente hasta una subcarpeta. Retorna la ruta o None si se cancela.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        "Carpeta raíz del dispositivo",
        "Selecciona la carpeta RAÍZ del dispositivo\n"
        "(contiene los casos: Sin turbulencia, Transitorio, Con turbulencia).",
        parent=root
    )
    carpeta_raiz = filedialog.askdirectory(
        title="Selecciona la carpeta raíz del dispositivo",
        initialdir=os.path.expanduser("~"),
        parent=root
    )
    root.destroy()
    if not carpeta_raiz:
        return None
    return utils_carpetas.normalizar_carpeta_raiz_dispositivo(carpeta_raiz)


# =============================================================================
# UTILIDADES DE CONVERSIÓN ESPACIAL
# =============================================================================
# Todo el análisis interno trabaja en píxeles; estas 4 utilidades convierten
# a µm SOLO al momento de graficar/exportar, usando la calibración global
# TAMANO_PIXEL_UM (fijada por la sesión o por metadata_adquisicion.json).
# Si no hay calibración (TAMANO_PIXEL_UM=None), todo se reporta en px --
# el análisis sigue siendo válido en unidades relativas, solo pierde la
# escala física absoluta.

def _to_um(val):
    """Convierte px → µm (escalar o array); identidad si no hay calibración."""
    if TAMANO_PIXEL_UM is None:
        return val
    if isinstance(val, np.ndarray):
        return val * TAMANO_PIXEL_UM
    return val * TAMANO_PIXEL_UM

def _unidad() -> str:
    """Etiqueta de unidad para ejes/leyendas: "µm" con calibración, "px" sin ella."""
    return "µm" if TAMANO_PIXEL_UM is not None else "px"

def _factor() -> float:
    """Factor multiplicativo px → unidad de reporte (µm/px, o 1.0 sin calibración)."""
    return float(TAMANO_PIXEL_UM) if TAMANO_PIXEL_UM is not None else 1.0

def _extent(H: int, W: int) -> list:
    """`extent` para imshow en unidades físicas, con eje Y invertido
    (origen arriba-izquierda, convención de imagen)."""
    f = _factor()
    return [0, W * f, H * f, 0]


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def _cargar_metadata_adquisicion(carpeta: str) -> dict:
    """
    Intenta cargar metadata_adquisicion.json de la carpeta de entrada.
    Retorna el dict o {} si no existe.
    """
    ruta = os.path.join(carpeta, "metadata_adquisicion.json")
    if os.path.exists(ruta):
        try:
            import json as _json
            with open(ruta, "r", encoding="utf-8") as f:
                data = _json.load(f)
            print("  Metadatos cargados: metadata_adquisicion.json")
            return data
        except Exception as e:
            print(f"  Advertencia: no se pudo leer metadata_adquisicion.json: {e}")
    return {}


def fig_a_frame_bgr(fig: plt.Figure) -> np.ndarray:
    """
    Rasteriza una figura de matplotlib a un array BGR de OpenCV, para poder
    escribirla como frame de video con cv2.VideoWriter. Es el puente entre
    los dos mundos gráficos del proyecto: todas las funciones `video_*`
    dibujan cada frame con matplotlib y lo convierten aquí antes de
    escribirlo al .mp4.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def _agregar_marca_pie(fig: plt.Figure) -> None:
    """
    Agrega "<dispositivo> | <caso>" como pie de página discreto, fuera del
    área de datos, en la esquina inferior derecha de la figura -- gris,
    cursiva, pequeño (no estorba, no es llamativo, y una franja horizontal
    en el borde se puede recortar/tapar fácilmente en edición de imágenes
    sin afectar el contenido real de la gráfica).

    No hace nada si `_MARCA_PIE` no está definido (p. ej. Parte3.py
    ejecutado en modo standalone sobre una carpeta con nombre atípico, o
    llamado antes de resolver el caso). Debe llamarse justo antes de
    guardar la figura (savefig) o de capturar el frame para video.
    """
    if _MARCA_PIE:
        fig.text(0.995, 0.005, _MARCA_PIE, fontsize=7, style="italic",
                 color="#999999", ha="right", va="bottom")


# =============================================================================
# BÚSQUEDA Y CARGA
# =============================================================================

def encontrar_carpeta_estabilidad(carpeta_entrada: str) -> str | None:
    """Localiza la subcarpeta 'estabilidad_temporal' (búsqueda por nombre,
    insensible a mayúsculas) dentro de Adquisicion/ o Preprocesado/.
    Retorna None si no existe -- el llamador decide si eso es error."""
    for item in os.listdir(carpeta_entrada):
        ruta_item = os.path.join(carpeta_entrada, item)
        if os.path.isdir(ruta_item) and "estabilidad_temporal" in item.lower():
            return ruta_item
    return None

def encontrar_video(carpeta_estabilidad: str) -> str | None:
    """Retorna el PRIMER archivo de video encontrado en la carpeta (por
    extensión: mp4/avi/mov/mkv). La adquisición solo genera un video por
    caso, así que en el flujo normal no hay ambigüedad. Nunca selecciona
    el .npz de respaldo crudo (no está entre las extensiones buscadas)."""
    extensiones = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MOV", "*.AVI", "*.MP4"]
    for ext in extensiones:
        archivos = glob.glob(os.path.join(carpeta_estabilidad, ext))
        if archivos:
            return archivos[0]
    return None

def cargar_frames(ruta_video: str) -> tuple[list, float, list]:
    """
    Carga frames de un video .mp4 para analisis.

    Por decision de diseno, el preprocesado y el analisis del haz optico
    trabajan EXCLUSIVAMENTE con el .mp4 de la adquisicion (no con el .npz
    de datos crudos), para acelerar el computo. Los datos crudos siguen
    guardandose durante la adquisicion por si se necesitan mas adelante,
    pero no se usan en esta etapa.
    """
    if ANALIZAR_TODOS:
        intervalo_ef = 1
        modo_txt = "TODOS los frames"
    else:
        intervalo_ef = max(1, INTERVALO_FRAMES)
        modo_txt = f"cada {intervalo_ef} frames"

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {ruta_video}")

    fps          = cap.get(cv2.CAP_PROP_FPS) if FPS_VIDEO is None else FPS_VIDEO
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video        : {os.path.basename(ruta_video)}")
    print(f"  Total frames : {total_frames}")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Modo         : {modo_txt}")

    indices = list(range(0, total_frames, intervalo_ef))
    frames  = []
    indices_leidos = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # No convertir a float64 aqui: con video sin recortar (resolucion
            # completa del sensor) esto puede disparar el uso de RAM a decenas
            # de GB. Las funciones que necesitan float64 (normalizar_frame,
            # etc.) ya hacen su propio .astype(np.float64) frame por frame.
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gris)
            indices_leidos.append(idx)
    cap.release()

    print(f"  Frames cargados: {len(frames)}")
    return frames, fps, indices_leidos


def _promedio_primer_segundo_video(ruta_video: str) -> np.ndarray | None:
    """
    Carga y promedia solo el primer segundo (aprox., según fps real del
    archivo) de un video, en escala de grises, sin aplicar el muestreo de
    ANALIZAR_TODOS/INTERVALO_FRAMES -- siempre lee frames consecutivos
    reales desde el inicio, porque el objetivo es un promedio de corto
    plazo representativo del estado inicial del haz, no una serie temporal
    completa. Usado como referencia compartida de correlación espacial
    (ver _ejecutar_analisis_caso).

    Retorna None si el video no se pudo abrir o no tiene frames legibles.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) if FPS_VIDEO is None else FPS_VIDEO
    if not fps or fps <= 0:
        fps = 30.0
    n_objetivo = max(1, int(round(fps * 1.0)))
    acum = None
    leidos = 0
    for _ in range(n_objetivo):
        ret, frame = cap.read()
        if not ret:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        acum = gris if acum is None else acum + gris
        leidos += 1
    cap.release()
    if leidos == 0:
        return None
    return acum / leidos


def _buscar_video_estabilidad(carpeta_caso: str, eleccion: str) -> str | None:
    """Ubica el video de estabilidad_temporal de `<carpeta_caso>/<eleccion>`,
    o None si la subcarpeta, la carpeta 'estabilidad_temporal' o el video
    no existen."""
    carpeta_entrada = os.path.join(carpeta_caso, eleccion)
    if not os.path.isdir(carpeta_entrada):
        return None
    carpeta_estabilidad = encontrar_carpeta_estabilidad(carpeta_entrada)
    if carpeta_estabilidad is None:
        return None
    return encontrar_video(carpeta_estabilidad)


def _cargar_referencia_correlacion(carpeta_raiz: str, eleccion: str,
                                    caso_actual: str, ruta_video_actual: str,
                                    forma_frame_actual: tuple) -> tuple[np.ndarray | None, str | None]:
    """
    Retorna (referencia_2D, motivo_de_fallo) para la correlación espacial:
    el promedio del primer segundo del caso SinTurbulencia del mismo
    dispositivo (mismo `eleccion`), que se usa como referencia única en
    los 3 casos -- así "con turbulencia"/"transitorio" miden degradación
    respecto al estado normal del haz, no autoconsistencia interna de su
    propio video (que en "con turbulencia" ya está perturbado desde el
    frame 0).

    Si `caso_actual` YA ES 'SinTurbulencia', reutiliza su propio video
    (`ruta_video_actual`) en vez de "buscarse a sí mismo" de nuevo.

    Si el caso SinTurbulencia no existe o no tiene video, retorna
    (None, motivo) -- el llamador debe usar el frame 0 propio como
    respaldo y avisar explícitamente al usuario (menos comparable entre
    casos, pero mejor que fallar el análisis completo). Si la resolución
    difiere (p. ej. el ROI se recortó distinto entre casos en la Opción 2),
    la referencia se redimensiona para que la correlación se pueda
    calcular igual, en vez de fallar (misma estrategia que
    ComparacionResiliencia.py::_cargar_condicion).
    """
    if caso_actual == utils_carpetas.CASOS[0]:
        ref = _promedio_primer_segundo_video(ruta_video_actual)
        if ref is None:
            return None, "no se pudo leer el propio video de SinTurbulencia"
        return ref, None

    carpeta_raiz_dispositivo = carpeta_raiz
    carpeta_sin_turb = os.path.join(carpeta_raiz_dispositivo, utils_carpetas.CASOS[0])
    if not os.path.isdir(carpeta_sin_turb):
        return None, "el caso 'SinTurbulencia' no existe para este dispositivo"
    ruta_video_st = _buscar_video_estabilidad(carpeta_sin_turb, eleccion)
    if ruta_video_st is None:
        return None, (f"el caso 'SinTurbulencia' no tiene video de estabilidad "
                       f"temporal con origen '{eleccion}'")
    ref = _promedio_primer_segundo_video(ruta_video_st)
    if ref is None:
        return None, "no se pudo leer el video de 'SinTurbulencia'"
    if ref.shape != forma_frame_actual:
        print_warn(
            f"El ROI del caso SinTurbulencia ({ref.shape[1]}x{ref.shape[0]} px) "
            f"difiere del de este caso ({forma_frame_actual[1]}x{forma_frame_actual[0]} px). "
            "Redimensionando la referencia para poder calcular la correlación.")
        ref = cv2.resize(ref.astype(np.float32),
                         (forma_frame_actual[1], forma_frame_actual[0]),
                         interpolation=cv2.INTER_LINEAR).astype(np.float64)
    return ref, None


# =============================================================================
# CÁLCULOS
# =============================================================================

def _verificar_saturacion(frame: np.ndarray, umbral_frac_pixeles: float = 0.001) -> None:
    """
    Avisa si una fracción significativa de píxeles está en el valor máximo
    representable para el tipo de dato del frame (ej. 255 para Mono8) --
    indicio de saturación de cámara, que aplana el pico de intensidad y
    puede sesgar D4σ, energía encerrada y centroide. Solo aplica a frames
    con dtype entero (la verificación no tiene sentido sobre datos ya
    convertidos a float, p. ej. promedios de varias imágenes).
    """
    if not np.issubdtype(frame.dtype, np.integer):
        return
    valor_max = np.iinfo(frame.dtype).max
    n_saturados = int(np.count_nonzero(frame >= valor_max))
    if n_saturados == 0:
        return
    frac = n_saturados / frame.size
    if frac >= umbral_frac_pixeles:
        print_warn(f"Posible saturación de cámara: {n_saturados} píxeles "
                   f"({frac*100:.2f}%) en el valor máximo representable "
                   f"({valor_max}) -- puede sesgar D4σ, energía encerrada y "
                   "centroide. Revisar exposición/ganancia de la adquisición.")


def calcular_centroide(frame: np.ndarray) -> tuple[float, float]:
    """
    Centroide de intensidad (cx, cy) en píxeles, por momentos de imagen
    (ver MathematicalReference.md §1). No lanza excepción: si el frame no
    tiene señal medible (m00=0), retorna el centro geométrico de la
    imagen como valor de respaldo, con `print_warn` explícito -- ese
    valor no es físicamente significativo y el llamador debería
    considerar excluir el frame.
    """
    _verificar_saturacion(frame)
    M = cv2.moments(frame)
    if M["m00"] == 0:
        print_warn("Frame sin señal medible (intensidad total = 0) — centroide "
                   "devuelto como centro geométrico de la imagen (valor no físico, "
                   "revisar si este frame debe excluirse del análisis).")
        return frame.shape[1] / 2, frame.shape[0] / 2
    return M["m10"] / M["m00"], M["m01"] / M["m00"]

def normalizar_frame(frame: np.ndarray) -> np.ndarray:
    """
    Reescala un frame al rango [0, 1] restando su mínimo y dividiendo por
    su máximo (normalización min-max en float64).

    Uso EXCLUSIVAMENTE de visualización y de detección de borde relativa
    (`calcular_ancho_haz` compara el perfil contra una fracción del pico).
    NO debe usarse antes de métricas fotométricas absolutas —centroide,
    segundo momento, potencia total—: al ser una normalización por frame,
    destruye la información de intensidad relativa ENTRE frames, que es
    justamente lo que mide el índice de centelleo.
    """
    img = frame.astype(np.float64)
    img -= img.min()
    mx = img.max()
    if mx > 0:
        img /= mx
    return img

def _caminar_desde_borde(perfil_norm: np.ndarray, borde: int,
                          direccion: int, umbral: float) -> float:
    """
    Recorre `perfil_norm` desde `borde` en la dirección indicada (+1/-1)
    hasta encontrar el primer cruce del `umbral` de intensidad, e
    interpola linealmente entre esa muestra y la anterior para ubicar el
    cruce con precisión sub-píxel (en vez de solo el índice entero más
    cercano). Retorna la posición del cruce en unidades de índice de
    `perfil_norm` (píxeles).
    """
    n = len(perfil_norm)
    i = borde
    while 0 <= i <= n - 1:
        sig = i + direccion
        if not (0 <= sig <= n - 1):
            return float(i)
        if perfil_norm[sig] > umbral:
            p0, p1 = perfil_norm[i], perfil_norm[sig]
            dp = p1 - p0
            if dp != 0:
                return float(i + direccion * (umbral - p0) / dp)
            return float(sig)
        i += direccion
    return float(max(0, min(i, n - 1)))

ANCHO_PROMEDIO_UMBRAL = 5  # filas/columnas a combinar (mediana) en torno al
                           # centroide (impar recomendado) al calcular el
                           # ancho por el metodo de umbral -- una sola fila/
                           # columna de 1px es muy sensible a un solo pixel
                           # de ruido cerca del borde. Se usa la MEDIANA (no
                           # el promedio) porque rechaza por completo un
                           # pixel/fila atipico aislado (mientras sea
                           # minoria dentro de la ventana), sin diluir ni
                           # deformar la forma real del perfil -- importante
                           # para perfiles estructurados/no gaussianos
                           # (ver calcular_ancho_haz, metodo propio del autor).

def calcular_ancho_haz(frame: np.ndarray,
                       umbral: float = None,
                       ancho_promedio: int = None
                       ) -> tuple[float, float, dict]:
    """
    Ancho del haz por el método de umbral de intensidad (distinto de D4σ):
    a partir del perfil horizontal/vertical que pasa por el centroide
    (mediana de `ancho_promedio` filas/columnas vecinas, ver
    ANCHO_PROMEDIO_UMBRAL), camina desde cada borde de la imagen hacia el
    centro con `_caminar_desde_borde` hasta cruzar `umbral` (fracción de
    la intensidad pico normalizada). El ancho es la distancia entre los
    dos cruces (izquierdo/derecho, arriba/abajo).

    Retorna (ancho_x, ancho_y, info) — `info` trae los perfiles y bordes
    detectados, usados por las funciones de dibujo/video (ver
    _dibujar_ancho_en_axes).
    """
    if umbral is None:
        umbral = UMBRAL_INTENSIDAD
    if ancho_promedio is None:
        ancho_promedio = ANCHO_PROMEDIO_UMBRAL
    cx, cy = calcular_centroide(frame)
    icy = max(0, min(int(round(cy)), frame.shape[0] - 1))
    icx = max(0, min(int(round(cx)), frame.shape[1] - 1))

    img_norm = normalizar_frame(frame)
    if img_norm.max() <= 0:
        print_warn("Frame sin señal medible (intensidad total = 0) — el ancho "
                   "calculado por el método de umbral corresponderá al tamaño "
                   "completo de la imagen (valor no físico, revisar si este "
                   "frame debe excluirse del análisis).")
    medio = max(0, ancho_promedio // 2)
    fila_ini = max(0, icy - medio);  fila_fin = min(img_norm.shape[0], icy + medio + 1)
    col_ini  = max(0, icx - medio);  col_fin  = min(img_norm.shape[1], icx + medio + 1)
    perfil_h = np.median(img_norm[fila_ini:fila_fin, :], axis=0)
    perfil_v = np.median(img_norm[:, col_ini:col_fin], axis=1)

    n_h = len(perfil_h)
    n_v = len(perfil_v)

    x_izq  = _caminar_desde_borde(perfil_h, 0,       +1, umbral)
    x_der  = _caminar_desde_borde(perfil_h, n_h - 1, -1, umbral)
    y_arr  = _caminar_desde_borde(perfil_v, 0,       +1, umbral)
    y_abaj = _caminar_desde_borde(perfil_v, n_v - 1, -1, umbral)

    return (x_der - x_izq, y_abaj - y_arr, {
        "cx": cx, "cy": cy, "icy": icy, "icx": icx,
        "perfil_h": perfil_h, "perfil_v": perfil_v,
        "x_izq": x_izq, "x_der": x_der,
        "y_arr": y_arr, "y_abaj": y_abaj,
        "umbral": umbral, "img_norm": img_norm,
    })


# =============================================================================
# DIBUJO 
# =============================================================================

def _crear_fig_ancho(H: int = 1, W: int = 1) -> tuple:
    """
    Crea la figura del ancho del haz.
    H, W: dimensiones de la imagen en píxeles — se usan para calcular
    el aspect ratio correcto del panel de imagen y ajustar las proporciones
    de los paneles de perfil para que coincidan visualmente con los lados
    de la imagen.
    """
    f = _factor()
    img_w_um = W * f
    img_h_um = H * f

    aspect_img = img_w_um / img_h_um

    img_h_in = 6.0
    img_w_in = img_h_in * aspect_img
    img_w_in = max(3.0, min(img_w_in, 10.0))
    img_h_in = img_w_in / aspect_img

    pv_w_in = 2.4    # panel perfil vertical (derecha)
    ph_h_in = 2.2    # panel perfil horizontal (abajo)
    cb_w_in = 1.2    # colorbar

    # Espacios físicos entre paneles
    hgap_in = 0.70   # espacio vertical entre imagen y perfil horizontal
    wgap_in = 0.65   # espacio horizontal entre imagen y perfil vertical

    fig_w = img_w_in + wgap_in + pv_w_in + cb_w_in + 1.6
    fig_h = img_h_in + hgap_in + ph_h_in + 1.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[img_w_in, pv_w_in],
        height_ratios=[img_h_in, ph_h_in],
        hspace=hgap_in / img_h_in,   # fracción del alto del subplot
        wspace=wgap_in / img_w_in,   # fracción del ancho del subplot
        left=0.09, right=0.87,
        top=0.92,  bottom=0.07,
    )

    ax_img = fig.add_subplot(gs[0, 0])
    ax_pv  = fig.add_subplot(gs[0, 1], sharey=ax_img)
    ax_ph  = fig.add_subplot(gs[1, 0], sharex=ax_img)
    ax_cb  = fig.add_subplot(gs[1, 1])
    ax_cb.axis("off")

    for a in [ax_img, ax_ph, ax_pv]:
        a.set_facecolor("white")
        for spine in a.spines.values():
            spine.set_edgecolor("#CCCCCC")

    # ── Tick labels: MOSTRAR en todos los paneles ─────────────────────────────
    # ax_ph: eje X compartido → mostrar labels en su parte inferior
    plt.setp(ax_ph.get_xticklabels(), visible=True, color="#333333")
    ax_ph.tick_params(axis="x", labelbottom=True, colors="#333333", labelsize=9)

    # ax_pv: eje Y compartido → mover labels al lado derecho para evitar empalme
    ax_pv.yaxis.set_ticks_position("right")
    ax_pv.yaxis.set_label_position("right")
    plt.setp(ax_pv.get_yticklabels(), visible=True, color="#333333")
    ax_pv.tick_params(axis="y", labelright=True, labelleft=False,
                      colors="#333333", labelsize=9)

    return fig, ax_img, ax_pv, ax_ph, ax_cb

def _dibujar_ancho_en_axes(ax_img, ax_ph, ax_pv, info: dict,
                            wx: float, wy: float, t: float, n_frame: int):
    """
    Dibuja la visualización diagnóstica del método de umbral sobre un
    conjunto de ejes ya creado por `_crear_fig_ancho`: la imagen del haz
    con los bordes detectados marcados, más los perfiles de intensidad
    horizontal y vertical que pasan por el centroide.

    Es el panel que permite AUDITAR visualmente la medición de ancho: se
    ve simultáneamente dónde quedó el centroide, qué perfil se extrajo y
    exactamente en qué píxel el perfil cruzó el umbral por cada lado. Si
    una medición de ancho parece anómala, esta figura muestra por qué.

    Recibe `info`, el dict devuelto por `calcular_ancho_haz` (perfiles y
    posiciones de cruce ya calculados), de modo que la figura refleja
    exactamente los mismos números que se reportan — nunca recalcula.

    Se reutiliza tanto para la figura estática 05 como para cada frame de
    los videos de ancho, garantizando que ambos sean idénticos.
    """
    cx       = info["cx"];    cy      = info["cy"]
    icy      = info["icy"];   icx     = info["icx"]
    perfil_h = info["perfil_h"]
    perfil_v = info["perfil_v"]
    x_izq    = info["x_izq"]; x_der  = info["x_der"]
    y_arr    = info["y_arr"]; y_abaj = info["y_abaj"]
    umbral   = info["umbral"]
    img_norm = info["img_norm"]
    H, W     = img_norm.shape

    f = _factor()
    u = _unidad()

    r_izq  = cx - x_izq
    r_der  = x_der - cx
    r_arr  = cy - y_arr
    r_abaj = y_abaj - cy

    ext = _extent(H, W)
    ax_img.imshow(img_norm, cmap="inferno", origin="upper",
                  aspect="auto", vmin=0, vmax=1, extent=ext)
    ax_img.axhline(icy * f, color="cyan", linestyle=":", linewidth=0.9, alpha=0.5)
    ax_img.axvline(icx * f, color="lime", linestyle=":", linewidth=0.9, alpha=0.5)

    fp_c = dict(arrowstyle="-|>", lw=2, mutation_scale=14, color="cyan")
    fp_l = dict(arrowstyle="-|>", lw=2, mutation_scale=14, color="lime")

    ax_img.annotate("", xy=(x_der*f, cy*f), xytext=(cx*f, cy*f), arrowprops=fp_c)
    ax_img.annotate("", xy=(x_izq*f, cy*f), xytext=(cx*f, cy*f), arrowprops=fp_c)
    ax_img.annotate("", xy=(cx*f, y_arr*f), xytext=(cx*f, cy*f), arrowprops=fp_l)
    ax_img.annotate("", xy=(cx*f, y_abaj*f), xytext=(cx*f, cy*f), arrowprops=fp_l)

    # Todas las etiquetas numericas se consolidan en un unico cuadro anclado
    # a una esquina fija de los ejes (coordenadas de figura, no de imagen),
    # para que NUNCA se superpongan entre si -- sin importar cuan pequeno
    # sea el haz, a diferencia de posicionarlas con offsets fijos en pixeles
    # cerca del centroide (que colisionaban cuando el haz era chico). Las
    # flechas (sin cambios) siguen indicando la direccion/extension real.
    resumen = (
        f"→ {_to_um(r_der):.1f} {u}    ← {_to_um(r_izq):.1f} {u}\n"
        f"↑ {_to_um(r_arr):.1f} {u}    ↓ {_to_um(r_abaj):.1f} {u}\n"
        f"wx = {_to_um(wx):.1f} {u}  |  wy = {_to_um(wy):.1f} {u}"
    )
    ax_img.text(0.98, 0.98, resumen, transform=ax_img.transAxes,
                fontsize=10.5, ha="right", va="top", color="#111111",
                linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.92))
    ax_img.plot(cx*f, cy*f, "o", color="red", markersize=8,
                markerfacecolor="none", markeredgewidth=2)
    ax_img.plot(cx*f, cy*f, "+", color="red", markersize=14, markeredgewidth=2,
                label=f"Centroide ({_to_um(cx):.1f}, {_to_um(cy):.1f}) {u}\n[punto de partida]")
    ax_img.legend(fontsize=11, loc="upper left",
                  facecolor="white", labelcolor="#111111", framealpha=0.9)
    ax_img.set_title(
        f"Frame {n_frame}  |  t = {t:.2f} s  |  umbral = {umbral:.2f}",
        fontsize=12, fontweight="bold", color="#111111")
    ax_img.set_xlabel(f"x ({u})", fontsize=11, color="#111111")
    ax_img.set_ylabel(f"y ({u})", fontsize=11, color="#111111")
    ax_img.tick_params(colors="#333333", labelsize=9)

    xs = np.arange(len(perfil_h)) * f
    ax_ph.plot(xs, perfil_h, color="steelblue", linewidth=1.8)
    ax_ph.axhline(umbral,  color="#888888",  linestyle=":",  linewidth=1)
    ax_ph.axvline(icx*f,   color="red",    linestyle="-",  linewidth=1.3, alpha=0.9)
    ax_ph.axvline(x_izq*f, color="darkorange", linestyle="--", linewidth=1.4)
    ax_ph.axvline(x_der*f, color="darkorange", linestyle="--", linewidth=1.4)
    ax_ph.fill_between(xs, perfil_h, 0,
                       where=(xs >= x_izq*f) & (xs <= x_der*f),
                       color="steelblue", alpha=0.22)
    ax_ph.fill_between(xs, perfil_h, 0,
                       where=(xs < x_izq*f) | (xs > x_der*f),
                       color="steelblue", alpha=0.08)
    ax_ph.plot(icx*f, perfil_h[icx], "o", color="red", markersize=5, zorder=5)
    y_fl = umbral + 0.05
    ax_ph.annotate("", xy=(x_izq*f, y_fl), xytext=(icx*f, y_fl),
                   arrowprops=dict(arrowstyle="<->", color="darkorange", lw=2.0, mutation_scale=11))
    ax_ph.annotate("", xy=(x_der*f, y_fl), xytext=(icx*f, y_fl),
                   arrowprops=dict(arrowstyle="<->", color="darkorange", lw=2.0, mutation_scale=11))
    ax_ph.set_xlim(0, (len(perfil_h) - 1) * f)
    ax_ph.set_ylim(-0.05, 1.12)
    ax_ph.set_xlabel(f"x ({u})", fontsize=11, color="#111111")
    ax_ph.set_ylabel("Intens. norm.", fontsize=11, color="#111111")
    ax_ph.tick_params(colors="#333333", labelsize=9)
    ax_ph.grid(True, linestyle="--", alpha=0.35)

    ys = np.arange(len(perfil_v)) * f
    ax_pv.plot(perfil_v, ys, color="forestgreen", linewidth=1.8)
    ax_pv.axvline(umbral,   color="#888888",  linestyle=":",  linewidth=1)
    ax_pv.axhline(icy*f,    color="red",    linestyle="-",  linewidth=1.3, alpha=0.9)
    ax_pv.axhline(y_arr*f,  color="darkorange", linestyle="--", linewidth=1.4)
    ax_pv.axhline(y_abaj*f, color="darkorange", linestyle="--", linewidth=1.4)
    ax_pv.fill_betweenx(ys, perfil_v, 0,
                        where=(ys >= y_arr*f) & (ys <= y_abaj*f),
                        color="forestgreen", alpha=0.22)
    ax_pv.fill_betweenx(ys, perfil_v, 0,
                        where=(ys < y_arr*f) | (ys > y_abaj*f),
                        color="forestgreen", alpha=0.08)
    ax_pv.plot(perfil_v[icy], icy*f, "o", color="red", markersize=5, zorder=5)
    x_fl = umbral + 0.05
    ax_pv.annotate("", xy=(x_fl, y_arr*f),  xytext=(x_fl, icy*f),
                   arrowprops=dict(arrowstyle="<->", color="darkorange", lw=2.0, mutation_scale=11))
    ax_pv.annotate("", xy=(x_fl, y_abaj*f), xytext=(x_fl, icy*f),
                   arrowprops=dict(arrowstyle="<->", color="darkorange", lw=2.0, mutation_scale=11))
    ax_pv.set_ylim((len(perfil_v) - 1) * f, 0)   # mantiene el eje Y invertido
    ax_pv.set_xlim(-0.05, 1.12)
    ax_pv.set_xlabel("Intens. norm.", fontsize=12, color="#111111")
    ax_pv.set_ylabel(f"y ({u})", fontsize=12, color="#111111")   # mostrado en lado derecho
    # Solo el eje x (intensidad) necesita estilo aqui: el eje y ya se
    # configuro correctamente (color oscuro, lado derecho) en _crear_fig_ancho;
    # sobreescribirlo sin axis= aqui volvia blancos (invisibles) ambos ejes.
    ax_pv.tick_params(axis="x", colors="#333333", labelsize=7)
    ax_pv.grid(True, linestyle="--", alpha=0.35)

def _agregar_colorbar_ancho(fig, ax_img, ax_cb, img_norm):
    """
    Añade la barra de color de intensidad normalizada a la figura de ancho
    de haz, alojándola en el cuadrante inferior derecho (`ax_cb`), que la
    rejilla de `_crear_fig_ancho` deja libre a propósito.

    Reutiliza la imagen ya dibujada en `ax_img` si existe, para que la
    escala de color corresponda exactamente a lo mostrado y no a un
    renderizado independiente.
    """
    im = ax_img.get_images()[0] if ax_img.get_images() else \
         ax_img.imshow(img_norm, cmap="inferno", origin="upper",
                       aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax_cb, fraction=0.9, pad=0.05, location="right")
    cbar.set_label("Intensidad normalizada", fontsize=11, color="#111111")
    cbar.ax.yaxis.set_tick_params(color="#333333")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#333333")


# =============================================================================
# SERIES COMPARTIDAS (gráficas estáticas y videos deben usar exactamente
# los mismos números -- antes cada par graficar_*/video_* recalculaba la
# serie por separado; si se corregía una fórmula en una copia y se
# olvidaba la otra, la gráfica estática y el video quedaban inconsistentes
# entre sí sin que fuera obvio)
# =============================================================================

def _serie_desplazamiento_centroide(frames, tiempos):
    """dx, dy [unidad activa] respecto al centroide promedio temporal, más
    estadísticos derivados (RMS, máximos, límite de ejes) usados tanto por
    graficar_desplazamiento_centroide como por video_desplazamiento_centroide."""
    cx_all = np.array([calcular_centroide(f)[0] for f in frames])
    cy_all = np.array([calcular_centroide(f)[1] for f in frames])
    cx0 = float(cx_all.mean())
    cy0 = float(cy_all.mean())
    dx = _to_um(cx_all - cx0)
    dy = _to_um(cy_all - cy0)
    dist_eucl = np.sqrt(dx**2 + dy**2)
    rms      = float(np.sqrt(np.mean(dx**2 + dy**2)))
    dx_max   = float(np.max(np.abs(dx)))
    dy_max   = float(np.max(np.abs(dy)))
    dist_max = float(dist_eucl.max())
    lim = max(np.abs(np.concatenate([dx, dy])).max(), 1e-6) * 1.15
    return dx, dy, dist_eucl, rms, dx_max, dy_max, dist_max, lim, _unidad()


def _serie_potencia_normalizada(frames):
    """Intensidad total normalizada por frame, más media/std/sigma_I, usada
    tanto por graficar_potencia_normalizada como por video_potencia_normalizada."""
    potencias = np.array([f.sum() for f in frames])
    p_norm    = potencias / potencias.mean()
    p_media   = float(p_norm.mean())
    p_std     = float(p_norm.std())
    sigma_I   = float((potencias**2).mean() / potencias.mean()**2 - 1)
    return potencias, p_norm, p_media, p_std, sigma_I


def _serie_correlacion_espacial(frames, ref_frame_flat=None):
    """Correlación de Pearson de cada frame contra una referencia, más
    media/std/límites de eje, usada tanto por graficar_correlacion_espacial
    como por video_correlacion_espacial.

    `ref_frame_flat`: referencia ya aplanada (1D) a usar en vez del frame 0
    de `frames`. `_ejecutar_analisis_caso` siempre pasa aquí el promedio
    del primer segundo del caso SinTurbulencia del mismo dispositivo (ver
    `_promedio_primer_segundo_video`), para que la correlación mida
    degradación respecto al estado normal del haz en los 3 casos, no
    autoconsistencia interna de cada video por separado. Si es None
    (p. ej. no se pudo ubicar/cargar el caso SinTurbulencia), cae de
    vuelta al frame 0 propio como respaldo.
    """
    ref     = ref_frame_flat if ref_frame_flat is not None else frames[0].flatten().astype(np.float64)
    corrs   = np.array([np.corrcoef(ref, f.flatten())[0, 1] for f in frames])
    c_media = float(corrs.mean())
    c_std   = float(corrs.std())
    rango   = max(corrs.max() - corrs.min(), 1e-9)
    margen  = rango * 0.20
    ylim    = [max(0.0, corrs.min() - margen), min(1.0 + margen, corrs.max() + margen)]
    return corrs, c_media, c_std, ylim


def ks_pvalue_bootstrap_rayleigh(dist_eucl: np.ndarray, sigma_hat: float,
                                  ks_stat_obs: float, n_boot: int = 1000,
                                  semilla: int = 0) -> float:
    """
    Valor p corregido (bootstrap paramétrico, equivalente al test de
    Lilliefors aplicado a la Rayleigh) para la bondad de ajuste del beam
    wander a una distribución Rayleigh.

    El valor p que retorna `scipy.stats.kstest` directamente NO es válido
    aquí: el test KS clásico asume una distribución de referencia con
    parámetros fijados DE ANTEMANO, independientes de la muestra. Como
    `sigma_hat` se estimó de la MISMA muestra que se está probando (MLE:
    σ = sqrt(mean(r²)/2)), la curva ajustada "persigue" a la distribución
    empírica y el estadístico KS observado tiende a ser menor de lo que
    sería bajo la hipótesis nula real -- esto infla artificialmente el
    valor p (el test dice "ajuste aceptable" con más frecuencia de la que
    corresponde). Es el mismo problema que resuelve el test de Lilliefors
    para la normal, aplicado aquí a la Rayleigh.

    Procedimiento: simular `n_boot` muestras de tamaño len(dist_eucl) de
    una Rayleigh(sigma_hat), RE-ESTIMAR sigma en cada muestra simulada
    (reproduciendo exactamente el procedimiento de estimar-y-probar sobre
    los datos reales) y calcular su propio estadístico KS. El valor p
    corregido es la fracción de esos estadísticos simulados que son
    mayores o iguales al observado.

    `semilla` fija por defecto (reproducibilidad experimental): dos
    corridas sobre los mismos datos deben dar el mismo valor p.
    """
    from scipy.stats import rayleigh as _rayleigh_dist, kstest as _kstest
    n = len(dist_eucl)
    if n < 2 or not np.isfinite(ks_stat_obs) or sigma_hat <= 0:
        return float("nan")
    rng = np.random.default_rng(semilla)
    d_sim = np.empty(n_boot)
    for i in range(n_boot):
        muestra = _rayleigh_dist.rvs(scale=sigma_hat, size=n, random_state=rng)
        sigma_sim = float(np.sqrt(np.mean(muestra ** 2) / 2.0))
        if sigma_sim <= 0:
            d_sim[i] = 0.0
            continue
        d_sim[i], _ = _kstest(muestra, "rayleigh", args=(0, sigma_sim))
    return float(np.mean(d_sim >= ks_stat_obs))


# =============================================================================
# GRÁFICAS ESTÁTICAS
# =============================================================================

def graficar_desplazamiento_centroide(frames, tiempos, carpeta_salida):
    """
    Gráfica 01 — La figura central del análisis de estabilidad: caracteriza
    el BEAM WANDER, es decir, el vagabundeo del centroide del haz a lo
    largo del tiempo.

    Produce dos paneles complementarios:
      - Izquierdo: mapa de dispersión (Δx, Δy) de cada frame respecto al
        centroide promedio temporal, coloreado por tiempo. Muestra la
        geometría del movimiento (¿isótropo? ¿deriva direccional?).
      - Derecho: histograma de la distancia radial r(t) = √(Δx²+Δy²) con
        el ajuste de una distribución de Rayleigh superpuesto.

    Por qué Rayleigh: si los desplazamientos en x e y son gaussianos,
    independientes y de igual varianza —lo esperable para turbulencia
    isótropa—, entonces r sigue necesariamente una distribución de
    Rayleigh. Que el ajuste sea bueno es, por tanto, evidencia de que el
    movimiento observado es consistente con turbulencia atmosférica y no
    con una deriva mecánica sistemática (que produciría otra forma).

    La bondad del ajuste se evalúa con un test KS cuyo valor p se corrige
    por bootstrap paramétrico (`ks_pvalue_bootstrap_rayleigh`), porque el
    parámetro σ se estimó de la misma muestra que se prueba.

    Genera `01_desplazamiento_centroide.png` + CSV con las series
    completas (t, Δx, Δy, r) en `Datos_Crudos/`.
    """
    from scipy.stats import rayleigh as rayleigh_dist
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    print("\n[1/6] Calculando desplazamiento del centroide...")
    dx, dy, dist_eucl, rms, dx_max, dy_max, dist_max, lim, u = \
        _serie_desplazamiento_centroide(frames, tiempos)

    # ── Ajuste de Rayleigh ────────────────────────────────────────────────────
    # Estimación MLE del parámetro σ de Rayleigh: σ = sqrt(mean(r²) / 2)
    sigma_ray = float(np.sqrt(np.mean(dist_eucl**2) / 2.0))
    # Bondad de ajuste: estadístico KS + valor p corregido por bootstrap
    # paramétrico (ver ks_pvalue_bootstrap_rayleigh -- el valor p directo de
    # kstest no es válido porque sigma_ray se estimó de la misma muestra).
    try:
        from scipy.stats import kstest
        ks_stat, _ = kstest(dist_eucl, "rayleigh", args=(0, sigma_ray))
        ks_p = ks_pvalue_bootstrap_rayleigh(dist_eucl, sigma_ray, ks_stat)
    except Exception:
        ks_stat, ks_p = float("nan"), float("nan")

    # ── Figura: scatter (izq) + histograma Rayleigh (der) ────────────────────
    fig, (ax_sc, ax_hi) = plt.subplots(
        1, 2, figsize=(14, 7),
        gridspec_kw={"wspace": 0.35})

    # ── Panel izquierdo: scatter x-y ─────────────────────────────────────────
    sc = ax_sc.scatter(dx, dy, c=tiempos, cmap="viridis", s=50, zorder=3,
                       label="Frames analizados")
    ax_sc.scatter(0, 0, color="red", s=140, zorder=4, marker="*",
                  label="Frame 0 (referencia)")

    div = make_axes_locatable(ax_sc)
    cax = div.append_axes("right", size="5%", pad=0.12)
    fig.colorbar(sc, cax=cax).set_label("Tiempo (s)", fontsize=11)

    ax_sc.annotate(
        f"RMS        = {rms:.3f} {u}\n"
        f"Δx máx.    = {dx_max:.3f} {u}\n"
        f"Δy máx.    = {dy_max:.3f} {u}\n"
        f"Dist. máx. = {dist_max:.3f} {u}",
        xy=(0.04, 0.96), xycoords="axes fraction",
        fontsize=12, fontweight="bold", color="#111111", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.92))
    ax_sc.set_xlim(-lim, lim);  ax_sc.set_ylim(-lim, lim)
    ax_sc.set_aspect('equal', adjustable='box')
    ax_sc.set_title("Mapa de dispersión del centroide\n(respecto al centroide promedio temporal)",
                    fontsize=12, fontweight="bold")
    ax_sc.set_xlabel(f"$\\Delta x$ ({u})", fontsize=12)
    ax_sc.set_ylabel(f"$\\Delta y$ ({u})", fontsize=12)
    # loc explícito (no "best") -- la anotación de RMS/máximos ocupa la
    # esquina superior izquierda; la leyenda va en la esquina opuesta para
    # que nunca se solapen entre sí.
    ax_sc.legend(fontsize=11, loc="lower right")
    ax_sc.grid(True, linestyle="--", alpha=0.6)

    # ── Panel derecho: histograma de r(t) + curva Rayleigh ───────────────────
    n_bins = max(15, min(50, len(dist_eucl) // 20))
    counts, bin_edges, patches = ax_hi.hist(
        dist_eucl, bins=n_bins, density=True,
        color="#4C72B0", edgecolor="#CCCCCC", linewidth=0.6,
        alpha=0.80, label="Distribución empírica")

    # Curva Rayleigh teórica
    r_plot = np.linspace(0, dist_eucl.max() * 1.15, 400)
    pdf_ray = rayleigh_dist.pdf(r_plot, scale=sigma_ray)
    ax_hi.plot(r_plot, pdf_ray, color="#DD4444", lw=2.2,
               label=f"Ajuste Rayleigh\n$\\sigma_R$ = {sigma_ray:.3f} {u}")

    # Línea vertical en la moda (= σ_R) y en el RMS
    ax_hi.axvline(sigma_ray, color="#DD4444", lw=1.2, ls="--", alpha=0.7)
    ax_hi.axvline(rms,       color="#888888", lw=1.2, ls=":",  alpha=0.8,
                  label=f"RMS = {rms:.3f} {u}")

    # Anotación bondad de ajuste
    ks_txt = (f"KS stat = {ks_stat:.4f}\np-valor (bootstrap) = {ks_p:.4f}"
              if not np.isnan(ks_stat) else "")
    interpret = ""
    if not np.isnan(ks_p):
        # "√"/"×" (no "✓"/"✗"): DejaVu Serif -- la fuente usada en todas
        # las figuras de este módulo -- no incluye los glifos de check
        # mark/ballot X reales, y los dibujaba en blanco (glifo faltante,
        # UserWarning en consola) en el PNG guardado.
        interpret = ("  √ ajuste aceptable" if ks_p > 0.05
                     else "  × desviación del modelo")
    ax_hi.annotate(
        ks_txt + interpret,
        xy=(0.97, 0.04), xycoords="axes fraction",
        fontsize=11, color="black", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eeeeee",
                  ec="#aaaaaa", alpha=0.9))

    ax_hi.set_title(
        "Distribución estadística del beam wander\n"
        "$r(t) = \\sqrt{\\Delta x^2 + \\Delta y^2}$",
        fontsize=12, fontweight="bold")
    ax_hi.set_xlabel(f"Distancia radial $r$ ({u})", fontsize=12)
    ax_hi.set_ylabel("Densidad de probabilidad", fontsize=12)
    # loc="upper right" (no "upper left"): el pico de la distribucion
    # Rayleigh cae en r=sigma, no en r=0 -- queda en la zona baja-media
    # del eje, así que "upper left" terminaba tapando las barras más
    # altas del histograma. La cola derecha (r grande) sí es, por
    # construcción del eje (xlim hasta dist_eucl.max()*1.15), siempre de
    # densidad baja/nula, así que esa esquina queda libre en cualquier
    # caso -- y no choca con la anotación de bondad de ajuste, que ocupa
    # la esquina inferior derecha (misma columna, distinta fila).
    ax_hi.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax_hi.grid(True, linestyle="--", alpha=0.5)
    ax_hi.set_xlim(left=0)

    fig.suptitle("Desplazamiento del centroide",
                 fontsize=14, fontweight="bold", y=1.01)

    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "01_desplazamiento_centroide.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _guardar_csv(os.path.join(_carpeta_datos_crudos(carpeta_salida), "01_desplazamiento_centroide.csv"),
                ["t_s", f"dx_{u}", f"dy_{u}", f"dist_{u}"],
                tiempos, dx, dy, dist_eucl)
    print(f"  Guardada: 01_desplazamiento_centroide.png  |  RMS = {rms:.3f} {u}"
          f"  |  σ_Rayleigh = {sigma_ray:.3f} {u}"
          f"  |  Dist. máx. = {dist_max:.3f} {u}")


def graficar_potencia_normalizada(frames, tiempos, carpeta_salida):
    """
    Gráfica 02: intensidad total normalizada I(t)/⟨I⟩ frame a frame, con
    su media/desviación estándar y el índice de centelleo σ_I anotados
    (ver MathematicalReference.md §7). Guarda PNG + CSV hermano en
    Datos_Crudos/.
    """
    print("\n[2/6] Calculando intensidad normalizada en función del tiempo...")
    potencias, p_norm, p_media, p_std, sigma_I = _serie_potencia_normalizada(frames)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos, p_norm, color="steelblue", linewidth=1.5,
            marker="o", markersize=3, label="Intensidad normalizada")
    ax.axhline(p_media, color="red", linestyle="--", linewidth=1,
               label=f"Media = {p_media:.4f}")
    ax.axhspan(p_media - p_std, p_media + p_std,
               color="red", alpha=0.10, label=f"$\\pm$ std = {p_std:.4f}")
    ax.annotate(f"I(t) = {p_media:.4f} $\\pm$ {p_std:.4f}\n$\\sigma_I$ = {sigma_I:.6f}",
                xy=(0.04, 0.96), xycoords="axes fraction",
                fontsize=12, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="steelblue", alpha=0.88))
    ax.set_title("Intensidad normalizada del haz en función del tiempo", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tiempo (s)", fontsize=12);  ax.set_ylabel("Intensidad normalizada (I / ⟨I)", fontsize=12)
    ax.legend(fontsize=12);  ax.grid(True, linestyle="--", alpha=0.6)

    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "02_intensidad_normalizada.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _guardar_csv(os.path.join(_carpeta_datos_crudos(carpeta_salida), "02_intensidad_normalizada.csv"),
                 ["tiempo_s", "I_norm", "I_total"],
                 tiempos.tolist(), p_norm.tolist(), potencias.tolist())
    print(f"  Guardada: 02_intensidad_normalizada.png  |  I(t) = {p_media:.4f} $\\pm$ {p_std:.4f}  |  s_I = {sigma_I:.6f}")


def graficar_correlacion_espacial(frames, tiempos, carpeta_salida, ref_frame_flat=None):
    """
    Gráfica 03: correlación de Pearson C(t) de cada frame contra la
    referencia (ver `ref_frame_flat` en `_serie_correlacion_espacial`),
    con media/desviación estándar anotadas. Ver la limitación
    metodológica importante en el comentario justo debajo (mezcla
    desplazamiento y cambio de forma en un solo número).
    """
    # LIMITACION METODOLOGICA CONOCIDA (documentada, no un bug): esta es una
    # correlacion de Pearson pixel-a-pixel entre cada frame y el frame de
    # referencia, SIN registrar/alinear las imagenes primero. Si el haz se
    # desplaza lateralmente (wander, ya cuantificado por separado en
    # graficar_desplazamiento_centroide), esa sola traslacion reduce la
    # correlacion aunque el PERFIL del haz no haya cambiado de forma en
    # absoluto. Es decir, esta metrica mezcla dos efectos fisicos distintos
    # (desplazamiento de posicion vs. cambio de forma/estructura del haz) en
    # un solo numero. Una caida de correlacion aqui no debe interpretarse
    # automaticamente como "degradacion estructural" -- podria ser enteramente
    # el mismo wander ya reportado en la Fig. 01. Recomendacion: aclarar esta
    # limitacion en la tesis al presentar esta figura/video.
    print("\n[3/6] Calculando correlacion espacial en función del tiempo...")
    corrs, c_media, c_std, ylim = _serie_correlacion_espacial(frames, ref_frame_flat)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos, corrs, color="darkorange", linewidth=1.5,
            marker="o", markersize=3, label="Correlacion con referencia SinTurbulencia")
    ax.axhline(c_media, color="red", linestyle="--", linewidth=1,
               label=f"Media = {c_media:.6f}")
    ax.axhspan(c_media - c_std, c_media + c_std,
               color="red", alpha=0.10, label=f"$\\pm$ std = {c_std:.6f}")
    ax.annotate(f"C(t) = {c_media:.6f} $\\pm$ {c_std:.6f}",
                xy=(0.04, 0.04), xycoords="axes fraction",
                fontsize=12, fontweight="bold", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="darkorange", alpha=0.88))
    ax.set_ylim(ylim)
    ax.set_title("Correlacion espacial respecto a SinTurbulencia en función del tiempo", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tiempo (s)", fontsize=12)
    ax.set_ylabel("Coeficiente de correlacion de Pearson", fontsize=12)
    ax.legend(fontsize=12);  ax.grid(True, linestyle="--", alpha=0.6)

    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "03_correlacion_espacial.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _guardar_csv(os.path.join(_carpeta_datos_crudos(carpeta_salida), "03_correlacion_espacial.csv"),
                 ["tiempo_s", "C_Pearson"],
                 tiempos.tolist(), corrs.tolist())
    print(f"  Guardada: 03_correlacion_espacial.png  |  C(t) = {c_media:.6f} $\\pm$ {c_std:.6f}")


def graficar_ancho_haz(frames, tiempos, carpeta_salida):
    """
    Gráfica 04: ancho horizontal/vertical del haz (método de umbral, ver
    `calcular_ancho_haz`) frame a frame, con medias/desviaciones
    anotadas. El CSV hermano conserva el prefijo histórico `D4sx_`/
    `D4sy_` por compatibilidad con datos ya generados, aunque el método
    no es D4σ real (ver nota junto al `_guardar_csv` más abajo).
    """
    print("\n[4/6] Calculando ancho del haz en función del tiempo...")
    anchos   = [calcular_ancho_haz(f)[:2] for f in frames]
    anchos_x = _to_um(np.array([a[0] for a in anchos]))
    anchos_y = _to_um(np.array([a[1] for a in anchos]))
    wx_media = float(anchos_x.mean());  wx_std = float(anchos_x.std())
    wy_media = float(anchos_y.mean());  wy_std = float(anchos_y.std())
    u        = _unidad()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos, anchos_x, color="royalblue", linewidth=1.5,
            marker="o", markersize=3, label="Ancho horizontal (wx)")
    ax.plot(tiempos, anchos_y, color="firebrick", linewidth=1.5,
            marker="s", markersize=3, label="Ancho vertical (wy)")
    ax.annotate(
        f"wx = {wx_media:.2f} $\\pm$ {wx_std:.2f} {u}\n"
        f"wy = {wy_media:.2f} $\\pm$ {wy_std:.2f} {u}",
        xy=(0.04, 0.96), xycoords="axes fraction",
        fontsize=12, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555555", alpha=0.88))
    ax.set_title(f"Evolución temporal del ancho del haz en las direcciones x e y [umbral = {UMBRAL_INTENSIDAD}]",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tiempo (s)", fontsize=12)
    ax.set_ylabel(f"Ancho del haz — método de umbral ({u})", fontsize=12)
    ax.legend(fontsize=12);  ax.grid(True, linestyle="--", alpha=0.6)

    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "04_ancho_haz.png"), facecolor="white", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Nota: el encabezado del CSV sigue usando el prefijo historico "D4sx_"/
    # "D4sy_" a proposito (no "D4sigma") -- ya existen ensayos anteriores en
    # disco con ese encabezado, y cambiarlo invalidaria comparaciones contra
    # datos ya generados. La etiqueta visible (graficas/consola) ya no dice
    # D4sigma porque este metodo NO es D4sigma real -- ver calcular_ancho_haz().
    _guardar_csv(os.path.join(_carpeta_datos_crudos(carpeta_salida), "04_ancho_haz.csv"),
                 ["tiempo_s", f"D4sx_{u}", f"D4sy_{u}"],
                 tiempos.tolist(), anchos_x.tolist(), anchos_y.tolist())
    print(f"  Guardada: 04_ancho_haz.png  |  wx = {wx_media:.2f} ± {wx_std:.2f} {u}  |  wy = {wy_media:.2f} ± {wy_std:.2f} {u}")


def graficar_imagen_referencia(frames, tiempos, carpeta_salida):
    """
    Gráfica 05: visualización diagnóstica del método de umbral
    (`calcular_ancho_haz`) sobre el frame 0 — imagen + perfiles
    horizontal/vertical con los bordes detectados superpuestos.
    """
    print("\n[5/6] Generando imagen de referencia (frame 0) del calculo del ancho del haz por método del umbral...")
    wx, wy, info = calcular_ancho_haz(frames[0])
    H, W = info["img_norm"].shape
    fig, ax_img, ax_pv, ax_ph, ax_cb = _crear_fig_ancho(H, W)
    _agregar_colorbar_ancho(fig, ax_img, ax_cb, info["img_norm"])
    ax_img.cla();  ax_img.set_facecolor("white")
    _dibujar_ancho_en_axes(ax_img, ax_ph, ax_pv, info, wx, wy, t=tiempos[0], n_frame=0)
    ax_img.set_title(f"Frame 0 (primer frame) — referencia  |  umbral = {info['umbral']:.2f}",
                     fontsize=11, fontweight="bold", color="#111111")
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "05_imagen_referencia_haz.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    carpeta_datos5 = _carpeta_datos_crudos(carpeta_salida)
    _guardar_matriz_csv(os.path.join(carpeta_datos5, "05_imagen_referencia_haz_datos.csv"),
                        info["img_norm"])
    _guardar_resumen_txt(os.path.join(carpeta_datos5, "05_imagen_referencia_haz_resumen.txt"),
                         {"wx_px": wx, "wy_px": wy, "cx_px": info["cx"], "cy_px": info["cy"],
                          "umbral": info["umbral"]})
    print("  Guardada: 05_imagen_referencia_haz.png")


def graficar_imagen_normalizada(frames, carpeta_salida):
    """
    Gráfica 06: distribución espacial de intensidad del frame 0, ya
    normalizada a [0,1] (`normalizar_frame`), sin marcas de ancho/umbral
    superpuestas (a diferencia de la Gráfica 05) — visualización pura de
    la forma del haz.
    """
    print("\n[6/6] Generando imagen normalizada del haz (Frame 0)...")
    img_norm = normalizar_frame(frames[0])
    H, W     = img_norm.shape
    u        = _unidad()

    # Calcular tamaño de figura proporcional a la imagen para que los ejes queden iguales
    aspect = W / H
    fig_h  = 7.0
    fig_w  = max(6.0, min(fig_h * aspect + 1.8, 14.0))   # +1.8 para colorbar

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(img_norm, cmap="inferno", origin="upper",
                   vmin=0, vmax=1, aspect="equal", extent=_extent(H, W))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensidad normalizada [0 – 1]", fontsize=11)
    cbar.set_ticks(np.linspace(0, 1, 11))
    ax.set_title("Distribución espacial de la intensidad normalizada del haz — Frame inicial",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"x ({u})", fontsize=12)
    ax.set_ylabel(f"y ({u})", fontsize=12)

    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "06_imagen_normalizada_haz.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _guardar_matriz_csv(os.path.join(_carpeta_datos_crudos(carpeta_salida),
                                     "06_imagen_normalizada_haz_datos.csv"),
                        img_norm)
    print("  Guardada: 06_imagen_normalizada_haz.png")


# =============================================================================
# VIDEOS
# =============================================================================
# Cada video anima la MISMA serie temporal que ya reporta su gráfica
# estática equivalente (comparten las funciones `_serie_*`), pero dibujando
# la curva progresivamente frame a frame.
#
# Aportan algo que la gráfica estática no puede: permiten ver el video del
# haz real y la métrica derivada avanzando en sincronía, lo que hace
# evidente a simple vista si un pico de la curva corresponde a un suceso
# físico visible (una ráfaga de turbulencia, una vibración) o a un
# artefacto de medición en un frame concreto. Son material de verificación
# y de presentación, no una fuente de datos distinta.

def _video_grafica_temporal(tiempos, series_data, config, ruta_out,
                             nombre_corto, fps_salida, repeticiones=1, anotacion="",
                             frame_indices=None):
    """
    Motor genérico de los videos de serie temporal: dibuja, frame a frame,
    una o varias curvas que crecen hacia la derecha y escribe el resultado
    como .mp4.

    Lo comparten los videos 02/03/04 (intensidad, correlación, ancho), que
    solo se diferencian en los datos y el formato pasados en `series_data`
    y `config` — así, un cambio de estilo se aplica a los tres a la vez.

    `frame_indices`: índices REALES del video original, para que el rótulo
    muestre el número de frame correcto cuando el análisis submuestrea
    (`intervalo_frames` > 1) en lugar del índice local de la serie.
    `repeticiones`: repite cada frame N veces para ralentizar la
    reproducción sin recalcular nada.
    """
    writer = None
    total  = len(tiempos)
    for i in range(total):
        if i % max(1, total // 20) == 0:
            print(f"    {nombre_corto}: frame {i + 1}/{total}...", end="\r")

        real_frame = frame_indices[i] if frame_indices is not None else i

        # Figura mas ancha, y titulo dividido en dos lineas (descripcion
        # estatica + tiempo/frame dinamico) para que el titulo completo
        # nunca se recorte, sin importar cuan largo sea 'titulo_base'.
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for k, datos in enumerate(series_data):
            ax.plot(tiempos[:i + 1], datos[:i + 1],
                    color=config["colores"][k], linewidth=1.5,
                    marker=config["markers"][k], markersize=3,
                    label=config["labels"][k])
        for hl in config.get("hlines", []):
            ax.axhline(hl["y"], color=hl["color"], linestyle="--", linewidth=1,
                       label=hl.get("label", ""))
        if "ylim" in config:
            ax.set_ylim(config["ylim"])
        fig.suptitle(config['titulo_base'], fontsize=13, fontweight="bold", y=0.98)
        ax.set_title(f"t = {tiempos[i]:.2f} s   (Frame {real_frame})",
                     fontsize=10.5, color="#555555")
        ax.set_xlabel("Tiempo (s)", fontsize=11)
        ax.set_ylabel(config["ylabel"], fontsize=11)
        ax.set_xlim(tiempos[0], tiempos[-1])
        ax.legend(fontsize=11);  ax.grid(True, linestyle="--", alpha=0.6)
        if anotacion:
            ax.annotate(anotacion, xy=(0.04, 0.96), xycoords="axes fraction",
                        fontsize=11, fontweight="bold", va="top",
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555555", alpha=0.88))
        fig.subplots_adjust(top=0.85, left=0.08, right=0.97, bottom=0.11)
        _agregar_marca_pie(fig)
        frame_bgr = fig_a_frame_bgr(fig)
        if writer is None:
            w, h = fig.canvas.get_width_height()
            writer = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_salida, (w, h))
        for _ in range(repeticiones):
            writer.write(frame_bgr)
        plt.close(fig)
    if writer:
        writer.release()
    print(f"\n    Guardado: {os.path.basename(ruta_out)}")


def video_desplazamiento_centroide(frames, tiempos, carpeta_salida, fps_salida,
                                    repeticiones=1, frame_indices=None):
    """
    Video 01 — Anima el mapa de dispersión del centroide: cada frame añade
    un punto (Δx, Δy) al scatter, dejando visible la traza acumulada.

    Complementa la gráfica estática 01 mostrando el ORDEN TEMPORAL del
    recorrido, que el scatter estático pierde: permite distinguir un
    vagabundeo aleatorio (nube que se llena de forma dispersa) de una
    deriva sistemática (traza que avanza en una dirección preferente),
    dos situaciones físicamente muy distintas que producen mapas finales
    parecidos.

    No tiene su propio CSV: reutiliza `_serie_desplazamiento_centroide`,
    los mismos números ya exportados por la gráfica 01.
    """
    print("\n  [V1/5] Generando video: desplazamiento del centroide en función del tiempo...")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    dx, dy, dist_eucl, rms, dx_max, dy_max, dist_max, lim, u = \
        _serie_desplazamiento_centroide(frames, tiempos)

    ruta_out = os.path.join(carpeta_salida, "01_video_desplazamiento_centroide.mp4")
    writer   = None
    total    = len(frames)
    for i in range(total):
        if i % max(1, total // 20) == 0:
            print(f"    frame {i + 1}/{total}...", end="\r")
        real_frame = frame_indices[i] if frame_indices is not None else i
        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(dx[:i + 1], dy[:i + 1], c=tiempos[:i + 1], cmap="viridis", s=50,
                        vmin=tiempos[0], vmax=tiempos[-1], zorder=3)
        ax.scatter(0, 0, color="red", s=140, marker="*", zorder=4,
                   label="Centroide promedio (referencia)")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.12)
        fig.colorbar(sc, cax=cax).set_label("Tiempo (s)", fontsize=12)
        ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Desplazamiento del centroide — t = {tiempos[i]:.2f} s  (Frame {real_frame})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"$\\Delta x$ ({u})", fontsize=11)
        ax.set_ylabel(f"$\\Delta y$ ({u})", fontsize=11)
        ax.annotate(
            f"RMS        = {rms:.3f} {u}\n"
            f"Δx máx.    = {dx_max:.3f} {u}\n"
            f"Δy máx.    = {dy_max:.3f} {u}\n"
            f"Dist. máx. = {dist_max:.3f} {u}",
            xy=(0.04, 0.96), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color="#111111", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#AAAAAA", alpha=0.92))
        # loc explícito -- la anotación de RMS/máximos ocupa la esquina
        # superior izquierda; la leyenda va en la esquina opuesta.
        ax.legend(fontsize=12, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        _agregar_marca_pie(fig)
        frame_bgr = fig_a_frame_bgr(fig)
        if writer is None:
            w, h = fig.canvas.get_width_height()
            writer = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_salida, (w, h))
        for _ in range(repeticiones):
            writer.write(frame_bgr)
        plt.close(fig)
    if writer:
        writer.release()
    print("\n    Guardado: 01_video_desplazamiento_centroide.mp4")


def video_potencia_normalizada(frames, tiempos, carpeta_salida, fps_salida,
                                repeticiones=1, frame_indices=None):
    """
    Video 02 — Anima la intensidad total normalizada I(t)/⟨I⟩, es decir,
    el CENTELLEO (scintillation) del haz: las fluctuaciones de potencia
    recibida causadas por la turbulencia.

    Reutiliza `_serie_potencia_normalizada`, la misma serie de la gráfica
    02, e imprime en la anotación el índice de centelleo σ_I ya calculado.
    """
    print("\n  [V2/5] Generando video: intensidad normalizada en función del tiempo...")
    potencias, p_norm, p_media, p_std, sigma_I = _serie_potencia_normalizada(frames)
    anotacion = f"I(t) = {p_media:.4f} $\\pm$ {p_std:.4f}\n$\\sigma_I$ = {sigma_I:.6f}"
    _video_grafica_temporal(
        tiempos, [p_norm],
        config={"titulo_base": "Intensidad normalizada del haz en función del tiempo",
                "ylabel": "Intensidad normalizada (I / ⟨I))",
                "labels": ["Intensidad normalizada"], "colores": ["steelblue"],
                "markers": ["o"],
                "hlines": [{"y": p_media, "color": "red", "label": f"Media = {p_media:.4f}"}]},
        ruta_out=os.path.join(carpeta_salida, "02_video_intensidad_normalizada.mp4"),
        nombre_corto="Intens. norm.", fps_salida=fps_salida,
        repeticiones=repeticiones, anotacion=anotacion, frame_indices=frame_indices)


def video_correlacion_espacial(frames, tiempos, carpeta_salida, fps_salida,
                                repeticiones=1, frame_indices=None, ref_frame_flat=None):
    """
    Video 03 — Anima la correlación de Pearson C(t) entre cada frame y la
    referencia sin turbulencia: cuánto se parece el haz, en cada instante,
    a su propio estado no perturbado.

    Es el video más útil para inspección visual conjunta: al reproducirlo
    junto al video del haz se ve directamente qué deformación concreta del
    patrón corresponde a cada caída de la curva.

    `ref_frame_flat`: referencia compartida (promedio del primer segundo
    de SinTurbulencia) aplanada; ver `_serie_correlacion_espacial`.
    """
    # Ver limitacion metodologica documentada en graficar_correlacion_espacial
    # (mezcla wander con cambio de forma del haz) -- aplica igual aqui.
    print("\n  [V3/5] Generando video: correlacion espacial en función del tiempo...")
    corrs, c_media, c_std, ylim = _serie_correlacion_espacial(frames, ref_frame_flat)
    anotacion = f"C(t) = {c_media:.6f} $\\pm$ {c_std:.6f}"
    _video_grafica_temporal(
        tiempos, [corrs],
        config={"titulo_base": "Correlacion espacial respecto a SinTurbulencia en función del tiempo",
                "ylabel": "Coeficiente de correlacion de Pearson",
                "labels": ["Correlacion con referencia SinTurbulencia"], "colores": ["darkorange"],
                "markers": ["o"],
                "hlines": [{"y": c_media, "color": "red", "label": f"Media = {c_media:.6f}"}],
                "ylim": ylim},
        ruta_out=os.path.join(carpeta_salida, "03_video_correlacion_espacial.mp4"),
        nombre_corto="Correlacion", fps_salida=fps_salida,
        repeticiones=repeticiones, anotacion=anotacion, frame_indices=frame_indices)


def video_ancho_haz_temporal(frames, tiempos, carpeta_salida, fps_salida,
                              repeticiones=1, frame_indices=None):
    """
    Video 04 — Anima los anchos horizontal y vertical del haz frente al
    tiempo (curvas wx(t) y wy(t) superpuestas), cuantificando el
    ENSANCHAMIENTO (beam spreading) inducido por la turbulencia.

    Que las dos curvas se muevan juntas indica un ensanchamiento
    aproximadamente isótropo; que se separen indica deformación con
    dirección preferente. Ancho medido por el método de umbral, ver
    `calcular_ancho_haz`.
    """
    print("\n  [V4/5] Generando video: ancho del haz en función del tiempo...")
    anchos   = [calcular_ancho_haz(f)[:2] for f in frames]
    anchos_x = _to_um(np.array([a[0] for a in anchos]))
    anchos_y = _to_um(np.array([a[1] for a in anchos]))
    wx_media = float(anchos_x.mean());  wx_std = float(anchos_x.std())
    wy_media = float(anchos_y.mean());  wy_std = float(anchos_y.std())
    u        = _unidad()
    anotacion = (f"wx = {wx_media:.2f} $\\pm$ {wx_std:.2f} {u}\n"
                 f"wy = {wy_media:.2f} $\\pm$ {wy_std:.2f} {u}")
    _video_grafica_temporal(
        tiempos, [anchos_x, anchos_y],
        config={
            "titulo_base": f"Evolución temporal del ancho del haz en las direcciones x e y  [umbral = {UMBRAL_INTENSIDAD}]",
            "ylabel":      f"Ancho del haz ({u})",
            "labels":      ["Ancho horizontal (wx)", "Ancho vertical (wy)"],
            "colores":     ["royalblue", "firebrick"],
            "markers":     ["o", "s"],
        },
        ruta_out=os.path.join(carpeta_salida, "04_video_ancho_haz_temporal.mp4"),
        nombre_corto="Ancho haz", fps_salida=fps_salida,
        repeticiones=repeticiones, anotacion=anotacion, frame_indices=frame_indices)


def video_ancho_haz_frames(frames, tiempos, carpeta_salida, fps_salida,
                            repeticiones=1, frame_indices=None):
    """
    Video 05 — Anima la figura diagnóstica del método de umbral (la misma
    de la gráfica 05, `_dibujar_ancho_en_axes`) aplicada a CADA frame del
    video: imagen del haz, perfiles horizontal/vertical y bordes
    detectados, todo evolucionando en el tiempo.

    Es la herramienta de AUDITORÍA del método de medición de ancho:
    permite confirmar frame a frame que el umbral detecta bordes
    razonables incluso cuando el haz se deforma bajo turbulencia. Si el
    ensanchamiento reportado en el video 04 tuviera un artefacto, aquí se
    ve exactamente en qué frame y por qué.

    A diferencia de los otros videos, no delega en
    `_video_grafica_temporal`: recalcula y redibuja la figura completa por
    frame, por lo que es notablemente más lento de generar.
    """
    print("\n  [V5/5] Generando video: calculo del ancho del haz por método del umbral...")
    ruta_out = os.path.join(carpeta_salida, "06_video_ancho_haz_frames.mp4")
    writer   = None
    total    = len(frames)
    for i, frame in enumerate(frames):
        if i % max(1, total // 20) == 0:
            print(f"    frame {i + 1}/{total}...", end="\r")
        real_frame = frame_indices[i] if frame_indices is not None else i
        wx, wy, info = calcular_ancho_haz(frame)
        H, W = info["img_norm"].shape
        fig, ax_img, ax_pv, ax_ph, ax_cb = _crear_fig_ancho(H, W)
        _agregar_colorbar_ancho(fig, ax_img, ax_cb, info["img_norm"])
        ax_img.cla();  ax_img.set_facecolor("white")
        _dibujar_ancho_en_axes(ax_img, ax_ph, ax_pv, info, wx, wy,
                                t=tiempos[i], n_frame=real_frame)
        _agregar_marca_pie(fig)
        frame_bgr = fig_a_frame_bgr(fig)
        if writer is None:
            w, h = fig.canvas.get_width_height()
            writer = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_salida, (w, h))
        for _ in range(repeticiones):
            writer.write(frame_bgr)
        plt.close(fig)
    if writer:
        writer.release()
    print("\n    Guardado: 06_video_ancho_haz_frames.mp4")


def video_resumen_analisis(frames, tiempos, carpeta_salida, fps_salida,
                            repeticiones=1, frame_indices=None):
    """
    Video resumen de 5 paneles, pensado para verse cómodo en una
    computadora o proyector: haz óptico (con ancho por método de umbral), mapa de
    desplazamiento del centroide, evolución temporal del ancho del haz,
    intensidad normalizada y correlación espacial — todos sincronizados
    al mismo frame.

    Reutiliza exactamente las mismas primitivas que ya usan los videos
    individuales (calcular_centroide, calcular_ancho_haz, la suma de
    intensidad y np.corrcoef) en una sola pasada por los frames, sin
    volver a decodificar el video ni duplicar la física.
    """
    print("\n  [V6/6] Generando video resumen (5 paneles)...")
    ruta_out = os.path.join(carpeta_salida, "07_video_resumen_analisis.mp4")
    total = len(frames)
    u = _unidad()
    f_esc = _factor()

    # ── Una sola pasada: calcular las 5 series completas por adelantado ────
    # Nota: NO se guarda el diccionario "info" (ni su "img_norm") de cada
    # frame -- para un video de N frames eso retendria N copias completas
    # de la imagen normalizada en float64 simultaneamente en RAM (decenas
    # de GB para videos largos), agotando la memoria disponible cerca del
    # final del bucle. img_norm se recalcula (barato) en el bucle de
    # renderizado de mas abajo, y cx/cy ya quedan guardados en cxs/cys.
    cx0, cy0 = calcular_centroide(frames[0])
    cxs = np.zeros(total); cys = np.zeros(total)
    anchos_x = np.zeros(total); anchos_y = np.zeros(total)
    potencias = np.zeros(total)
    corrs = np.ones(total)
    ref_flat = frames[0].astype(np.float64).flatten()
    for i, frame in enumerate(frames):
        if i % max(1, total // 20) == 0:
            print(f"    resumen: calculando frame {i + 1}/{total}...", end="\r")
        cx, cy = calcular_centroide(frame)
        cxs[i], cys[i] = cx, cy
        wx, wy, _ = calcular_ancho_haz(frame)
        anchos_x[i], anchos_y[i] = wx, wy
        f64 = frame.astype(np.float64)
        potencias[i] = f64.sum()
        if i > 0:
            corrs[i] = np.corrcoef(ref_flat, f64.flatten())[0, 1]
    p_norm = potencias / potencias.mean()
    dx_um = _to_um(cxs - cx0)
    dy_um = _to_um(cys - cy0)
    anchos_x_um = _to_um(anchos_x)
    anchos_y_um = _to_um(anchos_y)

    lim_dx = max(np.abs(dx_um).max(), np.abs(dy_um).max(), 1.0) * 1.15

    writer = None
    for i in range(total):
        if i % max(1, total // 20) == 0:
            print(f"    resumen: render frame {i + 1}/{total}...", end="\r")
        real_frame = frame_indices[i] if frame_indices is not None else i
        img_norm = normalizar_frame(frames[i])
        H, W = img_norm.shape

        fig = plt.figure(figsize=(16, 9), facecolor="white")
        gs = gridspec.GridSpec(
            2, 3, figure=fig,
            width_ratios=[1.2, 1, 1], height_ratios=[1, 1],
            hspace=0.40, wspace=0.30,
            left=0.055, right=0.985, top=0.88, bottom=0.08)

        # ── Panel 1 (grande, ocupa las 2 filas): haz óptico ─────────────────
        ax_beam = fig.add_subplot(gs[:, 0])
        ax_beam.imshow(img_norm, cmap="inferno", origin="upper", aspect="equal",
                       vmin=0, vmax=1, extent=_extent(H, W))
        ax_beam.plot(cxs[i]*f_esc, cys[i]*f_esc, "+", color="cyan",
                    markersize=13, markeredgewidth=2)
        ax_beam.set_title("Haz óptico", fontsize=13, fontweight="bold", color="#111111")
        ax_beam.set_xlabel(f"x ({u})", fontsize=10.5, color="#111111")
        ax_beam.set_ylabel(f"y ({u})", fontsize=10.5, color="#111111")
        ax_beam.tick_params(colors="#333333", labelsize=9)
        ax_beam.text(0.98, 0.98,
                    f"wx = {anchos_x_um[i]:.1f} {u}\nwy = {anchos_y_um[i]:.1f} {u}",
                    transform=ax_beam.transAxes, ha="right", va="top",
                    fontsize=10, color="#111111", linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#AAAAAA", alpha=0.9))

        # ── Panel 2: mapa de desplazamiento del centroide ───────────────────
        ax_cent = fig.add_subplot(gs[0, 1])
        ax_cent.plot(dx_um[:i+1], dy_um[:i+1], "-", color="mediumpurple",
                    linewidth=0.8, alpha=0.75)
        ax_cent.scatter([dx_um[i]], [dy_um[i]], color="red", s=35, zorder=5)
        ax_cent.scatter([0], [0], color="black", marker="x", s=35, zorder=5)
        ax_cent.set_xlim(-lim_dx, lim_dx); ax_cent.set_ylim(-lim_dx, lim_dx)
        ax_cent.set_aspect("equal")
        ax_cent.set_title("Desplazamiento del centroide", fontsize=11.5, fontweight="bold")
        ax_cent.set_xlabel(f"Δx ({u})", fontsize=9.5)
        ax_cent.set_ylabel(f"Δy ({u})", fontsize=9.5)
        ax_cent.grid(True, linestyle="--", alpha=0.4)

        # ── Panel 3: ancho del haz vs tiempo ─────────────────────────────────
        ax_w = fig.add_subplot(gs[0, 2])
        ax_w.plot(tiempos[:i+1], anchos_x_um[:i+1], color="royalblue", linewidth=1.4, label="wx")
        ax_w.plot(tiempos[:i+1], anchos_y_um[:i+1], color="firebrick", linewidth=1.4, label="wy")
        ax_w.set_xlim(tiempos[0], tiempos[-1])
        ax_w.set_title("Ancho del haz (método de umbral)", fontsize=11.5, fontweight="bold")
        ax_w.set_xlabel("Tiempo (s)", fontsize=9.5)
        ax_w.set_ylabel(f"Ancho ({u})", fontsize=9.5)
        ax_w.legend(fontsize=8, loc="upper right")
        ax_w.grid(True, linestyle="--", alpha=0.4)

        # ── Panel 4: intensidad normalizada vs tiempo ────────────────────────
        ax_int = fig.add_subplot(gs[1, 1])
        ax_int.plot(tiempos[:i+1], p_norm[:i+1], color="darkorange", linewidth=1.4)
        ax_int.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
        ax_int.set_xlim(tiempos[0], tiempos[-1])
        ax_int.set_title("Intensidad normalizada", fontsize=11.5, fontweight="bold")
        ax_int.set_xlabel("Tiempo (s)", fontsize=9.5)
        ax_int.set_ylabel(r"I(t) / $\langle I \rangle$", fontsize=9.5)
        ax_int.grid(True, linestyle="--", alpha=0.4)

        # ── Panel 5: correlación espacial vs tiempo ──────────────────────────
        ax_corr = fig.add_subplot(gs[1, 2])
        ax_corr.plot(tiempos[:i+1], corrs[:i+1], color="seagreen", linewidth=1.4)
        ax_corr.set_ylim(min(0.9, float(np.nanmin(corrs)) - 0.02), 1.02)
        ax_corr.set_xlim(tiempos[0], tiempos[-1])
        ax_corr.set_title("Correlación espacial", fontsize=11.5, fontweight="bold")
        ax_corr.set_xlabel("Tiempo (s)", fontsize=9.5)
        ax_corr.set_ylabel("C(t)", fontsize=9.5)
        ax_corr.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(
            f"Resumen del análisis de estabilidad temporal  |  "
            f"t = {tiempos[i]:.2f} s   (Frame {real_frame})",
            fontsize=15, fontweight="bold", y=0.965)

        _agregar_marca_pie(fig)
        frame_bgr = fig_a_frame_bgr(fig)
        if writer is None:
            w, h = fig.canvas.get_width_height()
            writer = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_salida, (w, h))
        for _ in range(repeticiones):
            writer.write(frame_bgr)
        plt.close(fig)
    if writer:
        writer.release()
    print("\n    Guardado: 07_video_resumen_analisis.mp4")


# =============================================================================
# MÓDULO DE ANÁLISIS DE POLARIZACIONES
# =============================================================================
# Analiza el conjunto de imágenes capturadas en la Etapa 2 (hasta 10
# estados de polarización distintos, ver config.PolarizationConfig).
#
# Pregunta científica que responde este módulo: ¿el perfil espacial del
# haz depende del estado de polarización de entrada? Un haz estructurado
# solo es útil para la tesis si su patrón es ESTABLE frente a esa
# variable; si se deformara al cambiar la polarización, no podría
# atribuirse a la turbulencia ningún cambio observado más adelante.
#
# Las métricas se agrupan en tres familias complementarias:
#   - Morfológicas por imagen (D4σ, elipticidad, radios de energía
#     encerrada): ¿cambia el tamaño/forma del haz entre estados?
#   - De consistencia entre estados (matriz de correlación cruzada, mapa
#     de varianza espacial, dispersión del centroide): ¿es el MISMO
#     patrón en todos los estados?
#   - De acoplamiento FSO (área efectiva, integral de solapamiento,
#     sensibilidad a desalineación): ¿qué implicaría este perfil en un
#     enlace óptico en espacio libre real?

def encontrar_carpeta_polarizaciones(carpeta_entrada: str) -> str | None:
    """Localiza la subcarpeta de polarizaciones del HAZ (Etapa 2) dentro de
    Adquisicion/ o Preprocesado/. Retorna None si el caso no incluyó esa
    etapa — el análisis de polarizaciones se omite entonces sin error."""
    for item in os.listdir(carpeta_entrada):
        ruta_item = os.path.join(carpeta_entrada, item)
        if os.path.isdir(ruta_item) and "diferentes_polarizaciones_haz" in item.lower():
            return ruta_item
    return None

def cargar_imagenes_polarizacion(carpeta_pol: str) -> tuple[list, list]:
    """
    Carga en memoria todas las imágenes de polarización de una carpeta,
    en escala de grises y como float64 (necesario para los momentos
    estadísticos posteriores, que perderían precisión en uint8).

    El orden alfabético de archivo determina el orden de las
    polarizaciones (P01, P02, ...), lo que hace que la matriz de
    correlación y las tablas sean directamente interpretables por índice.

    Retorna (lista_de_imagenes, lista_de_nombres_de_archivo). Lanza
    FileNotFoundError si la carpeta no contiene ninguna imagen legible;
    las imágenes individuales ilegibles se omiten con aviso.
    """
    extensiones = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg",
                   "*.JPEG", "*.tif", "*.tiff", "*.bmp", "*.BMP"]
    archivos = []
    for ext in extensiones:
        archivos.extend(glob.glob(os.path.join(carpeta_pol, ext)))
    archivos = sorted(set(archivos))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron imagenes en: {carpeta_pol}")
    imagenes, nombres = [], []
    for ruta in archivos:
        img = utils_imagenes.leer_imagen(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  No se pudo leer: {os.path.basename(ruta)}")
            continue
        imagenes.append(img.astype(np.float64))
        nombres.append(os.path.basename(ruta))
    print(f"  Imagenes cargadas: {len(imagenes)}")
    for n in nombres:
        print(f"    · {n}")
    return imagenes, nombres

def calcular_segundo_momento(img, cx, cy, umbral_frac=0.05):
    """
    Segundo momento espacial (D4sigma real). Aplica umbral ISO 11146 antes
    de integrar: solo se consideran pixeles por encima de umbral_frac*I_max,
    porque el momento de segundo orden pondera por r^2 y cualquier fondo/
    ruido residual lejos del centro (incluso pequeno) infla sigma_x/sigma_y
    de forma no lineal. Mismo criterio (umbral_frac=0.05 por defecto) que
    CamaraTurbulencia._segundo_momento, para que las implementaciones
    reales de D4sigma del proyecto sean consistentes entre si.

    No lanza excepcion: si el frame no tiene señal medible tras aplicar
    el umbral, retorna (0.0, 0.0) con print_warn explícito -- valor no
    físico, el llamador debería considerar excluir el frame.
    """
    i_max = img.max() if img.size else 0.0
    img_u = img
    if i_max > 0:
        img_u = img.copy()
        img_u[img_u < umbral_frac * i_max] = 0.0
    H, W = img_u.shape
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    total = img_u.sum()
    if total == 0:
        print_warn("Frame sin señal medible tras aplicar el umbral ISO 11146 — "
                   "sigma_x/sigma_y devueltos como 0 (valor no físico, revisar si "
                   "este frame debe excluirse del análisis).")
        return 0.0, 0.0
    sigma_x = np.sqrt(((XX - cx) ** 2 * img_u).sum() / total)
    sigma_y = np.sqrt(((YY - cy) ** 2 * img_u).sum() / total)
    return float(sigma_x), float(sigma_y)

def calcular_radio_energia_encerrada(img, cx, cy, fracciones=None, umbral_frac=0.05):
    """
    Radios de energia encerrada (r_50%, r_86%, r_99%, ...). Aplica el mismo
    umbral ISO 11146 que calcular_segundo_momento antes de acumular energia,
    para que fondo/ruido residual no infle artificialmente el radio
    necesario para alcanzar fracciones altas (r_99 en particular).
    """
    if fracciones is None:
        fracciones = FRACCIONES_ENERGIA
    i_max = img.max() if img.size else 0.0
    img_u = img
    if i_max > 0:
        img_u = img.copy()
        img_u[img_u < umbral_frac * i_max] = 0.0
    H, W = img_u.shape
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    distancias   = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2).ravel()
    intensidades = img_u.ravel()
    total        = intensidades.sum()
    if total == 0:
        return {f: 0.0 for f in fracciones}
    orden    = np.argsort(distancias)
    dist_ord = distancias[orden]
    acum     = np.cumsum(intensidades[orden]) / total
    resultado = {}
    for f in fracciones:
        idx = np.searchsorted(acum, f)
        if idx == 0:
            resultado[f] = float(dist_ord[0])
        elif idx >= len(dist_ord):
            resultado[f] = float(dist_ord[-1])
        else:
            a0, a1 = acum[idx - 1], acum[idx]
            d0, d1 = dist_ord[idx - 1], dist_ord[idx]
            resultado[f] = float(d0 + (f - a0) / (a1 - a0 + 1e-300) * (d1 - d0))
    return resultado

def calcular_metricas_imagen(img, potencia_promedio=None):
    """
    Calcula de una sola pasada el conjunto completo de métricas
    morfológicas de UNA imagen del haz. Es la unidad de medida básica del
    análisis de polarizaciones: se aplica a cada estado por separado y
    después se comparan los resultados entre estados.

    Métricas devueltas y su significado físico:
      - `cx`,`cy`          : centroide de intensidad [px]
      - `sigma_x`,`sigma_y`: anchos por segundo momento [px]
      - `d4s_x`,`d4s_y`    : diámetros D4σ ISO 11146 (= 4σ) [px]
      - `elipticidad`      : min(D4σ)/max(D4σ) ∈ (0,1]; 1 = circular, y
                             valores bajos indican un haz elongado
      - `potencia_total`   : suma de intensidades (energía relativa)
      - `potencia_norm`    : la anterior dividida entre `potencia_promedio`
                             si se proporciona — permite comparar
                             estados entre sí en escala relativa
      - `r_enc`            : radios de energía encerrada (dict por fracción)

    Todo se reporta en píxeles; la conversión a µm ocurre al graficar.
    """
    cx, cy = calcular_centroide(img)
    sx, sy = calcular_segundo_momento(img, cx, cy)
    d4s_x = 4.0 * sx;  d4s_y = 4.0 * sy  # D4sigma ISO 11146 = 4 * sigma (segundo momento)
    elip   = min(d4s_x, d4s_y) / max(d4s_x, d4s_y) if max(d4s_x, d4s_y) > 0 else 1.0
    p_total = float(img.sum())
    p_norm  = (p_total / potencia_promedio) if potencia_promedio else None
    r_enc   = calcular_radio_energia_encerrada(img, cx, cy, FRACCIONES_ENERGIA)
    return {"potencia_total": p_total, "potencia_norm": p_norm,
            "cx": cx, "cy": cy, "sigma_x": sx, "sigma_y": sy,
            "d4s_x": d4s_x, "d4s_y": d4s_y, "elipticidad": elip, "r_enc": r_enc}

def _overlap_integral(img_a, img_b):
    """
    Coeficiente de solapamiento (tipo Bhattacharyya) entre dos perfiles de
    intensidad normalizados a energía unitaria: 1.0 si son idénticos,
    decrece hacia 0 a medida que se desplazan uno respecto al otro. Usado
    por la curva de sensibilidad a desalineación (ver más abajo) para
    estimar cuánto se degrada el acoplamiento del haz ante un
    desplazamiento lateral de d_px píxeles.
    """
    a = img_a / (img_a.sum() + 1e-300)
    b = img_b / (img_b.sum() + 1e-300)
    return float(np.sum(np.sqrt(a * b)) ** 2)

def _desplazar_imagen(img, dx, dy):
    """
    Traslada la imagen (dx, dy) píxeles rellenando con ceros el borde que
    queda descubierto. Se usa para simular una desalineación lateral
    controlada entre el haz y un receptor ideal: la pareja
    (imagen original, imagen desplazada) alimenta `_overlap_integral`
    para construir la curva de sensibilidad a desalineación.

    El relleno con 0 es físicamente correcto aquí: representa ausencia de
    señal fuera del área capturada, no un valor desconocido.
    """
    H, W = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img.astype(np.float32), M, (W, H),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float64)



def _guardar_csv(ruta: str, encabezado: list, *columnas):
    """
    Guarda columnas de datos como CSV.

    Ejemplo:
        _guardar_csv(ruta, ['t_s', 'I_norm'], tiempos, p_norm)
    """
    import csv
    filas = zip(*columnas)
    with open(ruta, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(encabezado)
        for fila in filas:
            w.writerow([f'{v:.8g}' if isinstance(v, float) else v
                        for v in fila])


def _carpeta_datos_crudos(carpeta_resultados: str) -> str:
    """
    Dada la carpeta de resultados graficados (Estabilidad_Temporal o
    Polarizaciones, dentro de <Analisis>/), retorna la carpeta HERMANA
    de datos numéricos en crudo <Analisis>/Datos_Crudos/<misma_subcarpeta>/
    — separada de las gráficas/videos normales, para que los números
    puedan reutilizarse en otro software o para rehacer figuras
    manualmente. La crea si no existe.
    """
    carpeta = os.path.join(os.path.dirname(carpeta_resultados),
                           NOMBRE_SUBCARPETA_DATOS_CRUDOS,
                           os.path.basename(carpeta_resultados))
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def _guardar_matriz_csv(ruta: str, matriz: np.ndarray, etiquetas_fila: list = None):
    """
    Exporta un array 2D (imagen, mapa, matriz) como CSV — una fila de
    archivo por fila del array, sin graficar, para que pueda reabrirse
    en cualquier software (Excel, Origin, Python, MATLAB, ...).
    """
    import csv
    with open(ruta, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        for i, fila in enumerate(matriz):
            prefijo = [etiquetas_fila[i]] if etiquetas_fila is not None else []
            w.writerow(prefijo + [f'{v:.8g}' for v in fila])


def _guardar_resumen_txt(ruta: str, valores: dict):
    """
    Exporta un conjunto de valores escalares como 'clave: valor',
    una línea por entrada — complemento en texto plano de las curvas/
    matrices exportadas como CSV.
    """
    with open(ruta, 'w', encoding='utf-8') as fh:
        for clave, valor in valores.items():
            if isinstance(valor, float):
                fh.write(f"{clave}: {valor:.8g}\n")
            else:
                fh.write(f"{clave}: {valor}\n")


def _etiqueta_pol(nombre_archivo: str, indice: int) -> str:
    """
    Deriva la etiqueta corta de una polarización ("P01", "P02", ...) a
    partir del nombre de archivo generado por Parte1.py
    (`<Disp>_Haz_<CH>_P01.png`). Si el patrón no aparece (archivo
    renombrado a mano), cae de vuelta al índice de orden de carga, para
    que las figuras nunca queden sin etiqueta.
    """
    stem = os.path.splitext(nombre_archivo)[0]
    m = re.search(r'[Pp]([0-9]+)', stem)
    if m:
        return f"P{m.group(1)}"
    return f"P{indice}"


def _pol_tabla_metricas(metricas, nombres, carpeta_salida):
    """
    Paso 1/8 — Genera `01_tabla_metricas.png` (+ CSV hermano): tabla con
    las métricas morfológicas de CADA estado de polarización, una fila por
    imagen.

    Es la vista más directa para responder "¿cambia el haz al cambiar la
    polarización?": si D4σ, elipticidad y radios de energía encerrada se
    mantienen aproximadamente constantes entre filas, el perfil es robusto
    frente a la polarización de entrada. Variaciones grandes en el
    centroide entre filas indicarían además desplazamiento del haz, no
    solo cambio de forma.

    Recibe la lista de dicts producida por `calcular_metricas_imagen` y
    los nombres de archivo correspondientes (para etiquetar las filas).
    """
    N = len(metricas)
    u = _unidad()
    enc_headers = [f"$r_{{{int(f*100)}\\%}}$ ({u})" for f in FRACCIONES_ENERGIA]
    cabeceras = (["Imagen", "I_total", "I_norm", f"cx ({u})", f"cy ({u})",
                  f"$\\sigma_x$ ({u})", f"$\\sigma_y$ ({u})", f"D4σx ({u})", f"D4σy ({u})",
                  "Elipticidad"] + enc_headers)
    filas = []
    for i, m in enumerate(metricas):
        p_norm_str = f"{m['potencia_norm']:.4f}" if m['potencia_norm'] is not None else "—"
        enc_vals   = [f"{_to_um(m['r_enc'][f]):.2f}" for f in FRACCIONES_ENERGIA]
        filas.append([
            _etiqueta_pol(nombres[i], i),
            f"{m['potencia_total']:.0f}",
            p_norm_str,
            f"{_to_um(m['cx']):.2f}",
            f"{_to_um(m['cy']):.2f}",
            f"{_to_um(m['sigma_x']):.3f}",
            f"{_to_um(m['sigma_y']):.3f}",
            f"{_to_um(m['d4s_x']):.3f}",
            f"{_to_um(m['d4s_y']):.3f}",
            f"{m['elipticidad']:.4f}",
        ] + enc_vals)

    fig_w = max(16, 1.3 * len(cabeceras))
    fig, ax = plt.subplots(figsize=(fig_w, 0.55 * N + 2.0), facecolor="white")
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=cabeceras, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False);  tabla.set_fontsize(8.5);  tabla.scale(1, 1.5)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor("#aaaaaa")
        if fila == 0:
            celda.set_facecolor("#1565C0")
            celda.set_text_props(color="white", fontweight="bold")
        else:
            celda.set_facecolor("#f5f5f5" if fila % 2 == 0 else "white")
            celda.set_text_props(color="#111111")
    ax.set_title("Metricas morfologicas por estado de polarizacion",
                 fontsize=13, fontweight="bold", color="black", pad=15)
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "01_tabla_metricas.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    cabeceras_csv = (["imagen", "I_total", "I_norm", f"cx_{u}", f"cy_{u}",
                      f"sigma_x_{u}", f"sigma_y_{u}", f"D4sx_{u}", f"D4sy_{u}",
                      "elipticidad"] +
                     [f"r_{int(f*100)}pct_{u}" for f in FRACCIONES_ENERGIA])
    ruta_tabla_csv = os.path.join(_carpeta_datos_crudos(carpeta_salida), "01_tabla_metricas_datos.csv")
    import csv as _csv1
    with open(ruta_tabla_csv, "w", newline="", encoding="utf-8") as _fh1:
        _w1 = _csv1.writer(_fh1)
        _w1.writerow(cabeceras_csv)
        _w1.writerows(filas)
    print("  Guardada: 01_tabla_metricas.png")


def _pol_imagen_promedio_std(imagenes, carpeta_salida):
    """
    Paso 2/8 — Genera `02a_imagen_promedio.png` y `02b_imagen_std.png`
    (+ CSV de ambas matrices): promedio y desviación estándar píxel a
    píxel calculados a lo largo del eje de polarizaciones.

    Interpretación física de cada panel:
      - La imagen PROMEDIO representa el perfil "típico" del haz,
        independiente del estado de polarización concreto. Es la que se
        usa después como entrada del análisis radial y de las métricas
        FSO, porque promedia el ruido de captura de N imágenes.
      - La imagen de DESVIACIÓN ESTÁNDAR muestra DÓNDE cambia el haz al
        variar la polarización. Un mapa casi uniforme y de valor bajo
        indica un perfil robusto; zonas brillantes localizadas señalan
        exactamente qué región del patrón es sensible a la polarización.

    Retorna (media, std, variabilidad_global), donde `variabilidad_global`
    = ⟨σ⟩/⟨I⟩ es un único escalar adimensional que resume el segundo
    panel y se reutiliza en la tabla de métricas FSO.
    """
    plt.close("all")
    stack = np.stack(imagenes, axis=0)
    # ddof=1 (std muestral, correccion de Bessel): con N pequeno (numero de
    # polarizaciones capturadas) el std poblacional (ddof=0) subestima la
    # variabilidad real -- convencion estandar para reportar incertidumbre
    # de N repeticiones experimentales.
    media = stack.mean(axis=0);  std = stack.std(axis=0, ddof=1)
    variabilidad_global = float(std.mean() / (media.mean() + 1e-300))
    u = _unidad()

    def _guardar_panel(data, titulo, cmap, cbar_label, nombre_archivo):
        """Renderiza y guarda un mapa 2D normalizado con colorbar y ejes en
        unidades físicas. Local a esta función porque los dos paneles
        (promedio y std) comparten exactamente el mismo formato."""
        data_norm = normalizar_frame(data)
        H, W      = data_norm.shape
        panel_h   = 7.0
        panel_w   = max(4.0, panel_h * W / H)
        fig, ax   = plt.subplots(figsize=(panel_w + 1.4, panel_h + 1.0), facecolor="white")
        ax.set_facecolor("white")
        im = ax.imshow(data_norm, cmap=cmap, origin="upper",
                       aspect="auto", vmin=0, vmax=1, extent=_extent(H, W))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=12, color="black")
        cbar.ax.yaxis.set_tick_params(color="black")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")
        ax.set_title(titulo, fontsize=12, fontweight="bold", color="black")
        ax.set_xlabel(f"x ({u})", fontsize=12, color="black")
        ax.set_ylabel(f"y ({u})", fontsize=12, color="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaaaaa")
        _agregar_marca_pie(fig)
        fig.savefig(os.path.join(carpeta_salida, nombre_archivo),
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Guardada: {nombre_archivo}")

    _guardar_panel(media, "Distribución de intensidad promedio del haz sobre diferentes polarizaciones",
                   "inferno", "Intensidad normalizada [0-1]",
                   "02a_imagen_promedio.png")
    plt.close("all")
    _guardar_panel(std, "Distribución de la desviación estándar del haz sobre diferentes polarizaciones",
                   "hot", "$\\sigma$(x,y) / $\\sigma_{max}$  [desviación estándar relativa]",
                   "02b_imagen_std.png")
    carpeta_datos2 = _carpeta_datos_crudos(carpeta_salida)
    _guardar_matriz_csv(os.path.join(carpeta_datos2, "02a_imagen_promedio_datos.csv"), media)
    _guardar_matriz_csv(os.path.join(carpeta_datos2, "02b_imagen_std_datos.csv"), std)
    _guardar_resumen_txt(os.path.join(carpeta_datos2, "02_resumen.txt"),
                         {"variabilidad_global": variabilidad_global})
    print(f"  Variabilidad global normalizada = {variabilidad_global:.5f}")
    return media, std, variabilidad_global


def _pol_mapa_varianza(imagenes, carpeta_salida):
    """
    Paso 3/8 — Genera `03_mapa_varianza.png` (+ CSV): mapa espacial de la
    varianza píxel a píxel entre los distintos estados de polarización.

    Es el mismo dato que el panel 02b elevado al cuadrado, pero se genera
    como figura independiente porque la varianza ACENTÚA los contrastes:
    las regiones sensibles a la polarización destacan mucho más que en el
    mapa de desviación estándar, lo que facilita identificar visualmente
    si la sensibilidad está localizada (p. ej. en un lóbulo concreto del
    patrón) o distribuida por todo el haz.
    """
    plt.close("all")
    stack    = np.stack(imagenes, axis=0)
    varianza = stack.var(axis=0, ddof=1)  # ddof=1: ver nota en _pol_imagen_promedio_std
    var_norm = normalizar_frame(varianza)
    u        = _unidad()
    H, W     = var_norm.shape
    panel_h  = 7.0
    panel_w  = max(4.0, panel_h * W / H)
    fig, ax  = plt.subplots(figsize=(panel_w + 1.4, panel_h + 1.0), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(var_norm, cmap="plasma", origin="upper",
                   aspect="auto", vmin=0, vmax=1, extent=_extent(H, W))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\sigma^2$(x,y) / $\\sigma^2_{max}$  [varianza relativa]", fontsize=12, color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")
    ax.set_title("Distribución de la varianza del haz sobre diferentes polarizaciones ($\\sigma^2$)",
                 fontsize=12, fontweight="bold", color="black")
    ax.set_xlabel(f"x ({u})", fontsize=12, color="black")
    ax.set_ylabel(f"y ({u})", fontsize=12, color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "03_mapa_varianza.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _guardar_matriz_csv(os.path.join(_carpeta_datos_crudos(carpeta_salida),
                                     "03_mapa_varianza_datos.csv"),
                        varianza)
    print("  Guardada: 03_mapa_varianza.png")


def _pol_matriz_correlacion(imagenes, nombres, carpeta_salida):
    """
    Paso 4/8 — Genera `04_matriz_correlacion.png` (+ CSV): matriz N×N con
    la correlación de Pearson entre CADA par de estados de polarización.

    Es la métrica más directa de "robustez estructural" del haz. La
    diagonal vale 1 por construcción; lo informativo es el promedio de
    los elementos FUERA de la diagonal (`C_off`, anotado en el título y
    reutilizado después en la tabla de métricas FSO): un valor cercano a
    1 significa que todos los estados producen esencialmente el mismo
    patrón espacial, es decir, un haz insensible a la polarización de
    entrada. Valores bajos aislados identifican qué estado concreto se
    desvía del resto.

    Retorna la matriz `C` completa, que consume después `_pol_metricas_fso`.

    ⚠ Comparte la limitación metodológica de la correlación temporal (ver
    `graficar_correlacion_espacial`): al no registrar/alinear las
    imágenes antes de correlacionar, un simple desplazamiento lateral del
    haz entre estados reduce C aunque la FORMA no haya cambiado.
    """
    # Ver limitacion metodologica documentada en graficar_correlacion_espacial
    # (correlacion de Pearson sin registrar imagenes -- mezcla desplazamiento
    # de posicion con cambio de forma) -- aplica igual aqui, entre pares de
    # imagenes de distintas polarizaciones en vez de frames en el tiempo.
    N = len(imagenes)
    C = np.zeros((N, N))
    plt.close("all")
    for i in range(N):
        for j in range(N):
            C[i, j] = np.corrcoef(imagenes[i].flatten(), imagenes[j].flatten())[0, 1]
    mask_off = ~np.eye(N, dtype=bool)
    c_off    = C[mask_off]
    c_prom   = float(c_off.mean());  c_min = float(c_off.min());  c_max = float(c_off.max())

    labels_cortos = [_etiqueta_pol(n, i) for i, n in enumerate(nombres)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(C, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlacion de Pearson", fontsize=11)
    ax.set_xticks(range(N));  ax.set_yticks(range(N))
    ax.set_xticklabels(labels_cortos, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(labels_cortos, fontsize=12)
    for i in range(N):
        for j in range(N):
            color_txt = "black" if C[i, j] > -0.5 else "white"
            ax.text(j, i, f"{C[i,j]:.3f}", ha="center", va="center",
                    fontsize=6.5, color=color_txt)
    ax.set_title(
        f"Matriz de correlacion cruzada entre estados de polarizacion (N={N})\n"
        f"$\\bar{{C}}_{{off}}$ = {c_prom:.4f}  |  $C_{{min}}$ = {c_min:.4f}  |  $C_{{max}}$ = {c_max:.4f}",
        fontsize=11, fontweight="bold")
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "04_matriz_correlacion.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    import csv as _csv4
    with open(os.path.join(_carpeta_datos_crudos(carpeta_salida), "04_matriz_correlacion.csv"),
              "w", newline="", encoding="utf-8") as _fh4:
        _w4 = _csv4.writer(_fh4)
        _w4.writerow([""] + labels_cortos)
        for _i4, _lbl in enumerate(labels_cortos):
            _w4.writerow([_lbl] + [f"{v:.6f}" for v in C[_i4]])
    print(f"  Guardada: 04_matriz_correlacion.png  |  C_off = {c_prom:.4f}  |  C_min = {c_min:.4f}  |  C_max = {c_max:.4f}")
    return C


def _pol_desplazamiento_centroide(imagenes, nombres, carpeta_salida):
    """
    Paso 5/8 — Genera `05_desplazamiento_centroide.png` (+ CSV y resumen):
    mapa de dispersión del centroide de cada estado de polarización,
    tomando el PRIMER estado (P01) como origen de coordenadas.

    Separa el efecto "cambio de posición" del efecto "cambio de forma"
    que la matriz de correlación mezcla: si los puntos están agrupados
    cerca del origen, el haz no se desplaza al cambiar la polarización y
    cualquier caída de correlación observada en el paso 4 se debe a
    deformación real del patrón, no a traslación.

    Retorna el RMS del desplazamiento (escalar, en µm o px), que se
    reutiliza en la tabla de métricas FSO como indicador resumido de
    estabilidad posicional frente a la polarización.
    """
    cx0, cy0 = calcular_centroide(imagenes[0])
    dx = _to_um(np.array([calcular_centroide(img)[0] - cx0 for img in imagenes]))
    dy = _to_um(np.array([calcular_centroide(img)[1] - cy0 for img in imagenes]))
    rms = float(np.sqrt(np.mean(dx**2 + dy**2)))
    lim = max(np.abs(np.concatenate([dx, dy])).max(), 1e-6) * 1.25
    u   = _unidad()
    N   = len(imagenes)
    # Misma paleta discreta (tab10) que _pol_curva_energia_encerrada, para
    # que cada polarización tenga un color fijo identificable por leyenda
    # en vez de etiquetas de texto junto a cada punto (con muchos puntos
    # cercanos entre sí, las etiquetas de texto se vuelven confusas aunque
    # se separen automáticamente).
    colores = plt.cm.tab10(np.linspace(0, 1, max(N, 10)))[:N]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(N):
        ax.scatter(dx[i], dy[i], color=colores[i], s=80, zorder=3,
                  edgecolors="#333333", linewidths=0.5,
                  label=_etiqueta_pol(nombres[i], i))
    ax.scatter(0, 0, color="red", s=180, zorder=4, marker="*",
               label=f"Referencia ({_etiqueta_pol(nombres[0], 0)})")
    ax.annotate(f"RMS = {rms:.4f} {u}", xy=(0.04, 0.96), xycoords="axes fraction",
                fontsize=11, fontweight="bold", color="#111111", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#333333", alpha=0.9))
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color="#aaaaaa", linewidth=0.8);  ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_title("Desplazamiento del centroide respecto a la primera polarización",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"$\\Delta x$ ({u})", fontsize=11);  ax.set_ylabel(f"$\\Delta y$ ({u})", fontsize=11)
    # Leyenda fuera del área de datos (columna a la derecha) -- con el
    # scatter simétrico alrededor del origen no hay una esquina interior
    # que esté garantizada libre de puntos para todos los casos.
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, borderaxespad=0.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "05_desplazamiento_centroide.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    carpeta_datos_pol5 = _carpeta_datos_crudos(carpeta_salida)
    _guardar_csv(os.path.join(carpeta_datos_pol5, "05_desplazamiento_centroide.csv"),
                 ["etiqueta", f"dx_{u}", f"dy_{u}"],
                 [_etiqueta_pol(n, i) for i, n in enumerate(nombres)],
                 dx.tolist(), dy.tolist())
    _guardar_resumen_txt(os.path.join(carpeta_datos_pol5, "05_desplazamiento_centroide_resumen.txt"),
                         {"rms": rms, "unidad": u})
    print(f"  Guardada: 05_desplazamiento_centroide.png  |  RMS = {rms:.4f} {u}")
    return rms


def _pol_curva_energia_encerrada(imagenes, nombres, metricas, carpeta_salida):
    """
    Paso 6/8 — Genera `06_curvas_energia_encerrada.png` (+ un CSV por
    polarización): curva de energía acumulada E(r)/E_total en función del
    radio desde el centroide, una por estado de polarización.

    Qué describe físicamente: cómo se distribuye la potencia del haz
    desde el centro hacia afuera. Es la forma estándar de comparar
    "concentración" de haces con perfiles distintos sin asumir que son
    gaussianos — algo esencial aquí, porque los haces MMI estructurados
    tienen anillos y lóbulos para los que un ancho gaussiano no es
    representativo.

    Se generan dos versiones del eje radial:
      - ABSOLUTA (µm): permite comparar tamaños reales entre estados.
      - NORMALIZADA por r₈₆ de cada curva: elimina la diferencia de
        tamaño y deja ver únicamente diferencias de FORMA del perfil
        (r₈₆ ≈ el radio que contiene el 86% de la energía, equivalente
        al criterio 1/e² para un haz gaussiano).
    """
    N = len(imagenes)
    colores = plt.cm.tab10(np.linspace(0, 1, max(N, 10)))[:N]
    plt.close("all")

    curvas = [];  r_max_global = 0.0
    for img in imagenes:
        cx, cy = calcular_centroide(img)
        H, W   = img.shape
        xs = np.arange(W, dtype=np.float64);  ys = np.arange(H, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys)
        distancias   = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2).ravel()
        intensidades = img.ravel()
        total        = intensidades.sum()
        orden        = np.argsort(distancias)
        dist_ord     = distancias[orden]
        acum         = np.cumsum(intensidades[orden]) / (total + 1e-300)
        curvas.append((dist_ord, acum))
        r_max_global = max(r_max_global, float(dist_ord[-1]))

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(wspace=0.35)
    u_lbl = _unidad();  f_sc = _factor()

    for ax in (ax_abs, ax_norm):
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"Radio desde el centroide ({u_lbl})", fontsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor("#888888")

    ax_abs.set_ylabel("Fraccion de energia encerrada", fontsize=12)
    ax_abs.set_title("Curvas de energía encerrada — absolutas", fontsize=11, fontweight="bold")
    ax_abs.set_xlim(0, r_max_global * f_sc * 0.75);  ax_abs.set_ylim(0, 1.05)

    ax_norm.set_ylabel("Fraccion de energia encerrada", fontsize=12)
    ax_norm.set_title("Curvas normalizadas por r_86\n(comparacion de forma)",
                      fontsize=11, fontweight="bold")
    ax_norm.set_xlim(0, 3.5);  ax_norm.set_ylim(0, 1.05)
    ax_norm.set_xlabel("Radio / r_86", fontsize=12)

    for f in FRACCIONES_ENERGIA:
        lbl = f"{int(f*100)}%"
        ax_abs.axhline(f, color="gray", linestyle=":", linewidth=1.0,
                       label=f"f = {lbl}" if f == FRACCIONES_ENERGIA[0] else "_")
        ax_norm.axhline(f, color="gray", linestyle=":", linewidth=1.0)
        ax_abs.text(r_max_global * f_sc * 0.76, f + 0.01, lbl, fontsize=11, color="gray", va="bottom")
        ax_norm.text(3.52, f + 0.01, lbl, fontsize=11, color="gray", va="bottom")

    ax_norm.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_norm.text(1.02, 0.05, "r_86", fontsize=12, color="gray")

    for i, ((dist_ord, acum), m, nombre) in enumerate(zip(curvas, metricas, nombres)):
        color = colores[i]
        etiq  = _etiqueta_pol(nombre, i)

        step    = max(1, len(dist_ord) // 300)
        r_sub   = dist_ord[::step]
        a_sub   = acum[::step]

        ax_abs.plot(r_sub * f_sc, a_sub, color=color, linewidth=1.4, label=etiq)
        for f in FRACCIONES_ENERGIA:
            ax_abs.plot(m["r_enc"][f] * f_sc, f, "o", color=color, markersize=4, zorder=5)

        f_86 = min(FRACCIONES_ENERGIA, key=lambda x: abs(x - 0.86))
        r_86 = m["r_enc"][f_86]
        if r_86 > 0:
            r_norm = r_sub / r_86
            mask   = r_norm <= 3.6
            ax_norm.plot(r_norm[mask], a_sub[mask], color=color, linewidth=1.4, label=etiq)
            for f in FRACCIONES_ENERGIA:
                ax_norm.plot(m["r_enc"][f] / r_86, f, "o", color=color, markersize=4, zorder=5)

    ax_abs.legend(fontsize=7.5, loc="lower right", framealpha=0.85, ncol=max(1, N // 6))
    ax_norm.legend(fontsize=7.5, loc="lower right", framealpha=0.85, ncol=max(1, N // 6))
    fig.suptitle(
        "Energia encerrada en función del radio  —  estados de polarizacion\n"
        f"Fracciones de referencia: {', '.join([str(int(f*100))+'%' for f in FRACCIONES_ENERGIA])}",
        fontsize=12, fontweight="bold")
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "06_curvas_energia_encerrada.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Exportar radios de energia encerrada por polarizacion (dato compacto
    # y util; la curva completa tiene miles de puntos por polarizacion).
    _encabezado6 = ["etiqueta"] + [f"r_{int(f*100)}%_{u_lbl}"
                                   for f in FRACCIONES_ENERGIA]
    _filas6 = []
    for i, (m, nombre) in enumerate(zip(metricas, nombres)):
        etiq = _etiqueta_pol(nombre, i)
        _filas6.append([etiq] + [m["r_enc"][f] * f_sc for f in FRACCIONES_ENERGIA])
    carpeta_datos_pol6 = _carpeta_datos_crudos(carpeta_salida)
    _guardar_csv(os.path.join(carpeta_datos_pol6, "06_curvas_energia_encerrada.csv"),
                _encabezado6,
                [f[0] for f in _filas6],
                *[[f[k] for f in _filas6] for k in range(1, len(_encabezado6))])
    # Curva completa por imagen (miles de puntos) -- un CSV por imagen, ya
    # que cada curva tiene una longitud distinta y no comparten un eje común.
    for i, ((dist_ord, acum), nombre) in enumerate(zip(curvas, nombres)):
        etiq = _etiqueta_pol(nombre, i)
        _guardar_csv(os.path.join(carpeta_datos_pol6, f"06_curva_energia_encerrada_{etiq}.csv"),
                    [f"radio_{u_lbl}", "energia_acumulada"],
                    (dist_ord * f_sc).tolist(), acum.tolist())
    print("  Guardada: 06_curvas_energia_encerrada.png")


def _pol_metricas_fso(imagenes, carpeta_salida, C, variabilidad_global, rms_centroide,
                       flags: dict = None):
    """
    Genera tres figuras FSO independientes y retorna el dict de métricas.

    07_imagen_promedio_fso.png  — imagen promedio con elipses σ y parámetros
    08_tabla_metricas_fso.png   — tabla de métricas FSO (standalone)
    09_sensibilidad_desalineacion.png — curva η vs desplazamiento (automática)
    """
    if flags is None:
        flags = {}
    _f = lambda k: flags.get(k, True)

    stack  = np.stack(imagenes, axis=0).mean(axis=0)
    cx, cy = calcular_centroide(stack)
    sx, sy = calcular_segundo_momento(stack, cx, cy)
    area_efectiva = 4.0 * np.pi * sx * sy

    f_s = _factor()
    u   = _unidad()

    N        = len(imagenes)
    mask_off = ~np.eye(N, dtype=bool)
    robustez = float(C[mask_off].mean())
    umbral_e2 = np.exp(-2)

    # ── Curva de overlap automática ────────────────────────────────────────────
    # Avanza de 1 px en 1 px hasta que las TRES curvas caen por debajo de THRESH.
    # THRESH bajo (0.0001) garantiza que todas lleguen visualmente a cero.
    THRESH    = 0.0001
    H_im, W_im = stack.shape
    # El límite debe ser la dimensión completa de la imagen, no la mitad.
    # La curva diagonal decae ~√2 veces más rápido que x o y por solas,
    # así que x e y necesitan más píxeles de recorrido para llegar a cero.
    max_d_px  = max(H_im, W_im)

    desplaz_px_list = []
    overlaps_x, overlaps_y, overlaps_diag = [], [], []

    for d_px in range(0, max_d_px + 1):
        ox = _overlap_integral(stack, _desplazar_imagen(stack, d_px, 0))
        oy = _overlap_integral(stack, _desplazar_imagen(stack, 0, d_px))
        od = _overlap_integral(stack, _desplazar_imagen(stack, d_px, d_px))
        desplaz_px_list.append(d_px)
        overlaps_x.append(ox)
        overlaps_y.append(oy)
        overlaps_diag.append(od)
        if d_px > 5 and max(ox, oy, od) < THRESH:
            break

    desplaz_um = np.array(desplaz_px_list, dtype=float) * f_s
    ovx = np.array(overlaps_x)
    ovy = np.array(overlaps_y)
    ovd = np.array(overlaps_diag)

    # Encontrar desplazamientos característicos en la curva X (interpolación)
    def _find_crossing(despl, ov, threshold):
        """Retorna el desplazamiento (en um) donde ov cae por debajo de threshold."""
        idx = np.where(ov <= threshold)[0]
        if len(idx) == 0:
            return despl[-1]
        i = idx[0]
        if i == 0:
            return despl[0]
        # Interpolación lineal entre i-1 e i
        x0, y0 = despl[i-1], ov[i-1]
        x1, y1 = despl[i],   ov[i]
        if abs(y1 - y0) < 1e-12:
            return x0
        return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)

    # Desplazamientos característicos sobre la curva DIAGONAL (más exigente)
    d_half  = _find_crossing(desplaz_um, ovd, 0.50)
    d_e2    = _find_crossing(desplaz_um, ovd, umbral_e2)
    d_zero  = _find_crossing(desplaz_um, ovd, 0.01)

    # ── FIGURA 07: imagen promedio con elipses ─────────────────────────────────
    plt.close("all")
    img_norm = normalizar_frame(stack)
    H_s, W_s = img_norm.shape
    dpi      = 300  # consistente con el resto de figuras del modulo (rcParams["savefig.dpi"])
    img_w_in = 7.0
    img_h_in = max(3.5, min(img_w_in * H_s / W_s, 9.0))

    fig7, ax7 = plt.subplots(figsize=(img_w_in + 1.4, img_h_in + 0.8),
                             facecolor="white")
    im_plot = ax7.imshow(img_norm, cmap="inferno", origin="upper",
                         aspect="equal", vmin=0, vmax=1,
                         extent=_extent(H_s, W_s))
    ax7.plot(cx * f_s, cy * f_s, "r+", markersize=14, markeredgewidth=2)
    theta_t = np.linspace(0, 2 * np.pi, 300)
    ax7.plot((cx + sx * np.cos(theta_t)) * f_s, (cy + sy * np.sin(theta_t)) * f_s,
             "cyan", lw=1.5, ls="--", label="diámetro: 2σ")
    ax7.plot((cx + 2 * sx * np.cos(theta_t)) * f_s, (cy + 2 * sy * np.sin(theta_t)) * f_s,
             "lime", lw=1.2, ls=":", label="diámetro: 4σ")
    ax7.legend(fontsize=11, facecolor="white", labelcolor="black", framealpha=0.85)
    ax7.set_title(
        f"Imagen promedio (polarizaciones)\n"
        f"Área efectiva = {area_efectiva * f_s**2:.1f} {u}²   "
        f"D4σx = {4*sx * f_s:.2f} {u}   D4σy = {4*sy * f_s:.2f} {u}",
        fontsize=12, fontweight="bold", color="black")
    ax7.set_xlabel(f"x ({u})", fontsize=11, color="black")
    ax7.set_ylabel(f"y ({u})", fontsize=11, color="black")
    ax7.tick_params(colors="black")
    for sp in ax7.spines.values():
        sp.set_edgecolor("#aaaaaa")
    cbar7 = fig7.colorbar(im_plot, ax=ax7, fraction=0.046, pad=0.04)
    cbar7.set_label("Intens. norm.", fontsize=12, color="black")
    cbar7.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar7.ax.yaxis.get_ticklabels(), color="black")
    fig7.tight_layout()
    if flags.get("generar_pol_img_fso", True):
     _agregar_marca_pie(fig7)
     fig7.savefig(os.path.join(carpeta_salida, "07_imagen_promedio_fso.png"),
                 dpi=dpi, bbox_inches="tight", facecolor="white")
     carpeta_datos7 = _carpeta_datos_crudos(carpeta_salida)
     _guardar_matriz_csv(os.path.join(carpeta_datos7, "07_imagen_promedio_fso_datos.csv"), stack)
     _guardar_resumen_txt(os.path.join(carpeta_datos7, "07_imagen_promedio_fso_resumen.txt"),
                          {"cx_px": cx, "cy_px": cy, "sx_px": sx, "sy_px": sy,
                           "area_efectiva": area_efectiva})
    plt.close(fig7)
    print("  Guardada: 07_imagen_promedio_fso.png")

    # ── FIGURA 08: tabla de métricas FSO ──────────────────────────────────────
    plt.close("all")
    metricas_fso_tabla = [
        ["4π·σx·σy  (área efectiva D4σ)",          f"{area_efectiva * f_s**2:.2f} {u}²"],
        ["σx  (imagen promedio)",                  f"{sx * f_s:.4f} {u}"],
        ["σy  (imagen promedio)",                  f"{sy * f_s:.4f} {u}"],
        ["D4σx = 4σx  (diámetro del haz)",          f"{4 * sx * f_s:.4f} {u}"],
        ["D4σy = 4σy  (diámetro del haz)",          f"{4 * sy * f_s:.4f} {u}"],
        ["Elipticidad del haz promedio",           f"{min(4*sx, 4*sy) / max(4*sx, 4*sy):.5f}"],
        ["Robustez estructural  (C̄ off-diag.)",    f"{robustez:.6f}"],
        ["Variabilidad global norm.  (σ̄ / Ī)",     f"{variabilidad_global:.6f}"],
        ["RMS desplazamiento centroide (pol.)",    f"{rms_centroide:.4f} {u}"],
        ["Despl. diagonal  →  η = 0.50",             f"{d_half:.2f} {u}"],
        ["Despl. diagonal  →  η = 1/e² ≈ 0.135",    f"{d_e2:.2f} {u}"],
        ["Despl. diagonal  →  η ≈ 0  (< 1 %)",      f"{d_zero:.2f} {u}"],
    ]
    n_rows = len(metricas_fso_tabla)
    fig8, ax8 = plt.subplots(figsize=(9, 0.55 * n_rows + 1.2), facecolor="white")
    ax8.axis("off")
    ax8.set_title("Resumen de métricas FSO",
                  fontsize=12, fontweight="bold", pad=12, color="black")
    tbl = ax8.table(cellText=metricas_fso_tabla,
                    colLabels=["Métrica FSO", "Valor"],
                    loc="center", cellLoc="center",
                    colWidths=[0.68, 0.32])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.7)
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if row_i == 0:
            cell.set_facecolor("#2e4057")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f4f8" if row_i % 2 == 0 else "#dde6f0")
            cell.set_text_props(color="#111111")
    if _f("generar_pol_tabla_fso"):
        fig8.tight_layout()
        _agregar_marca_pie(fig8)
        fig8.savefig(os.path.join(carpeta_salida, "08_tabla_metricas_fso.png"),
                     dpi=300, bbox_inches="tight", facecolor="white")
        import csv as _csv8
        with open(os.path.join(_carpeta_datos_crudos(carpeta_salida),
                               "08_tabla_metricas_fso_datos.csv"),
                  "w", newline="", encoding="utf-8") as _fh8:
            _w8 = _csv8.writer(_fh8)
            _w8.writerow(["Metrica FSO", "Valor"])
            _w8.writerows(metricas_fso_tabla)
        print("  Guardada: 08_tabla_metricas_fso.png")
    plt.close(fig8)

    # ── FIGURA 09: curva de sensibilidad a desalineación (automática) ─────────
    plt.close("all")
    fig9, ax9 = plt.subplots(figsize=(10, 6), facecolor="white")

    ax9.plot(desplaz_um, ovx, "-",  color="royalblue",  lw=2.0,
             label="Despl. en x")
    ax9.plot(desplaz_um, ovy, "-",  color="firebrick",   lw=2.0,
             label="Despl. en y")
    ax9.plot(desplaz_um, ovd, "-",  color="darkorange",  lw=2.0,
             label="Despl. diagonal  (x = y)")

    # Líneas de referencia horizontales
    for lvl, lbl, col in [
        (0.50,     "η = 0.50",          "#888888"),
        (umbral_e2, f"1/e² ≈ {umbral_e2:.3f}", "#555555"),
        (0.01,     "η = 0.01",          "#aaaaaa"),
    ]:
        ax9.axhline(lvl, color=col, lw=1.0, ls=":", alpha=0.8, label=lbl)

    # Anotación de los desplazamientos característicos sobre la curva diagonal
    for d_val, eta_val, lbl_txt in [
        (d_half, 0.50,      f"Δd₁/₂ = {d_half:.1f} {u}"),
        (d_e2,   umbral_e2, f"Δd_1/e² = {d_e2:.1f} {u}"),
        (d_zero, 0.01,      f"Δd₀ = {d_zero:.1f} {u}"),
    ]:
        ax9.annotate(
            lbl_txt,
            xy=(d_val, eta_val), xytext=(d_val + desplaz_um[-1] * 0.03, eta_val + 0.05),
            fontsize=8.5, color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.9))

    ax9.set_xlabel(f"Desplazamiento lateral ({u})", fontsize=12)
    ax9.set_ylabel("Integral de superposición  η", fontsize=12)
    ax9.set_title(
        "Sensibilidad a desalineación lateral\n"
        "(imagen promedio vs imagen desplazada — curva automática hasta η ≈ 0)",
        fontsize=11, fontweight="bold")
    ax9.legend(fontsize=12, framealpha=0.9)
    ax9.grid(True, ls="--", alpha=0.5)
    ax9.set_ylim(-0.03, 1.05)
    ax9.set_xlim(0, desplaz_um[-1] * 1.02)
    if _f("generar_pol_sensibilidad"):
        fig9.tight_layout()
        _agregar_marca_pie(fig9)
        fig9.savefig(os.path.join(carpeta_salida, "09_sensibilidad_desalineacion.png"),
                     dpi=300, bbox_inches="tight", facecolor="white")
        print("  Guardada: 09_sensibilidad_desalineacion.png")
    plt.close(fig9)
    carpeta_datos_pol9 = _carpeta_datos_crudos(carpeta_salida)
    _guardar_csv(os.path.join(carpeta_datos_pol9, "09_sensibilidad_desalineacion.csv"),
                 [f"desplaz_{u}", "eta_x", "eta_y", "eta_diagonal"],
                 list(desplaz_um), list(ovx), list(ovy), list(ovd))
    _guardar_resumen_txt(os.path.join(carpeta_datos_pol9, "09_sensibilidad_desalineacion_resumen.txt"),
                         {"d_half": d_half, "d_e2": d_e2, "d_zero": d_zero, "unidad": u})

    return {
        "area_efectiva":      area_efectiva,
        "sigma_x_prom":       sx,
        "sigma_y_prom":       sy,
        "robustez":           robustez,
        "variabilidad_global": variabilidad_global,
        "rms_centroide":      rms_centroide,
        "overlaps_x":         list(ovx),
        "overlaps_y":         list(ovy),
        "overlaps_diag":      list(ovd),
        "desplaz_um":         list(desplaz_um),
        "d_half":             d_half,
        "d_e2":               d_e2,
        "d_zero":             d_zero,
    }


def _pol_tabla_resumen_numerica(metricas_por_img, nombres, C, metricas_fso, carpeta_salida):
    """
    Paso 8/8 — Genera `10_resultados_numericos.txt`: consolidación en
    texto plano de TODO el análisis de polarizaciones (métricas por
    imagen, estadísticos de la matriz de correlación y métricas FSO) en
    un único archivo legible.

    Existe como complemento —no sustituto— de los CSV de `Datos_Crudos/`:
    los CSV están pensados para reprocesar en otro software, mientras que
    este .txt está pensado para leerse directamente o pegarse como anexo,
    con todas las cifras del análisis en un solo lugar y con sus unidades
    explícitas.

    Recibe ya calculado todo lo que reporta (no recalcula nada): las
    métricas por imagen del paso 1, la matriz `C` del paso 4 y el dict de
    métricas FSO del paso 7.
    """
    N   = len(metricas_por_img)
    sep = "─" * 66
    u   = _unidad();  f_s = _factor()

    lineas = [
        "=" * 70,
        "  RESULTADOS NUMERICOS — ANALISIS POLARIMETRICO",
        "=" * 70, "",
        sep, "  1. METRICAS MORFOLOGICAS POR IMAGEN", sep,
        (f"  {'#':<3}  {'Imagen':<10}  {'P_tot':>10}  {'P_norm':>7}  "
         f"{'cx':>9}  {'cy':>9}  "
         # "D4sx"/"D4sy" (no "wx_RMS"/"wy_RMS"): estas columnas contienen
         # d4s_x/d4s_y (diametro D4sigma por segundo momento, una medicion
         # directa por imagen), no un RMS de nada -- mismo prefijo "D4s"
         # que ya usa el encabezado CSV histórico (04_ancho_haz.csv:
         # D4sx_um/D4sy_um) para la misma cantidad, por consistencia.
         f"{'s_x':>9}  {'s_y':>9}  {'D4sx':>9}  {'D4sy':>9}  {'Elip.':>7}  "
         + "  ".join([f"r_{int(f*100):>2}%" for f in FRACCIONES_ENERGIA])
         + f"  [{u}]"),
        "  " + "-" * (80 + 10 * len(FRACCIONES_ENERGIA)),
    ]
    for i, m in enumerate(metricas_por_img):
        p_norm_str = f"{m['potencia_norm']:.4f}" if m['potencia_norm'] is not None else "   —  "
        enc_str    = "  ".join([f"{_to_um(m['r_enc'][f]):>8.2f}" for f in FRACCIONES_ENERGIA])
        lineas.append(
            f"  {i:<3}  {_etiqueta_pol(nombres[i], i):<10}  "
            f"{m['potencia_total']:>10.0f}  {p_norm_str:>7}  "
            f"{_to_um(m['cx']):>9.2f}  {_to_um(m['cy']):>9.2f}  "
            f"{_to_um(m['sigma_x']):>9.3f}  {_to_um(m['sigma_y']):>9.3f}  "
            f"{_to_um(m['d4s_x']):>9.3f}  {_to_um(m['d4s_y']):>9.3f}  "
            f"{m['elipticidad']:>7.4f}  {enc_str}")

    keys_espaciales = {"cx", "cy", "sigma_x", "sigma_y", "d4s_x", "d4s_y"}
    keys_num = ["potencia_total", "potencia_norm", "cx", "cy", "sigma_x",
                "sigma_y", "d4s_x", "d4s_y", "elipticidad"]
    lineas += ["", sep, f"  Estadisticos escalares (unidades espaciales en {u})", sep]
    for k in keys_num:
        vals_raw = [m[k] for m in metricas_por_img]
        if any(v is None for v in vals_raw):
            lineas.append(f"  {k:<22}  (no disponible)");  continue
        vals = np.array(vals_raw, dtype=float)
        if k in keys_espaciales:
            vals = vals * f_s
        # ddof=1 (std muestral): ver nota en _pol_imagen_promedio_std.
        lineas.append(f"  {k:<22}  media={vals.mean():.4f}  std={vals.std(ddof=1):.4f}  "
                      f"min={vals.min():.4f}  max={vals.max():.4f}")

    lineas += ["", sep, f"  Estadisticos de radios de energia encerrada ({u})", sep]
    for f in FRACCIONES_ENERGIA:
        rvals = np.array([_to_um(m["r_enc"][f]) for m in metricas_por_img])
        lineas.append(f"  r_{int(f*100):>2}%  media={rvals.mean():.3f}  std={rvals.std(ddof=1):.3f}  "
                      f"min={rvals.min():.3f}  max={rvals.max():.3f}  "
                      f"cv={rvals.std(ddof=1)/rvals.mean()*100:.2f}%")

    mask_off = ~np.eye(N, dtype=bool);  c_off = C[mask_off]
    lineas += [
        "", sep, "  2. MATRIZ DE CORRELACION CRUZADA (estadisticos off-diagonal)", sep,
        f"  C_off  = {c_off.mean():.6f}", f"  C_min  = {c_off.min():.6f}",
        f"  C_max  = {c_off.max():.6f}", f"  C_std  = {c_off.std(ddof=1):.6f}",
    ]

    f_s2 = f_s ** 2
    lineas += [
        "", sep, "  3. METRICAS FSO", sep,
        f"  Area efectiva del haz  = {metricas_fso['area_efectiva']*f_s2:.4f} {u}²",
        f"  D4σx (diámetro haz)    = {4*metricas_fso['sigma_x_prom']*f_s:.4f} {u}",
        f"  D4σy (diámetro haz)    = {4*metricas_fso['sigma_y_prom']*f_s:.4f} {u}",
        f"  Robustez estructural   = {metricas_fso['robustez']:.6f}",
        f"  Variabilidad global    = {metricas_fso['variabilidad_global']:.6f}",
        f"  RMS centroide (pol.)   = {metricas_fso['rms_centroide']:.4f} {u}",
        "",
        "  Sensibilidad a desalineación (curva automática, diagonal x=y):",
        f"    η = 0.50  →  Δd = {metricas_fso['d_half']:.2f} {u}",
        f"    η = 1/e²  →  Δd = {metricas_fso['d_e2']:.2f} {u}",
        f"    η ≈ 0     →  Δd = {metricas_fso['d_zero']:.2f} {u}",
    ]

    lineas += ["", "=" * 66, "  FIN DEL REPORTE", "=" * 66]
    ruta_txt = os.path.join(_carpeta_datos_crudos(carpeta_salida), "10_resultados_numericos.txt")
    with open(ruta_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    print("  Guardado: 10_resultados_numericos.txt")


def _pol_grafica_energia_angular(angulos: np.ndarray, energia_angular: np.ndarray,
                                  carpeta_salida: str, carpeta_datos: str):
    """
    Gráfica estática (no video) del análisis radial: energía integrada del
    perfil radial en función del ángulo (energia_angular[i] = integral de
    I(r)·r a lo largo del rayo en la dirección angulos[i], ponderada por r
    porque un anillo a radio r cubre un área proporcional a r — dA = r·dr —
    así que sumar sin ese peso subestima la contribución de la cola lejana
    del haz frente a la zona cercana al centro).

    A diferencia de usar solo el pico máximo del perfil, esta métrica
    responde directamente "¿hacia dónde se concentra la energía del haz?"
    — es robusta a un solo píxel brillante aislado (el pico máximo no lo
    es) y es la comparación natural con el resto del análisis, que ya
    trabaja con energía encerrada/acumulada en vez de intensidad puntual.

    A diferencia del video (que anima el perfil completo ángulo por
    ángulo), esta figura resume los 360 perfiles en una sola imagen — útil
    para ver de un vistazo si el haz tiene alguna dependencia angular
    (asimetría, lóbulos/deformaciones en ciertas direcciones, etc.).
    """
    e_norm = energia_angular / max(energia_angular.max(), 1e-300)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.plot(angulos, e_norm, color="#B8860B", linewidth=1.6,
             marker="o", markersize=3)
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("Ángulo θ (°)", fontsize=12)
    ax.set_ylabel("Energía angular integrada (norm.)", fontsize=12)
    ax.set_title("Energía angular integrada en función del ángulo",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _agregar_marca_pie(fig)
    fig.savefig(os.path.join(carpeta_salida, "12_energia_angular_integrada.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _guardar_csv(os.path.join(carpeta_datos, "12_energia_angular_integrada_datos.csv"),
                ["angulo_deg", "energia_angular_norm"],
                list(angulos), e_norm.tolist())
    print("  Guardada: 12_energia_angular_integrada.png")


def _pol_video_perfil_radial(imagen_promedio: np.ndarray, carpeta_salida: str,
                              generar_video: bool = True,
                              generar_grafica_angular: bool = True):
    """
    Video de análisis radial de la imagen promedio.
    360 frames (uno por grado). θ = 0° apunta a la derecha (+X) y aumenta
    en sentido de las manecillas del reloj (0°→derecha, 90°→arriba, 180°→izquierda,
    270°→abajo, referenciado al sistema de coordenadas estándar con Y hacia arriba).
    """
    from scipy.signal import find_peaks, savgol_filter

    img_norm = normalizar_frame(imagen_promedio)
    H, W     = img_norm.shape
    cx, cy   = calcular_centroide(imagen_promedio)
    f        = _factor()
    u        = _unidad()

    r_max_px  = max(5, int(min(cx, cy, W - cx, H - cy) * 0.98))
    radios_px = np.arange(r_max_px)
    radios_um = radios_px * f
    # Aplicar intervalo de ángulos según configuración
    _paso = 1 if ANALIZAR_TODOS_ANGULOS else max(1, INTERVALO_ANGULOS)
    angulos   = np.arange(0, 360, _paso)
    cmap_ang  = matplotlib.colormaps["hsv"]

    print(f"\n  Generando video de análisis radial ({len(angulos)} ángulos)...")

    # Ventana Savitzky-Golay para suavizado del perfil
    win_sg = max(5, min(21, (r_max_px // 10) * 2 + 1))  # impar, al menos 5

    def _perfil_y_maximos(theta_deg):
        """
        Extrae el perfil radial en la dirección theta_deg.

        Si ANCHO_PERFIL_RADIAL > 1, se promedian varias líneas paralelas
        centradas en el ángulo analizado. Las líneas vecinas se desplazan
        en la dirección perpendicular al radio:

            perp = (sin θ,  cos θ)  en coordenadas imagen (x→derecha, y→abajo)

        Para ancho W, los desplazamientos son k ∈ {-(W-1)/2, …, +(W-1)/2}.
        Cada línea se muestrea por interpolación bilineal y el perfil final
        es el promedio de las W líneas.
        """
        theta  = np.deg2rad(theta_deg)
        cos_t  = np.cos(theta)
        sin_t  = np.sin(theta)
        # Dirección perpendicular (rotación 90° CCW del vector radial)
        # radial: (cos θ, −sin θ) → perp: (sin θ, cos θ)
        offsets = np.arange(-(ANCHO_PERFIL_RADIAL - 1) / 2,
                             (ANCHO_PERFIL_RADIAL - 1) / 2 + 1)  # incluye 0
        acum = np.zeros(len(radios_px), dtype=np.float64)
        for k in offsets:
            xs = np.clip(cx + radios_px * cos_t + k * sin_t,  0, W - 1)
            ys = np.clip(cy - radios_px * sin_t + k * cos_t,  0, H - 1)
            xi = xs.astype(int);  xf = np.minimum(xi + 1, W - 1)
            yi = ys.astype(int);  yf = np.minimum(yi + 1, H - 1)
            dx_ = xs - xi;        dy_ = ys - yi
            acum += (img_norm[yi, xi] * (1 - dx_) * (1 - dy_) +
                     img_norm[yi, xf] * dx_       * (1 - dy_) +
                     img_norm[yf, xi] * (1 - dx_) * dy_       +
                     img_norm[yf, xf] * dx_       * dy_)
        perfil = acum / len(offsets)

        # ── Detección robusta de máximos locales ───────────────────────────────
        # 1. Suavizar para eliminar ruido de píxeles individuales
        p_s = savgol_filter(perfil, window_length=win_sg, polyorder=3)
        # 2. Saltar los primeros puntos (zona del centroide donde siempre hay máximo global)
        skip = max(2, r_max_px // 15)
        # 3. Encontrar picos en el perfil suavizado con criterios relajados:
        #    - prominence ≥ 1.5 % del rango del perfil (relativo al contexto local)
        #    - height ≥ 2 % del máximo absoluto (descartar cola plana)
        #    - distance ≥ 3 muestras (evitar dobles detecciones del mismo anillo)
        rango = max(p_s.max() - p_s.min(), 1e-9)
        pks, _ = find_peaks(p_s[skip:],
                            prominence=0.015 * rango,
                            height=0.02 * p_s.max(),
                            distance=3)
        pks = pks + skip  # corregir índices
        return perfil, pks

    # ── Pre-calcular todos los perfiles ────────────────────────────────────────
    perfiles   = []
    maximos_arr = []
    for theta_deg in angulos:
        perf, pks = _perfil_y_maximos(theta_deg)
        perfiles.append(perf)
        maximos_arr.append(pks)

    n_max_arr = np.array([len(pks) for pks in maximos_arr])
    # Energia angular integrada: integral de I(r)*r a lo largo de cada rayo
    # (peso "r" porque un anillo a radio r cubre un area proporcional a r,
    # dA = r*dr -- sin ese peso se subestima la contribucion de la cola
    # lejana del haz frente a la zona cercana al centro).
    energia_angular_arr = np.array([np.trapezoid(p * radios_px, radios_px) for p in perfiles])

    # ── Exportar datos crudos: matriz completa ángulo×radio (el dataset más
    #    grande de este archivo — sin exportar previamente) y conteo de
    #    máximos por ángulo ────────────────────────────────────────────────
    carpeta_datos11 = _carpeta_datos_crudos(carpeta_salida)
    import csv as _csv11
    with open(os.path.join(carpeta_datos11, "11_perfil_radial_matriz_angulo_radio.csv"),
              "w", newline="", encoding="utf-8") as _fh11:
        _w11 = _csv11.writer(_fh11)
        _w11.writerow(["angulo_deg\\radio_" + u] + [f"{r:.6g}" for r in radios_um])
        for theta_deg, perf in zip(angulos, perfiles):
            _w11.writerow([theta_deg] + [f"{v:.6g}" for v in perf])
    _guardar_csv(os.path.join(carpeta_datos11, "11_perfil_radial_maximos_por_angulo.csv"),
                ["angulo_deg", "n_maximos"],
                list(angulos), n_max_arr.tolist())

    if generar_grafica_angular:
        _pol_grafica_energia_angular(angulos, energia_angular_arr,
                                      carpeta_salida, carpeta_datos11)

    if not generar_video:
        return

    # ── Video ──────────────────────────────────────────────────────────────────
    ruta_out = os.path.join(carpeta_salida, "11_video_perfil_radial.mp4")
    writer   = None
    fps_vid  = 30

    for i, theta_deg in enumerate(angulos):
        if i % 30 == 0:
            print(f"    Ángulo {theta_deg}°/{angulos[-1]}°...", end="\r")

        theta    = np.deg2rad(theta_deg)
        # Se oscurece el color del colormap "hsv" antes de usarlo: sin esto,
        # la franja amarilla del ciclo de 360° queda con muy poco contraste
        # sobre los fondos claros/blancos de ambos paneles.
        color    = tuple(c * 0.6 for c in cmap_ang(theta_deg / 360.0)[:3])
        perfil   = perfiles[i]
        maximos  = maximos_arr[i]
        n_max    = n_max_arr[i]

        # Endpoint de la línea (CW: negar sin)
        x_end = cx + r_max_px * np.cos(theta)
        y_end = cy - r_max_px * np.sin(theta)   # negado → CW

        fig, (ax_img_r, ax_perfil) = plt.subplots(
            1, 2,
            figsize=(14, 5.5),
            gridspec_kw={"width_ratios": [1, 1.3]},
            facecolor="white"
        )

        # ── Panel izquierdo: imagen + líneas radiales ──────────────────────────
        ax_img_r.set_facecolor("white")
        ax_img_r.imshow(img_norm, cmap="inferno", origin="upper",
                        aspect="equal", vmin=0, vmax=1, extent=_extent(H, W))

        # Trazas fantasma de ángulos previos
        n_ghost = min(i, 30)
        step_g  = max(1, i // n_ghost) if n_ghost > 0 else 1
        for j in range(0, i, step_g):
            th_j = np.deg2rad(angulos[j])
            c_j  = tuple(c * 0.6 for c in cmap_ang(angulos[j] / 360.0)[:3])
            xe_j = cx + r_max_px * np.cos(th_j)
            ye_j = cy - r_max_px * np.sin(th_j)   # CW
            ax_img_r.plot([cx * f, xe_j * f], [cy * f, ye_j * f],
                          color=c_j, linewidth=0.5, alpha=0.20)

        # Línea actual
        ax_img_r.plot([cx * f, x_end * f], [cy * f, y_end * f],
                      color=color, linewidth=2.2, alpha=0.95)
        ax_img_r.plot(cx * f, cy * f, "r+", markersize=10, markeredgewidth=2)
        ax_img_r.set_title(f"Imagen promedio  —  θ = {theta_deg}°",
                           fontsize=12, fontweight="bold", color="#111111")
        ax_img_r.set_xlabel(f"x ({u})", fontsize=11, color="#111111")
        ax_img_r.set_ylabel(f"y ({u})", fontsize=11, color="#111111")
        ax_img_r.tick_params(colors="#333333", labelsize=9)

        # ── Panel derecho: perfil radial + máximos ─────────────────────────────
        ax_perfil.set_facecolor("#f5f5f5")

        # Todos los ángulos como contexto tenue (indices en perfiles, no grados)
        _step_ctx = max(1, len(perfiles) // 24)  # ~24 lineas de contexto
        for _j in range(0, len(perfiles), _step_ctx):
            ax_perfil.plot(radios_um, perfiles[_j], color="#AAAAAA",
                           linewidth=0.4, alpha=0.25)

        # Perfil actual
        ax_perfil.plot(radios_um, perfil, color=color, linewidth=2.5)
        ax_perfil.fill_between(radios_um, perfil, alpha=0.20, color=color)

        # Marcar máximos locales detectados
        # Color ámbar oscuro (en vez de amarillo puro) para que se vea bien
        # tanto sobre el panel claro (#f5f5f5) como sobre el recuadro oscuro
        # de la anotación de abajo — una sola paleta consistente.
        COLOR_RESALTADO_RADIAL = "#B8860B"
        if RADIAL_MOSTRAR_MAXIMOS and len(maximos) > 0:
            ax_perfil.plot(radios_um[maximos], perfil[maximos], "^",
                           color=COLOR_RESALTADO_RADIAL, markersize=8, zorder=6,
                           markeredgecolor="#5c4400", markeredgewidth=0.5,
                           label=f"Máximos: {len(maximos)}")

        # Línea 1/e²
        if RADIAL_MOSTRAR_E2:
            peak = perfil.max()
            if peak > 0:
                nivel_e2 = peak * np.exp(-2)
                cruces   = np.where(np.diff(np.sign(perfil - nivel_e2)))[0]
                if len(cruces) > 0:
                    r_e2 = radios_um[cruces[0]]
                    ax_perfil.axvline(r_e2, color=COLOR_RESALTADO_RADIAL, linestyle="--",
                                      linewidth=1.2, alpha=0.7)
                    ax_perfil.text(r_e2 + radios_um[-1] * 0.01, 0.92,
                                   f"1/e² = {r_e2:.1f} {u}",
                                   color=COLOR_RESALTADO_RADIAL, fontsize=7.5, va="top")
                ax_perfil.axhline(nivel_e2, color=COLOR_RESALTADO_RADIAL, linestyle=":",
                                  linewidth=0.8, alpha=0.45)

        # Anotación de máximos — se actualiza cada frame (no acumula)
        if RADIAL_MOSTRAR_MAXIMOS:
            ax_perfil.annotate(
                f"Máximos locales: {n_max}",
                xy=(0.97, 0.97), xycoords="axes fraction",
                fontsize=12, fontweight="bold", color=COLOR_RESALTADO_RADIAL,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="#222222",
                          ec=COLOR_RESALTADO_RADIAL, alpha=0.9))

        ax_perfil.set_xlim(0, radios_um[-1])
        ax_perfil.set_ylim(-0.03, 1.10)
        ax_perfil.set_title(f"Perfil radial  θ = {theta_deg}°",
                            fontsize=12, fontweight="bold", color="#111111")
        ax_perfil.set_xlabel(f"Radio desde centroide ({u})", fontsize=11, color="#111111")
        ax_perfil.set_ylabel("Intensidad normalizada", fontsize=11, color="#111111")
        ax_perfil.tick_params(colors="#333333", labelsize=8)
        ax_perfil.grid(True, linestyle="--", alpha=0.3)
        if len(maximos) > 0:
            ax_perfil.legend(fontsize=12, facecolor="#333333",
                             labelcolor="#111111", framealpha=0.9, edgecolor="#AAAAAA")

        fig.suptitle("Análisis radial — imagen promedio (polarizaciones)",
                     fontsize=12, fontweight="bold", color="#111111")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        _agregar_marca_pie(fig)
        frame_bgr = fig_a_frame_bgr(fig)
        if writer is None:
            ww, hh = fig.canvas.get_width_height()
            writer = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_vid, (ww, hh))
        writer.write(frame_bgr)
        plt.close(fig)

    if writer:
        writer.release()
    print("\n  Guardado: 11_video_perfil_radial.mp4")


def analizar_polarizaciones(carpeta_entrada: str, carpeta_pol_salida: str,
                             generar_video_radial: bool = True,
                             generar_grafica_angular: bool = True,
                             flags: dict = None):
    """
    Módulo de polarizaciones. Guarda todos sus resultados directamente en
    carpeta_pol_salida (ya creada y con prefijo de fecha por el caller).

    `flags` es un dict opcional con claves generar_pol_* que permiten
    deshabilitar pasos individuales del análisis de polarizaciones.
    """
    _f = flags or {}   # flags de activación individuales

    def _skip(key):
        """True → resultado deshabilitado por el usuario."""
        return not _f.get(key, True)

    print("\n" + "═" * 62)
    print("  MODULO: ANALISIS POLARIMETRICO ESPACIAL")
    print("═" * 62)

    carpeta_pol = encontrar_carpeta_polarizaciones(carpeta_entrada)
    if carpeta_pol is None:
        print("  No se encontro la carpeta 'diferentes_polarizaciones_haz'.")
        print("  Modulo polarimetrico omitido.")
        return
    print(f"\n  Carpeta de polarizaciones (entrada): {carpeta_pol}")
    print(f"  Carpeta de resultados (salida)     : {carpeta_pol_salida}")

    imagenes, nombres = cargar_imagenes_polarizacion(carpeta_pol)
    if len(imagenes) < 2:
        print("  Se necesitan al menos 2 imagenes. Modulo omitido.")
        return

    # Variables que pasos posteriores necesitan aunque el anterior esté desactivado
    C                  = None
    variabilidad_global = 0.0
    rms_centroide      = 0.0
    metricas_fso       = None
    imagen_promedio    = None

    print("\n─── [Pol 1/8] Calculando metricas por imagen ───")
    potencias_brutas  = [float(img.sum()) for img in imagenes]
    potencia_promedio = float(np.mean(potencias_brutas))
    metricas_por_img  = [calcular_metricas_imagen(img, potencia_promedio) for img in imagenes]
    if not _skip('generar_pol_tabla'):
        _pol_tabla_metricas(metricas_por_img, nombres, carpeta_pol_salida)
    else:
        print("  (omitido por configuración)")

    print("\n─── [Pol 2/8] Imagen promedio y desviacion estandar ───")
    if not _skip('generar_pol_promedio'):
        imagen_promedio, _, variabilidad_global = _pol_imagen_promedio_std(imagenes, carpeta_pol_salida)
    else:
        import numpy as _np
        imagen_promedio  = _np.mean(imagenes, axis=0)
        # ddof=1 (std muestral): ver nota en _pol_imagen_promedio_std.
        variabilidad_global = float(_np.std([img.sum() for img in imagenes], ddof=1) /
                                    max(1e-9, _np.mean([img.sum() for img in imagenes])))
        print("  (imágenes omitidas por configuración; promedio calculado internamente)")

    print("\n─── [Pol 3/8] Mapa de varianza espacial ───")
    if not _skip('generar_pol_varianza'):
        _pol_mapa_varianza(imagenes, carpeta_pol_salida)
    else:
        print("  (omitido por configuración)")

    print("\n─── [Pol 4/8] Matriz de correlacion cruzada ───")
    if not _skip('generar_pol_correlacion'):
        C = _pol_matriz_correlacion(imagenes, nombres, carpeta_pol_salida)
    else:
        import numpy as _np
        C = _np.corrcoef([img.flatten() for img in imagenes])
        print("  (imagen omitida por configuración; matriz calculada internamente)")

    print("\n─── [Pol 5/8] Desplazamiento del centroide ───")
    if not _skip('generar_pol_centroide'):
        rms_centroide = _pol_desplazamiento_centroide(imagenes, nombres, carpeta_pol_salida)
    else:
        rms_centroide = 0.0
        print("  (omitido por configuración)")

    print("\n─── [Pol 6/8] Curvas de energia encerrada ───")
    if not _skip('generar_pol_energia'):
        _pol_curva_energia_encerrada(imagenes, nombres, metricas_por_img, carpeta_pol_salida)
    else:
        print("  (omitido por configuración)")

    print("\n─── [Pol 7/8] Metricas FSO ───")
    if any(_f.get(k, True) for k in
           ['generar_pol_img_fso','generar_pol_tabla_fso','generar_pol_sensibilidad']):
        metricas_fso = _pol_metricas_fso(
            imagenes, carpeta_pol_salida, C, variabilidad_global, rms_centroide,
            flags=flags)
    else:
        print("  (omitido por configuración)")

    print("\n─── [Pol 8/8] Tabla de resultados numericos (txt) ───")
    if not _skip('generar_pol_txt') and metricas_fso is not None:
        _pol_tabla_resumen_numerica(metricas_por_img, nombres, C, metricas_fso, carpeta_pol_salida)
    elif _skip('generar_pol_txt'):
        print("  (omitido por configuración)")
    else:
        print("  (omitido: métricas FSO no calculadas)")

    if generar_video_radial or generar_grafica_angular:
        print("\n─── [Pol +] Análisis radial (imagen promedio) ───")
        _pol_video_perfil_radial(imagen_promedio, carpeta_pol_salida,
                                  generar_video=generar_video_radial,
                                  generar_grafica_angular=generar_grafica_angular)
    else:
        print("\n  (Análisis radial desactivado)")

    print(f"\n  Resultados polarimetricos en: {carpeta_pol_salida}")
    print("═" * 62)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def _ejecutar_analisis_caso(carpeta_caso: str, eleccion: str, session=None) -> bool:
    """
    Ejecuta el análisis completo (estabilidad temporal + polarizaciones)
    para UN caso (Sin turbulencia / Transitorio / Con turbulencia), leyendo
    de `<carpeta_caso>/<eleccion>` y escribiendo en `<carpeta_caso>/Analisis`.

    Lanza FileNotFoundError/ValueError si al caso le falta algo
    imprescindible (la subcarpeta `eleccion`, 'estabilidad_temporal', un
    video, o frames suficientes) — quien llama debe capturarlas para
    omitir ese caso sin abortar el resto del lote.

    Retorna True si el análisis se completó.
    """
    global TAMANO_PIXEL_UM, _MARCA_PIE

    carpeta_entrada = os.path.join(carpeta_caso, eleccion)
    if not os.path.exists(carpeta_entrada):
        raise FileNotFoundError(
            f"No existe la subcarpeta '{eleccion}' dentro de la carpeta del caso:\n"
            f"  {carpeta_caso}")

    print(f"\nCarpeta del caso     : {carpeta_caso}")
    print(f"Origen de los datos  : {eleccion}")

    # ── Marca de dispositivo/caso para pie de página de figuras/videos ────────
    _nombre_raiz = os.path.basename(os.path.dirname(carpeta_caso))
    _m_dev = re.match(r'^\d{8}_CaracterizacionHaz_(.+?)(?:_\d{2})?$', _nombre_raiz)
    _dispositivo_label = _m_dev.group(1) if _m_dev else _nombre_raiz
    _caso_key = os.path.basename(carpeta_caso)
    _caso_label_pie = utils_carpetas.CASO_LABELS.get(_caso_key, _caso_key)
    _MARCA_PIE = f"{_dispositivo_label}  |  {_caso_label_pie}"

    # ── Leer metadata de adquisicion para obtener um_per_px si no está disponible
    # (permite ejecutar la Opción 3 con una carpeta de día anterior sin recalibrar)
    _meta_adq = _cargar_metadata_adquisicion(carpeta_entrada)
    if _meta_adq and (TAMANO_PIXEL_UM is None or TAMANO_PIXEL_UM == 0.0):
        _um = (_meta_adq.get("calibracion") or {}).get("um_per_px")
        if _um:
            TAMANO_PIXEL_UM = float(_um)
            print(f"  Tamaño de píxel leído del metadata: {TAMANO_PIXEL_UM:.5f} µm/px")

    # ── Carpeta de resultados: <carpeta_caso>/Analisis ────────────────────────
    carpeta_salida = utils_carpetas.carpeta_salida_segura(
        os.path.join(carpeta_caso, utils_carpetas.NOMBRE_ANALISIS))
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"\nCarpeta de resultados: {carpeta_salida}")

    # ── Subcarpetas de resultados (sin prefijo de fecha) ──────────────────────
    carpeta_temporal = os.path.join(carpeta_salida, NOMBRE_SUBCARPETA_TEMPORAL)
    carpeta_pol_out  = os.path.join(carpeta_salida, NOMBRE_SUBCARPETA_POLARIZACION)
    os.makedirs(carpeta_temporal, exist_ok=True)
    os.makedirs(carpeta_pol_out, exist_ok=True)
    # Datos numéricos en crudo (separados de las gráficas/videos, para
    # reutilizarse en otro software o rehacer figuras manualmente)
    carpeta_datos_raiz = os.path.join(carpeta_salida, NOMBRE_SUBCARPETA_DATOS_CRUDOS)
    _carpeta_datos_crudos(carpeta_temporal)
    _carpeta_datos_crudos(carpeta_pol_out)
    print(f"  Subcarpeta temporal      : {carpeta_temporal}")
    print(f"  Subcarpeta polarizaciones: {carpeta_pol_out}")
    print(f"  Subcarpeta datos crudos  : {carpeta_datos_raiz}")

    # ── Análisis de estabilidad temporal ───────────────────────────────────
    carpeta_estabilidad = encontrar_carpeta_estabilidad(carpeta_entrada)
    if carpeta_estabilidad is None:
        raise FileNotFoundError(
            "No se encontro la carpeta 'estabilidad_temporal' dentro de:\n"
            f"  {carpeta_entrada}")
    print(f"\nCarpeta de estabilidad: {carpeta_estabilidad}")

    ruta_video = encontrar_video(carpeta_estabilidad)
    if ruta_video is None:
        raise FileNotFoundError(
            "No se encontro ningun archivo de video en:\n"
            f"  {carpeta_estabilidad}\n"
            "Extensiones soportadas: mp4, avi, mov, mkv")

    frames, fps, frame_indices = cargar_frames(ruta_video)
    if len(frames) < 2:
        raise ValueError("Se necesitan al menos 2 frames para el analisis.")

    intervalo_ef = 1 if ANALIZAR_TODOS else max(1, INTERVALO_FRAMES)
    # Use the real frame indices returned by cargar_frames (frame_indices),
    # not the enumeration index -- if any frame read fails partway through
    # the video, the enumeration index no longer matches the frame's real
    # position and every timestamp after that point would be shifted.
    if len(frame_indices) != len(frames):
        print_warn(f"Se esperaban {len(frames)} índices de frame reales, se "
                   f"obtuvieron {len(frame_indices)} -- el eje de tiempo podría "
                   "ser impreciso para este caso.")
    tiempos = np.array(frame_indices, dtype=np.float64) / fps

    # ── Referencia compartida de correlación espacial (promedio del primer
    #    segundo del caso SinTurbulencia del mismo dispositivo) ──────────────
    _caso_actual_key = os.path.basename(carpeta_caso)
    ref_correlacion, _motivo_fallback = _cargar_referencia_correlacion(
        os.path.dirname(carpeta_caso), eleccion, _caso_actual_key,
        ruta_video, frames[0].shape)
    if ref_correlacion is None:
        print_warn(
            f"Referencia de correlación espacial: {_motivo_fallback} -- se "
            "usará el frame 0 de este mismo caso como respaldo (autoconsistencia "
            "interna del video, no degradación respecto al estado sin turbulencia; "
            "menos comparable con otros casos de este dispositivo).")
        ref_correlacion_flat = None
    else:
        print_ok("Referencia de correlación espacial: promedio del primer "
                 "segundo de 'SinTurbulencia'.")
        ref_correlacion_flat = ref_correlacion.flatten()

    print_seccion("Generando graficas estaticas (PNG) — Estabilidad Temporal")
    graficar_desplazamiento_centroide(frames, tiempos, carpeta_temporal)
    graficar_potencia_normalizada(frames, tiempos, carpeta_temporal)
    graficar_correlacion_espacial(frames, tiempos, carpeta_temporal, ref_correlacion_flat)
    graficar_ancho_haz(frames, tiempos, carpeta_temporal)
    graficar_imagen_referencia(frames, tiempos, carpeta_temporal)
    graficar_imagen_normalizada(frames, carpeta_temporal)

    if GENERAR_VIDEOS:
        fps_salida   = fps
        repeticiones = max(1, round(intervalo_ef / VELOCIDAD_VIDEO))
        duracion_est = len(frames) * repeticiones / fps_salida

        print_seccion("Generando videos (MP4) — Estabilidad Temporal")
        print(f"  FPS salida   : {fps_salida:.2f}  (igual al video original)")
        print(f"  Velocidad    : {VELOCIDAD_VIDEO}x  →  {repeticiones} repeticion/frame")
        print(f"  Duracion est.: {duracion_est:.1f} s por video")

        video_desplazamiento_centroide(frames, tiempos, carpeta_temporal, fps_salida,
                                       repeticiones, frame_indices=frame_indices)
        video_potencia_normalizada(frames, tiempos, carpeta_temporal, fps_salida,
                                   repeticiones, frame_indices=frame_indices)
        video_correlacion_espacial(frames, tiempos, carpeta_temporal, fps_salida,
                                   repeticiones, frame_indices=frame_indices,
                                   ref_frame_flat=ref_correlacion_flat)
        video_ancho_haz_temporal(frames, tiempos, carpeta_temporal, fps_salida,
                                 repeticiones, frame_indices=frame_indices)
        video_ancho_haz_frames(frames, tiempos, carpeta_temporal, fps_salida,
                               repeticiones, frame_indices=frame_indices)
        video_resumen_analisis(frames, tiempos, carpeta_temporal, fps_salida,
                               repeticiones, frame_indices=frame_indices)
    else:
        print("\n  (Videos desactivados — GENERAR_VIDEOS = False)")

    # ── Análisis de polarizaciones ─────────────────────────────────────────────
    if ANALIZAR_POLARIZACIONES:
        _video_radial   = True
        _grafica_angular = True
        _pol_flags    = {}
        if session is not None:
            _video_radial    = getattr(session.analysis, 'video_perfil_radial', True)
            _grafica_angular = getattr(session.analysis, 'grafica_angulo_intensidad', True)
            _pol_flags = {
                'generar_pol_tabla':       getattr(session.analysis, 'generar_pol_tabla',       True),
                'generar_pol_promedio':    getattr(session.analysis, 'generar_pol_promedio',    True),
                'generar_pol_varianza':    getattr(session.analysis, 'generar_pol_varianza',    True),
                'generar_pol_correlacion': getattr(session.analysis, 'generar_pol_correlacion', True),
                'generar_pol_centroide':   getattr(session.analysis, 'generar_pol_centroide',   True),
                'generar_pol_energia':     getattr(session.analysis, 'generar_pol_energia',     True),
                'generar_pol_img_fso':       getattr(session.analysis, 'generar_pol_img_fso',       True),
                'generar_pol_tabla_fso':     getattr(session.analysis, 'generar_pol_tabla_fso',     True),
                'generar_pol_sensibilidad':  getattr(session.analysis, 'generar_pol_sensibilidad',  True),
                'generar_pol_txt':           getattr(session.analysis, 'generar_pol_txt',           True),
            }
        analizar_polarizaciones(carpeta_entrada, carpeta_pol_out,
                                generar_video_radial=_video_radial,
                                generar_grafica_angular=_grafica_angular,
                                flags=_pol_flags)
    else:
        print("\n  (Analisis de polarizaciones desactivado — ANALIZAR_POLARIZACIONES = False)")

    print_banner("Analisis completado. Estructura de resultados:")
    print(f"  {carpeta_salida}")
    print(f"    ├── {os.path.basename(carpeta_temporal)}/")
    print("    │      (graficas + videos de estabilidad temporal)")
    print(f"    ├── {os.path.basename(carpeta_pol_out)}/")
    print("    │      (metricas e imagenes de polarizaciones)")
    print(f"    └── {NOMBRE_SUBCARPETA_DATOS_CRUDOS}/")
    print("           (datos numericos en crudo, separados de las graficas)")
    return True


def main(session=None):
    """
    Punto de entrada del análisis.

    `session` (SessionConfig): si se pasa desde el orquestador, se usan los
    valores de session.analysis y session.um_per_px en lugar de las variables
    globales. La carpeta raíz DEL DISPOSITIVO se toma de session.cropped_folder
    o session.acquisition_folder si ya se conoce (o se pide por diálogo si no).

    Siempre se pregunta explícitamente con cuáles de los casos EXISTENTES
    (Sin turbulencia/Transitorio/Con turbulencia — no todos existen
    necesariamente) se desea trabajar, y con qué generación de datos
    (Adquisicion o Preprocesado) — esta última pregunta se hace UNA sola
    vez para todos los casos elegidos en la corrida. Los resultados de
    cada caso se escriben en <raiz>/<caso>/Analisis. Si a un caso elegido
    le falta algo imprescindible, se omite con advertencia sin abortar
    el resto del lote.

    Retorna True si al menos un caso pudo analizarse, False si el usuario
    canceló en cualquier paso o si ningún caso pudo analizarse. Nunca
    termina el proceso (no usa sys.exit/SystemExit) — el llamador decide
    cómo manejar la cancelación, igual que el resto de los flujos de
    main.py.
    """
    global TAMANO_PIXEL_UM, ANALIZAR_TODOS, INTERVALO_FRAMES, FPS_VIDEO
    global UMBRAL_INTENSIDAD, GENERAR_VIDEOS, VELOCIDAD_VIDEO
    global ANALIZAR_POLARIZACIONES, FRACCIONES_ENERGIA
    global ANALIZAR_TODOS_ANGULOS, INTERVALO_ANGULOS, ANCHO_PERFIL_RADIAL
    global RADIAL_MOSTRAR_E2, RADIAL_MOSTRAR_MAXIMOS

    # ── Aplicar valores de session si se proporcionó ──────────────────────────
    if session is not None:
        a = session.analysis
        TAMANO_PIXEL_UM        = session.um_per_px
        ANALIZAR_TODOS         = a.analizar_todos
        INTERVALO_FRAMES       = a.intervalo_frames
        FPS_VIDEO              = a.fps_video
        UMBRAL_INTENSIDAD      = a.umbral_intensidad
        GENERAR_VIDEOS         = a.generar_videos
        VELOCIDAD_VIDEO        = a.velocidad_video
        ANALIZAR_POLARIZACIONES   = a.analizar_polarizaciones
        FRACCIONES_ENERGIA        = a.fracciones_energia
        ANALIZAR_TODOS_ANGULOS    = getattr(a, 'analizar_todos_angulos', True)
        INTERVALO_ANGULOS         = max(1, getattr(a, 'intervalo_angulos', 1))
        ANCHO_PERFIL_RADIAL       = max(1, getattr(a, 'ancho_perfil_radial', 1))
        RADIAL_MOSTRAR_E2         = getattr(a, 'radial_mostrar_e2', True)
        RADIAL_MOSTRAR_MAXIMOS    = getattr(a, 'radial_mostrar_maximos', True)

        # Aplicar activación/desactivación de resultados individuales
        _patch_analysis_functions(a)

    print_banner("ANALISIS DE RESULTADOS EXPERIMENTALES")

    # ── 1. Selección de la carpeta raíz del dispositivo ───────────────────────
    carpeta_raiz = None
    if session is not None:
        carpeta_raiz = (
            getattr(session, 'cropped_folder', None) or
            getattr(session, 'acquisition_folder', None)
        )
        if not carpeta_raiz or not os.path.exists(carpeta_raiz):
            carpeta_raiz = None
    if not carpeta_raiz:
        carpeta_raiz = _solicitar_carpeta_raiz_interactivo()
    if not carpeta_raiz:
        print_error("Ejecución cancelada por el usuario.")
        return False

    # ── 2. Casos existentes → selección múltiple (siempre se pregunta) ───────
    casos_disp = utils_carpetas.casos_existentes(carpeta_raiz)
    if not casos_disp:
        print_error(
            "No se encontró ningún caso (Sin turbulencia/Transitorio/Con "
            f"turbulencia) dentro de la carpeta raíz: {carpeta_raiz}")
        return False

    caso_sesion = getattr(session, 'caso_actual', None) if session is not None else None
    preseleccionados = [caso_sesion] if caso_sesion in casos_disp else None
    casos_elegidos = pedir_casos_multiples(casos_disp, preseleccionados=preseleccionados)
    if not casos_elegidos:
        print_error("Ejecución cancelada por el usuario.")
        return False

    # Siempre se pregunta explícitamente con qué generación de datos trabajar,
    # una sola vez para todos los casos elegidos en esta corrida.
    eleccion = pedir_eleccion_carpeta(carpeta_raiz)
    if not eleccion:
        print_error("Ejecución cancelada por el usuario.")
        return False

    print(f"\nCarpeta raíz del dispositivo : {carpeta_raiz}")
    print("Casos elegidos               : " + ", ".join(
        utils_carpetas.CASO_LABELS.get(c, c) for c in casos_elegidos))
    print(f"Origen de los datos          : {eleccion}")

    # ── 3. Ejecutar el análisis por cada caso elegido ─────────────────────────
    algun_caso_ok = False
    for caso in casos_elegidos:
        carpeta_caso = os.path.join(carpeta_raiz, caso)
        print_banner(f"CASO: {utils_carpetas.CASO_LABELS.get(caso, caso)}")
        try:
            ok = _ejecutar_analisis_caso(carpeta_caso, eleccion, session=session)
        except (FileNotFoundError, ValueError) as e:
            ok = False
            print_warn(f"Caso omitido — {e}")
        except Exception as e:
            # Cualquier otro error no previsto (ej. MemoryError) tampoco debe
            # abortar el resto del lote -- se omite este caso y se continua
            # con el siguiente, igual que con FileNotFoundError/ValueError.
            # Se imprime el traceback completo (a diferencia de los casos
            # anteriores) porque un error no previsto no tiene un mensaje
            # de por si claro para el usuario.
            ok = False
            print_error(f"Caso omitido por error inesperado: {e!r}")
            traceback.print_exc()
        algun_caso_ok = algun_caso_ok or ok

    if not algun_caso_ok:
        print_error("Ningún caso pudo analizarse.")
        return False

    return True


_FUNCIONES_ANALISIS_ORIGINALES = {}  # se rellena la primera vez que se llama _patch_analysis_functions


def _patch_analysis_functions(a):
    """
    Activa/desactiva las funciones de generación de resultados según lo
    que el usuario eligió en la pestaña "Resultados" de AnalysisDialog.
    Restaura explícitamente la función real cuando el flag es True, no
    solo reemplaza por no-op cuando es False -- necesario porque main.py
    puede ejecutar la Opción 3 varias veces sin reiniciar el proceso
    Python (el usuario vuelve al menú principal entre corridas, sin
    borrar la terminal). Sin esta restauración explícita en ambos
    sentidos, una función desactivada en una corrida anterior quedaba
    "apagada" en el módulo para siempre, en todas las corridas siguientes
    del mismo proceso, aunque el usuario la reactivara -- verificado
    empíricamente antes de este fix (contaminación de estado entre
    corridas, sin ningún error visible).
    """
    import sys
    mod = sys.modules[__name__]

    def _noop(*args, **kwargs):
        """Sustituto inerte: acepta cualquier firma y no hace nada. Se
        instala en lugar de la función real cuando el usuario desactiva
        ese resultado en AnalysisDialog."""
        pass

    mapping = [
        ('generar_desplazamiento_centroide', 'graficar_desplazamiento_centroide'),
        ('generar_potencia_normalizada',     'graficar_potencia_normalizada'),
        ('generar_correlacion_espacial',     'graficar_correlacion_espacial'),
        ('generar_ancho_haz',               'graficar_ancho_haz'),
        ('generar_imagen_referencia',        'graficar_imagen_referencia'),
        ('generar_imagen_normalizada',       'graficar_imagen_normalizada'),
        ('video_desplazamiento_centroide',   'video_desplazamiento_centroide'),
        ('video_potencia_normalizada',       'video_potencia_normalizada'),
        ('video_correlacion_espacial',       'video_correlacion_espacial'),
        ('video_ancho_haz_temporal',         'video_ancho_haz_temporal'),
        ('video_ancho_haz_frames',           'video_ancho_haz_frames'),
        ('generar_video_resumen',            'video_resumen_analisis'),
    ]
    for attr_flag, func_name in mapping:
        # Capturar la función REAL la primera vez que se ve este nombre,
        # antes de que cualquier corrida la haya podido reemplazar por
        # _noop -- es la única referencia fiable para poder restaurarla.
        if func_name not in _FUNCIONES_ANALISIS_ORIGINALES:
            _FUNCIONES_ANALISIS_ORIGINALES[func_name] = getattr(mod, func_name)
        if getattr(a, attr_flag, True):
            setattr(mod, func_name, _FUNCIONES_ANALISIS_ORIGINALES[func_name])
        else:
            setattr(mod, func_name, _noop)


if __name__ == "__main__":
    main()