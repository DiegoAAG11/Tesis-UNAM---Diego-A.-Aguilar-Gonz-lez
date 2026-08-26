# -*- coding: utf-8 -*-
"""
ComparacionResiliencia.py — Comparación de la resiliencia de haz óptico bajo
                             turbulencia atmosférica.

Descripción:
Compara N haces (ej. Gaussiano, MMI-1, MMI-2) bajo distintas condiciones de
turbulencia (sin turbulencia / transitorio / estable), calculando para cada
haz y condición:
    - Desplazamiento radial del centroide r(t)  [µm]  y su RMS
    - Wander normalizado = r_RMS / W0            (W0 = ancho del haz sin turbulencia)
    - Intensidad normalizada I(t) y su fluctuación relativa σ_I
    - Correlación espacial de Pearson C(t) respecto al promedio del primer
      segundo de la condición 'sin_turbulencia' del mismo haz (misma
      referencia para las 3 condiciones, para que la caída de correlación
      mida degradación real respecto al estado normal del haz)
    - Ancho del haz D4σ (x,y) y su variación porcentual respecto a sin turbulencia

Metodología:
1) Se pregunta explícitamente UNA sola vez con qué origen de datos
   trabajar (ver gui/eleccion_origen_datos_dialog.py): "Preprocesado" o
   "Adquisicion" (ambos reprocesan video/imagen desde cero -- necesario
   para calcular la referencia compartida de correlación espacial).
2) Para cada haz se pide UNA sola carpeta — la RAÍZ del dispositivo (la
   misma que usan las Opciones 1-3) — y cada condición se resuelve a su
   subcarpeta de caso (Sin turbulencia/Transitorio/Con turbulencia) según
   ese origen.
3) Se calculan las métricas de arriba por haz y condición, y se generan
   las salidas (todas en la carpeta de resultados elegida por el usuario;
   los identificadores "FigN" son fijos, no un índice de orden de
   generación):

   Tabla resumen (todos los haces/condiciones):
       Tabla_Resumen_Resiliencia.csv   — tabla numérica completa
       Tabla_Resumen_Resiliencia.png   — tabla formateada para diapositivas

   Por condición (sin_turbulencia / transitorio / estable), comparando
   todos los haces entre sí:
       Fig1_desplazamiento_comparado_<condicion>.png
       Fig6_intensidad_comparada_<condicion>.png
       Fig2_correlacion_comparada_<condicion>.png
       Fig3_barras_resumen_<condicion>.png/.csv
       Fig7_dispersion_centroide_<haz>_<condicion>.png/_datos.csv/_resumen.txt
           (mapa de dispersión + histograma Rayleigh + test KS bootstrap,
           por haz; solo condiciones estacionarias, no 'transitorio')
       Fig8_varianza_polarizacion_<haz>_<condicion>.png/_datos.csv
           (mapa de varianza sobre polarizaciones, por haz; solo
           condiciones estacionarias)

   Por haz (sin_turbulencia vs. estable, sin el transitorio):
       Fig1b_desplazamiento_por_haz_<haz>.png/.csv
       Fig2b_correlacion_por_haz_<haz>.png/.csv
       Fig6b_intensidad_por_haz_<haz>.png/.csv

   Una sola vez (todos los haces):
       Fig4_imagenes_referencia.png       — sin vs. con turbulencia, por haz
       Fig5_imagenes_solo_referencia.png  — imágenes de referencia con
                                              colorbar compartido
       Video_transitorio_<haz>.mp4        — video del caso transitorio
                                              recortado y centrado, por haz

Autor: Diego Aguilar
"""

import os
import sys
import json as _json
import csv as _csv
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import utils_carpetas
import utils_imagenes
from console_ui import print_banner, print_error, print_warn
from gui.eleccion_origen_datos_dialog import pedir_origen_datos
from gui.dialogos_comunes import pedir_carpeta, pedir_entero, pedir_texto

# Reutilizamos las funciones ya validadas de Parte3.py para que los
# resultados sean exactamente consistentes con el análisis individual.
import Parte3 as _P3


# =============================================================================
# ESTILO DE PUBLICACIÓN (igual al resto del programa)
# =============================================================================

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.grid":         True,
    "grid.color":        "#CCCCCC",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.4,
    "lines.markersize":  8,
    # Math y texto normal deben usar el mismo estilo tipografico
    "mathtext.fontset":  "dejavuserif",
})

_PALETA = ["#1565C0", "#C62828", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F"]


_asegurar_carpeta = utils_carpetas.asegurar_carpeta


def _carpeta_datos_crudos(carpeta_salida: str) -> str:
    """
    Carpeta HERMANA de datos numericos en crudo (CSV/TXT), separada de las
    graficas/videos — mismo patron que usan Parte3.py/CamaraTurbulencia.py
    (Opciones 3/6). Se crea si no existe.
    """
    carpeta = os.path.join(carpeta_salida, utils_carpetas.NOMBRE_SUBCARPETA_DATOS_CRUDOS)
    _asegurar_carpeta(carpeta)
    return carpeta


# =============================================================================
# CARGA DE UN VIDEO DE ESTABILIDAD TEMPORAL DESDE UNA CARPETA "Preprocesado"
# =============================================================================

def _cargar_condicion(carpeta: str, ref_frame: np.ndarray | None = None) -> dict | None:
    """
    Dada una carpeta de haz (funciona tanto con la carpeta Adquisicion
    cruda como con la carpeta Preprocesado), localiza el video de
    estabilidad temporal y calcula todas las metricas leyendo EXCLUSIVAMENTE
    el .mp4 frame a frame (sin cargar toda la lista en RAM, y sin usar el
    .npz de datos crudos — eso queda pendiente para un analisis posterior).

    ref_frame: frame de referencia para la correlacion espacial (ya
    promediado). Si es None, se calcula AQUI como el promedio de los
    frames del primer segundo de ESTE video (uso tipico: la condicion
    'sin_turbulencia', que define la referencia). Si se provee, se usa
    directamente (uso tipico: 'transitorio'/'estable', reutilizando la
    referencia de 'sin_turbulencia' del mismo haz).
    """
    carpeta_est = _P3.encontrar_carpeta_estabilidad(carpeta)
    if carpeta_est is None:
        print_error(f"No se encontró 'estabilidad_temporal' en: {carpeta}")
        return None

    # encontrar_video solo busca extensiones de video (.mp4/.avi/.mov/.mkv);
    # nunca selecciona un .npz aunque exista junto al video.
    ruta_video = _P3.encontrar_video(carpeta_est)
    if ruta_video is None:
        print_error(f"No se encontró ningún video (.mp4) en: {carpeta_est}")
        return None

    print(f"  Cargando (.mp4): {os.path.basename(ruta_video)}")
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print_error(f"No se pudo abrir: {ruta_video}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if ref_frame is None:
        # Primera pasada: promediar los frames del primer segundo para
        # definir la referencia de correlacion (solo cuando este video es
        # el que define la referencia, tipicamente 'sin_turbulencia').
        n_ref = max(1, int(round(fps * 1.0)))
        buf_ref = []
        for _ in range(n_ref):
            ret, fr = cap.read()
            if not ret:
                break
            if fr.ndim == 3:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            buf_ref.append(fr.astype(np.float64))
        if buf_ref:
            ref_frame = np.mean(buf_ref, axis=0)
        cap.release()
        cap = cv2.VideoCapture(ruta_video)  # rebobinar para la pasada completa
        if not cap.isOpened():
            print_error(f"No se pudo reabrir: {ruta_video}")
            return None
    else:
        # La referencia viene de OTRA condicion del mismo haz (ej.
        # 'sin_turbulencia'). Si el ROI de esta condicion tiene dimensiones
        # distintas (p. ej. porque el recorte se hizo por separado en la
        # Opcion 2), se redimensiona la referencia para que la correlacion
        # se pueda calcular igual, en vez de fallar.
        _w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if ref_frame.shape[:2] != (_h_video, _w_video):
            print(f"  Advertencia: el ROI de esta condicion "
                  f"({_w_video}x{_h_video} px) difiere del de referencia "
                  f"({ref_frame.shape[1]}x{ref_frame.shape[0]} px). "
                  f"Redimensionando la referencia para la correlacion.")
            ref_frame = cv2.resize(ref_frame.astype(np.float32),
                                   (_w_video, _h_video),
                                   interpolation=cv2.INTER_LINEAR).astype(np.float64)

    ref_flat = (ref_frame.flatten().astype(np.float64)
                if ref_frame is not None else None)

    cx_list, cy_list   = [], []
    potencias          = []
    corrs              = []
    anchos_x, anchos_y = [], []
    frame0             = None
    n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame0 is None:
            frame0 = frame.copy()
            if ref_flat is None:  # resguardo de seguridad, no deberia pasar
                ref_flat = frame0.flatten().astype(np.float64)

        cx, cy = _P3.calcular_centroide(frame)
        cx_list.append(cx); cy_list.append(cy)

        potencias.append(float(frame.astype(np.float64).sum()))
        corrs.append(float(np.corrcoef(
            ref_flat, frame.flatten().astype(np.float64))[0, 1]))

        wx, wy, _ = _P3.calcular_ancho_haz(frame)
        anchos_x.append(wx); anchos_y.append(wy)
        n += 1

    cap.release()

    if n == 0:
        print_error(f"No se pudieron leer frames de: {ruta_video}")
        return None

    tiempos = np.arange(n) / fps

    # ── Centroide y desplazamiento radial (ref = promedio de todos los frames)
    cx_all = np.array(cx_list); cy_all = np.array(cy_list)
    cx0, cy0 = float(cx_all.mean()), float(cy_all.mean())
    dx_um = _P3._to_um(cx_all - cx0)
    dy_um = _P3._to_um(cy_all - cy0)
    r_um  = np.sqrt(dx_um**2 + dy_um**2)
    r_rms = float(np.sqrt(np.mean(r_um**2)))

    # ── Intensidad normalizada ────────────────────────────────────────────────
    potencias = np.array(potencias)
    p_norm    = potencias / potencias.mean()
    sigma_I   = float((potencias**2).mean() / potencias.mean()**2 - 1)

    # ── Correlacion espacial media ────────────────────────────────────────────
    corrs      = np.array(corrs)
    corr_media = float(corrs.mean())

    # ── Ancho del haz D4sigma (x, y) ──────────────────────────────────────────
    anchos_x_um = _P3._to_um(np.array(anchos_x))
    anchos_y_um = _P3._to_um(np.array(anchos_y))
    w_media     = float(((anchos_x_um + anchos_y_um) / 2.0).mean())

    return {
        "carpeta":         carpeta,
        "ruta_video":      ruta_video,
        "n_frames":        n,
        "fps":             fps,
        "tiempos":         tiempos,
        "frame0":          frame0,
        "ref_frame_used":  ref_frame,
        "dx_um":       dx_um,
        "dy_um":       dy_um,
        "r_um":        r_um,
        "r_rms":       r_rms,
        "p_norm":      p_norm,
        "sigma_I":     sigma_I,
        "corrs":       corrs,
        "corr_media":  corr_media,
        "w_media":     w_media,
        "anchos_x":    anchos_x_um,
        "anchos_y":    anchos_y_um,
    }


# =============================================================================
# DIÁLOGOS DE ENTRADA
# =============================================================================

_pedir_carpeta = pedir_carpeta
_pedir_entero  = pedir_entero
_pedir_texto   = pedir_texto


# =============================================================================
# RECOPILACIÓN INTERACTIVA DE TODOS LOS CASOS
# =============================================================================

_CONDICIONES = [
    ("sin_turbulencia", "Sin turbulencia (referencia)"),
    ("transitorio",      "Transitorio (impulso de turbulencia)"),
    ("estable",          "Turbulencia estable"),
]

# Etiquetas para graficas/tablas (formato de publicacion, en espanol).
_CONDICIONES_ES = {
    "sin_turbulencia": "Sin turbulencia atmosferica",
    "transitorio":      "Transitorio",
    "estable":          "Con turbulencia atmosferica",
}
_LABEL_CORTA_ES = {
    "sin_turbulencia": "Sin turbulencia",
    "transitorio":      "Transitorio",
    "estable":          "Con turbulencia",
}


_MAPA_CLAVE_A_CARPETA_CASO = {
    "sin_turbulencia": "SinTurbulencia",
    "transitorio":      "Transitorio",
    "estable":          "ConTurbulencia",
}


def _resolver_carpeta_condicion(carpeta_raiz: str, clave: str, origen: str) -> str | None:
    """
    Dada la carpeta RAÍZ de un dispositivo, resuelve la carpeta a usar
    para la condición `clave` (sin_turbulencia/transitorio/estable) según
    el `origen` elegido explícitamente por el usuario ("Adquisicion" o
    "Preprocesado"). Retorna None si el caso no existe en absoluto dentro
    de la raíz, o si la subcarpeta pedida no existe.
    """
    nombre_caso = _MAPA_CLAVE_A_CARPETA_CASO[clave]
    carpeta_caso = os.path.join(carpeta_raiz, nombre_caso)
    if not os.path.isdir(carpeta_caso):
        return None
    nombre_sub = (utils_carpetas.NOMBRE_ADQUISICION if origen == "Adquisicion"
                  else utils_carpetas.NOMBRE_PREPROCESADO)
    carpeta_sub = os.path.join(carpeta_caso, nombre_sub)
    if os.path.isdir(carpeta_sub):
        return carpeta_sub
    return None


def _recopilar_haces(origen: str) -> list[dict]:
    """
    Pregunta al usuario cuántos haces va a comparar y, para cada uno,
    pide UNA sola carpeta — la RAÍZ del dispositivo (la misma que usan
    las Opciones 1-3) — y resuelve cada condición de turbulencia a su
    subcarpeta de caso correspondiente, según el `origen` de datos
    elegido para toda la comparación.
    Retorna una lista de dicts: {"nombre": str, "condiciones": {clave: carpeta}}
    """
    n_haces = _pedir_entero(
        "Comparación de la resiliencia de haz",
        "¿Cuántos haces vas a comparar?\n(ej. 3: Gaussiano, MMI-1, MMI-2)",
        inicial=3, minimo=2)
    if not n_haces:
        return []

    haces = []
    nombres_usados = set()
    for i in range(1, n_haces + 1):
        _default_nombre = "Gaussiano" if i == 1 else f"Haz estructurado {i - 1}"
        nombre = _pedir_texto(
            f"Haz {i}/{n_haces}",
            f"Nombre del haz #{i} (ej. 'Gaussiano', 'Haz estructurado 1'):",
            inicial=_default_nombre)
        if not nombre:
            nombre = _default_nombre

        # Nombres duplicados sobrescribirían silenciosamente los archivos de
        # salida del haz anterior (mismo patrón usado en cada figura/CSV) --
        # se desambigua automáticamente, igual que utils_carpetas.
        if nombre in nombres_usados:
            _nombre_original = nombre
            _n = 2
            while f"{_nombre_original} ({_n})" in nombres_usados:
                _n += 1
            nombre = f"{_nombre_original} ({_n})"
            print_warn(f"El nombre '{_nombre_original}' ya se usó para otro haz en esta "
                       f"comparación -- este se renombró a '{nombre}' para no sobrescribir "
                       "sus resultados.")
        nombres_usados.add(nombre)

        print_banner(f"HAZ: {nombre}")

        carpeta_raiz = _pedir_carpeta(
            f"[{nombre}] Selecciona la carpeta RAÍZ del dispositivo\n"
            "(contiene los casos: Sin turbulencia, Transitorio, Con turbulencia).",
            obligatoria=True)
        if not carpeta_raiz:
            print_error(f"Carpeta raíz requerida para el haz '{nombre}'.")
            return []
        carpeta_raiz = utils_carpetas.normalizar_carpeta_raiz_dispositivo(carpeta_raiz)

        condiciones = {}
        for clave, descripcion in _CONDICIONES:
            obligatoria = (clave != "transitorio")  # transitorio es opcional
            carpeta = _resolver_carpeta_condicion(carpeta_raiz, clave, origen)
            if carpeta:
                condiciones[clave] = carpeta
                print(f"  {descripcion}: {carpeta}")
            elif obligatoria:
                print_error(f"'{descripcion}' es obligatoria y no se encontró "
                           f"dentro de: {carpeta_raiz}")
                return []
            else:
                print(f"  ({descripcion} no encontrada — se omite)")

        haces.append({"nombre": nombre, "condiciones": condiciones, "_origen": origen})

    return haces


def _leer_metadata_experimental_haz(haz: dict) -> dict | None:
    """
    Ubica el metadata_adquisicion.json REAL de un haz (siempre dentro de
    la subcarpeta Adquisicion de cualquiera de sus condiciones resueltas
    -- Adquisicion y Preprocesado tienen ambas una copia idéntica de este
    archivo) y retorna un dict con sus campos experimentales clave
    (um_per_px, channel, wavelength, exposure_time_us, gain_db), o None si
    no se pudo leer.
    """
    for carpeta in haz["condiciones"].values():
        carpeta_caso = os.path.dirname(carpeta)
        ruta_meta = os.path.join(carpeta_caso, utils_carpetas.NOMBRE_ADQUISICION,
                                  "metadata_adquisicion.json")
        if os.path.isfile(ruta_meta):
            try:
                with open(ruta_meta, "r", encoding="utf-8") as fh:
                    meta = _json.load(fh)
                cam = meta.get("camara") or {}
                cal = meta.get("calibracion") or {}
                um  = cal.get("um_per_px")
                return {
                    "um_per_px":        float(um) if um is not None else None,
                    "channel":          meta.get("channel"),
                    "wavelength":       meta.get("wavelength"),
                    "exposure_time_us": cam.get("exposure_time_us"),
                    "gain_db":          cam.get("gain_db"),
                }
            except Exception:
                return None
    return None


def _verificar_calibraciones(haces: list, um_per_px_usado: float) -> bool:
    """
    Compara la calibración um_per_px REAL con la que cada haz fue
    adquirido (guardada en su propio metadata_adquisicion.json) contra la
    calibración `um_per_px_usado` que se está aplicando a TODOS los
    cálculos en µm de esta comparación (propagada a _P3.TAMANO_PIXEL_UM).

    Sin esta verificación, comparar un haz adquirido con otra calibración
    (otra fecha/óptica/zoom) contra uno actual deja todas sus magnitudes
    en µm calculadas con la escala física incorrecta, sin ningún aviso.

    Adicionalmente, informa (sin bloquear) si canal/longitud de onda/
    exposición/ganancia difieren entre los haces comparados -- a diferencia
    de la calibración, estos SÍ pueden diferir legítimamente entre
    dispositivos/experimentos distintos, así que no ameritan bloquear la
    comparación, solo que el usuario los tenga a la vista.

    Retorna True si no hay discrepancias de calibración o el usuario
    confirma continuar de todas formas; False si el usuario decide cancelar.
    """
    TOLERANCIA_RELATIVA = 0.005  # 0.5%
    metadatas = {}
    discrepancias = []
    for haz in haces:
        meta = _leer_metadata_experimental_haz(haz)
        metadatas[haz["nombre"]] = meta
        if meta is None or meta["um_per_px"] is None:
            print_warn(f"  No se pudo verificar la calibración real de '{haz['nombre']}' "
                       "(metadata_adquisicion.json no encontrado o incompleto).")
            continue
        um_real = meta["um_per_px"]
        if um_per_px_usado <= 0 or abs(um_real - um_per_px_usado) / um_per_px_usado > TOLERANCIA_RELATIVA:
            discrepancias.append((haz["nombre"], um_real))

    # Verificacion informativa (no bloqueante) de otros parametros experimentales.
    _campos_info = [("channel", "Canal"), ("wavelength", "Longitud de onda"),
                     ("exposure_time_us", "Exposición (µs)"), ("gain_db", "Ganancia (dB)")]
    for campo, etiqueta in _campos_info:
        valores = {n: m.get(campo) for n, m in metadatas.items() if m is not None}
        distintos = set(v for v in valores.values() if v is not None)
        if len(distintos) > 1:
            print_warn(f"  {etiqueta} distinto entre los haces comparados: " +
                       ", ".join(f"'{n}'={v}" for n, v in valores.items() if v is not None))

    if not discrepancias:
        return True

    print_error("¡Calibración de píxel INCONSISTENTE entre los haces comparados!")
    print(f"  Calibración usada para TODOS los cálculos en µm: {um_per_px_usado:.5f} µm/px")
    for nombre, um_real in discrepancias:
        print(f"    - '{nombre}' fue adquirido con: {um_real:.5f} µm/px (DISTINTA)")
    print("  Si continúas, las magnitudes en µm de esos haces (ancho, desplazamiento, "
          "wander) quedarán calculadas con una escala física incorrecta.")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    continuar = messagebox.askyesno(
        "Calibración de píxel inconsistente",
        "Se detectaron haces adquiridos con una calibración de píxel (µm/px) "
        "distinta a la que se está usando para esta comparación (revisa la "
        "consola para el detalle).\n\n"
        "Continuar comparará magnitudes en µm calculadas con escalas físicas "
        "distintas entre haces.\n\n¿Deseas continuar de todas formas?",
        parent=root)
    root.destroy()
    return continuar


# =============================================================================
# TABLA RESUMEN
# =============================================================================

def _generar_tabla(resultados: list[dict], carpeta_salida: str,
                    haces_sin_ref: dict[str, str] | None = None):
    """
    resultados: lista de dicts {"nombre", "condicion", "metricas": {...}}
    haces_sin_ref: {nombre_haz: razon} para los haces sin referencia W0 --
    se anota como nota al pie en el CSV y la figura, en vez de dejar que
    las celdas "—" queden sin explicacion.
    Genera CSV + figura de tabla.
    """
    _asegurar_carpeta(carpeta_salida)
    encabezado = ["Haz", "Condicion", "N cuadros", "$r_{RMS}$ (µm)",
                  "$r_{RMS}/W_0$", "$\\sigma_I$", "$\\overline{C}$ (Pearson)",
                  "$W_0$ (µm)", "$\\Delta W$ (%)"]

    filas = []
    filas_csv = []
    for r in resultados:
        m = r["metricas"]
        etiqueta_corta = _LABEL_CORTA_ES.get(r["condicion_clave"], r["condicion_label"])
        fila = [
            r["nombre"], etiqueta_corta, m["n_frames"],
            f"{m['r_rms']:.4f}",
            f"{m['wander_norm']:.5f}" if m["wander_norm"] is not None else "—",
            f"{m['sigma_I']:.6f}",
            f"{m['corr_media']:.6f}",
            f"{m['w_media']:.2f}",
            f"{m['delta_w_pct']:+.2f}" if m["delta_w_pct"] is not None else "—",
        ]
        filas.append(fila)
        # El CSV conserva la etiqueta descriptiva completa para trazabilidad
        filas_csv.append([r["nombre"],
                          _CONDICIONES_ES.get(r["condicion_clave"], r["condicion_label"])]
                         + fila[2:])

    # ── CSV (Datos_Crudos, separado de la figura) ─────────────────────────────
    ruta_csv = os.path.join(_carpeta_datos_crudos(carpeta_salida), "Tabla_Resumen_Resiliencia.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(encabezado)
        w.writerows(filas_csv)
        if haces_sin_ref:
            fh.write("\n# Nota: '—' en 'r_RMS/W0' y 'Delta W (%)' indica ausencia de "
                     "referencia W0 (condicion 'sin_turbulencia'):\n")
            for nombre, razon in haces_sin_ref.items():
                fh.write(f"# - {nombre}: {razon}\n")
    print(f"  Guardada: {os.path.basename(ruta_csv)}")

    # ── Figura de tabla (para insertar directo en diapositivas) ──────────────
    nota_pie = None
    if haces_sin_ref:
        nota_pie = "Sin referencia W0 (—): " + "; ".join(
            f"{nombre} ({razon})" for nombre, razon in haces_sin_ref.items())
    fig_h = 1.3 + 0.55 * len(filas) + (0.4 if nota_pie else 0.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=encabezado,
                      loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(16)
    tabla.auto_set_column_width(col=list(range(len(encabezado))))
    tabla.scale(1, 1.7)
    for (row, col), cell in tabla.get_celld().items():
        if col in (0, 1):  # Haz y Condicion: texto, mejor alineado a la izquierda
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")
    fig.suptitle("Resumen de resiliencia bajo turbulencia",
                 fontsize=14, fontweight="bold", y=0.98)
    if nota_pie:
        fig.text(0.02, 0.01, nota_pie, fontsize=9, style="italic",
                 color="#555555", ha="left", va="bottom", wrap=True)
    fig.tight_layout()
    ruta_png = os.path.join(carpeta_salida, "Tabla_Resumen_Resiliencia.png")
    fig.savefig(ruta_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardada: {os.path.basename(ruta_png)}")


# =============================================================================
# FIGURAS COMPARATIVAS
# =============================================================================

def _fig_desplazamiento_comparado(haces: list[dict], condicion_clave: str,
                                   condicion_label: str, carpeta_salida: str):
    """Superpone r(t) de todos los haces bajo la misma condición."""
    _asegurar_carpeta(carpeta_salida)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hay_datos = False
    for i, haz in enumerate(haces):
        datos = haz.get("_datos", {}).get(condicion_clave)
        if datos is None:
            continue
        hay_datos = True
        color = _PALETA[i % len(_PALETA)]
        ax.plot(datos["tiempos"], datos["r_um"], color=color, lw=1.3,
                alpha=0.85, label=f"{haz['nombre']}  (RMS = {datos['r_rms']:.3f} \u00b5m)")

    if not hay_datos:
        plt.close(fig); return

    if condicion_clave == "transitorio":
        ax.axvline(3.5, color="black", ls=":", lw=1.5, alpha=0.7,
                   label="Inicio de turbulencia (t = 3.5 s)")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Desplazamiento radial $r(t)$ (µm)")
    ax.set_title(f"Desplazamiento del centroide — {condicion_label}",
                 fontweight="bold")
    # Leyenda fuera del area de datos -- con varios haces superpuestos en
    # todo el rango temporal, "best" no siempre encuentra una esquina libre.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    ruta = os.path.join(carpeta_salida,
                        f"Fig1_desplazamiento_comparado_{condicion_clave}.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Datos numericos en formato "long" (una fila por haz+frame), robusto
    # ante series de distinta longitud entre haces. Va en Datos_Crudos/,
    # separado de la grafica.
    _ruta_csv = os.path.join(_carpeta_datos_crudos(carpeta_salida),
                             os.path.basename(ruta).replace(".png", ".csv"))
    with open(_ruta_csv, "w", newline="", encoding="utf-8") as _fh:
        _w = _csv.writer(_fh)
        _w.writerow(["haz", "t_s", "r_um"])
        for _haz in haces:
            _datos = _haz.get("_datos", {}).get(condicion_clave)
            if _datos is None:
                continue
            for _t, _r in zip(_datos["tiempos"], _datos["r_um"]):
                _w.writerow([_haz["nombre"], f"{_t:.6f}", f"{_r:.4f}"])
    print(f"  Guardada: {os.path.basename(ruta)}")


def _fig_correlacion_comparada(haces: list[dict], condicion_clave: str,
                                condicion_label: str, carpeta_salida: str):
    """Superpone C(t) de todos los haces bajo la misma condición."""
    _asegurar_carpeta(carpeta_salida)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hay_datos = False
    _c_min, _c_max = 1.0, 1.0
    for i, haz in enumerate(haces):
        datos = haz.get("_datos", {}).get(condicion_clave)
        if datos is None:
            continue
        hay_datos = True
        color = _PALETA[i % len(_PALETA)]
        ax.plot(datos["tiempos"], datos["corrs"], color=color, lw=1.3,
                alpha=0.85,
                label=f"{haz['nombre']}  ($\\overline{{C}}$ = {datos['corr_media']:.4f})")
        _c_min = min(_c_min, float(np.min(datos["corrs"])))
        _c_max = max(_c_max, float(np.max(datos["corrs"])))

    if not hay_datos:
        plt.close(fig); return

    if condicion_clave == "transitorio":
        ax.axvline(3.5, color="black", ls=":", lw=1.5, alpha=0.7,
                   label="Inicio de turbulencia (t = 3.5 s)")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Correlacion de Pearson $C(t)$")
    ax.set_title(f"Correlacion espacial — {condicion_label}",
                 fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    # Ajustar el eje Y al rango real de los datos (con un margen del 8% del
    # rango) en vez de un limite fijo, que dejaba mucho espacio vacio cuando
    # la correlacion varia poco (ej. condicion sin turbulencia).
    _c_range = max(_c_max - _c_min, 1e-6)
    _margin  = _c_range * 0.08
    ax.set_ylim(_c_min - _margin, min(_c_max + _margin, 1.001))
    fig.tight_layout()
    ruta = os.path.join(carpeta_salida,
                        f"Fig2_correlacion_comparada_{condicion_clave}.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _ruta_csv2 = os.path.join(_carpeta_datos_crudos(carpeta_salida),
                              os.path.basename(ruta).replace(".png", ".csv"))
    with open(_ruta_csv2, "w", newline="", encoding="utf-8") as _fh2:
        _w2 = _csv.writer(_fh2)
        _w2.writerow(["haz", "t_s", "pearson_correlation"])
        for _haz2 in haces:
            _datos2 = _haz2.get("_datos", {}).get(condicion_clave)
            if _datos2 is None:
                continue
            for _t2, _c2 in zip(_datos2["tiempos"], _datos2["corrs"]):
                _w2.writerow([_haz2["nombre"], f"{_t2:.6f}", f"{_c2:.6f}"])
    print(f"  Guardada: {os.path.basename(ruta)}")


def _fig_intensidad_comparada(haces: list[dict], condicion_clave: str,
                               condicion_label: str, carpeta_salida: str):
    """
    Superpone I(t) normalizada de todos los haces bajo la misma condición.
    Representa el centelleo (scintillation) del haz.
    """
    _asegurar_carpeta(carpeta_salida)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hay_datos = False
    for i, haz in enumerate(haces):
        datos = haz.get("_datos", {}).get(condicion_clave)
        if datos is None:
            continue
        hay_datos = True
        color = _PALETA[i % len(_PALETA)]
        ax.plot(datos["tiempos"], datos["p_norm"], color=color, lw=1.0,
                alpha=0.8,
                label=f"{haz['nombre']}  ($\\sigma_I$ = {datos['sigma_I']:.6f})")

    if not hay_datos:
        plt.close(fig); return

    if condicion_clave == "transitorio":
        ax.axvline(3.5, color="black", ls=":", lw=1.5, alpha=0.7,
                   label="Inicio de turbulencia (t = 3.5 s)")

    ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Intensidad normalizada $I(t)/\\overline{I}$")
    ax.set_title(f"Intensidad normalizada — {condicion_label}",
                 fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    fig.tight_layout()
    ruta = os.path.join(carpeta_salida,
                        f"Fig6_intensidad_comparada_{condicion_clave}.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ruta_csv = os.path.join(_carpeta_datos_crudos(carpeta_salida),
                            os.path.basename(ruta).replace(".png", ".csv"))
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["haz", "t_s", "I_norm"])
        for haz in haces:
            datos = haz.get("_datos", {}).get(condicion_clave)
            if datos is None:
                continue
            for t, i_val in zip(datos["tiempos"], datos["p_norm"]):
                w.writerow([haz["nombre"], f"{t:.6f}", f"{i_val:.6f}"])
    print(f"  Guardada: {os.path.basename(ruta)}")


def _fig_barras_resumen(resultados: list[dict], condicion_clave: str,
                         condicion_label: str, carpeta_salida: str):
    """
    Grafica de barras agrupadas: 4 metricas x N haces, para una condicion dada.
    """
    _asegurar_carpeta(carpeta_salida)
    filtrados = [r for r in resultados if r["condicion_clave"] == condicion_clave]
    if not filtrados:
        return

    nombres = [r["nombre"] for r in filtrados]
    metricas_keys   = ["r_rms", "sigma_I", "corr_media", "delta_w_pct"]
    metricas_labels = ["Desplazamiento del centroide\n$r_{RMS}$ (\u00b5m)",
                       "Fluctuacion de\nintensidad $\\sigma_I$",
                       "Correlacion\nespacial media",
                       "Cambio de ancho\n$\\Delta W$ (%)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, key, label in zip(axes, metricas_keys, metricas_labels):
        vals = [r["metricas"][key] if r["metricas"][key] is not None else 0
                for r in filtrados]
        colores = [_PALETA[i % len(_PALETA)] for i in range(len(filtrados))]
        ax.bar(nombres, vals, color=colores, edgecolor="black", linewidth=0.6)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.tick_params(axis='x', rotation=25)
        ax.grid(True, axis='y', ls='--', alpha=0.5)

    fig.suptitle(f"Comparacion de metricas de resiliencia — {condicion_label}",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    ruta = os.path.join(carpeta_salida, f"Fig3_barras_resumen_{condicion_clave}.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ruta_csv = os.path.join(_carpeta_datos_crudos(carpeta_salida),
                            f"Fig3_barras_resumen_{condicion_clave}.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["haz"] + metricas_keys)
        for r in filtrados:
            fila = [r["nombre"]] + [
                r["metricas"][k] if r["metricas"][k] is not None else ""
                for k in metricas_keys]
            w.writerow(fila)
    print(f"  Guardada: {os.path.basename(ruta)}")


def _calcular_lim_global_dispersion(haces: list[dict]) -> float:
    """
    Calcula el limite de ejes GLOBAL para todos los mapas de dispersion
    de centroide generados en una misma corrida (todos los haces, ambas
    condiciones estacionarias), para permitir comparar visualmente el
    wander entre haces/condiciones con la misma escala de ejes.
    """
    max_val = 1e-6
    for haz in haces:
        for clave in ("sin_turbulencia", "estable"):
            datos = haz.get("_datos", {}).get(clave)
            if datos is None:
                continue
            max_val = max(max_val, float(
                np.abs(np.concatenate([datos["dx_um"], datos["dy_um"]])).max()))
    return max_val * 1.15


def _fig_dispersion_centroide(haces: list[dict], condicion_clave: str,
                               condicion_label: str, carpeta_salida: str,
                               lim_global: float):
    """
    Para cada haz, genera un mapa de dispersion del centroide (scatter
    dx vs dy, coloreado por tiempo) + histograma de r(t) con ajuste de
    Rayleigh. Replica el diseno de Parte3.graficar_desplazamiento_centroide.
    Solo aplica a condiciones estacionarias (no turbulencia / fully
    developed) -- no tiene sentido para el transitorio, donde la
    posicion media del haz cambia durante la grabacion.
    """
    _asegurar_carpeta(carpeta_salida)
    from scipy.stats import rayleigh as _rayleigh_dist
    from scipy.stats import kstest as _kstest
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for haz in haces:
        datos = haz.get("_datos", {}).get(condicion_clave)
        if datos is None:
            continue

        dist_eucl = datos["r_um"]
        tiempos   = datos["tiempos"]
        dx_signed = datos["dx_um"]
        dy_signed = datos["dy_um"]

        rms      = float(np.sqrt(np.mean(dist_eucl**2)))
        dx_max   = float(np.max(np.abs(dx_signed)))
        dy_max   = float(np.max(np.abs(dy_signed)))
        dist_max = float(dist_eucl.max())
        lim = lim_global  # misma escala para TODOS los scatter maps de esta corrida

        # ── Ajuste de Rayleigh (MLE) ──────────────────────────────────────────
        # Valor p corregido por bootstrap parametrico -- ver
        # Parte3.ks_pvalue_bootstrap_rayleigh (el valor p directo de kstest
        # no es valido porque sigma_ray se estimo de la misma muestra).
        sigma_ray = float(np.sqrt(np.mean(dist_eucl**2) / 2.0))
        try:
            ks_stat, _ = _kstest(dist_eucl, "rayleigh", args=(0, sigma_ray))
            ks_p = _P3.ks_pvalue_bootstrap_rayleigh(dist_eucl, sigma_ray, ks_stat)
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")

        fig, (ax_sc, ax_hi) = plt.subplots(1, 2, figsize=(16, 8),
                                            gridspec_kw={"wspace": 0.35})

        # ── Panel izquierdo: scatter dx vs dy ────────────────────────────────
        sc = ax_sc.scatter(dx_signed, dy_signed, c=tiempos, cmap="viridis",
                           s=60, zorder=3, label="Cuadros analizados")
        ax_sc.scatter(0, 0, color="red", s=180, zorder=4, marker="*",
                      label="Centroide promedio (referencia)")

        div = make_axes_locatable(ax_sc)
        cax = div.append_axes("right", size="5%", pad=0.12)
        fig.colorbar(sc, cax=cax).set_label("Tiempo (s)", fontsize=11)

        ax_sc.annotate(
            f"RMS       = {rms:.3f} \u00b5m\n"
            f"$\\Delta x$ max = {dx_max:.3f} \u00b5m\n"
            f"$\\Delta y$ max = {dy_max:.3f} \u00b5m\n"
            f"Dist. max = {dist_max:.3f} \u00b5m",
            xy=(0.04, 0.96), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color="#111111", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.92))
        ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)
        ax_sc.set_title("Mapa de dispersion del centroide\n(relativo al centroide promedio)",
                        fontsize=13, fontweight="bold")
        ax_sc.set_xlabel("$\\Delta x$ (\u00b5m)")
        ax_sc.set_ylabel("$\\Delta y$ (\u00b5m)")
        ax_sc.legend(fontsize=10, loc="lower right")
        ax_sc.grid(True, linestyle="--", alpha=0.6)

        # ── Panel derecho: histograma + ajuste Rayleigh ──────────────────────
        n_bins = max(15, min(50, len(dist_eucl) // 20))
        ax_hi.hist(dist_eucl, bins=n_bins, density=True,
                   color="#4C72B0", edgecolor="#CCCCCC", linewidth=0.6,
                   alpha=0.80, label="Distribucion empirica")

        r_plot  = np.linspace(0, dist_eucl.max() * 1.15, 400)
        pdf_ray = _rayleigh_dist.pdf(r_plot, scale=sigma_ray)
        ax_hi.plot(r_plot, pdf_ray, color="#DD4444", lw=2.5,
                   label=f"Ajuste de Rayleigh\n$\\sigma_R$ = {sigma_ray:.3f} \u00b5m")

        ax_hi.axvline(sigma_ray, color="#DD4444", lw=1.5, ls="--", alpha=0.7)
        ax_hi.axvline(rms, color="#888888", lw=1.5, ls=":", alpha=0.8,
                      label=f"RMS = {rms:.3f} \u00b5m")

        if not np.isnan(ks_p):
            _fit_txt = "buen ajuste" if ks_p > 0.05 else "se desvia del modelo"
            ax_hi.annotate(
                f"Estadistico KS = {ks_stat:.4f}\nvalor p (bootstrap) = {ks_p:.4f}\n{_fit_txt}",
                xy=(0.97, 0.04), xycoords="axes fraction",
                fontsize=10, color="black", va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.35", fc="#eeeeee",
                         ec="#AAAAAA", alpha=0.9))

        ax_hi.set_title("Distribucion del desplazamiento del centroide",
                        fontsize=13, fontweight="bold")
        ax_hi.set_xlabel("Desplazamiento radial $r$ (\u00b5m)")
        ax_hi.set_ylabel("Densidad de probabilidad")
        # loc="upper right" (no "upper left"): el pico de la Rayleigh cae
        # en r=sigma, no en r=0, as\u00ed que "upper left" tapaba las barras
        # m\u00e1s altas del histograma. La cola derecha (r grande) s\u00ed queda
        # libre por construcci\u00f3n del eje (xlim hasta dist_eucl.max()*1.15),
        # y no choca con la anotaci\u00f3n de bondad de ajuste (esquina
        # inferior derecha, misma columna, distinta fila).
        ax_hi.legend(fontsize=10, framealpha=0.9, loc="upper right")
        ax_hi.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"{haz['nombre']} — {condicion_label}",
                     fontsize=14, fontweight="bold", y=1.02)
        # No se llama a fig.tight_layout(): es incompatible con los ejes del
        # colorbar creados vía make_axes_locatable(...).append_axes(...) (ver
        # líneas arriba) y genera un UserWarning. El espaciado entre paneles
        # ya lo controla gridspec_kw={"wspace": 0.35} y el recorte final lo
        # da bbox_inches="tight" en savefig(), igual que en las funciones
        # equivalentes de Parte3.py y CamaraTurbulencia.py.

        _tag = condicion_clave
        _nombre_seguro = haz["nombre"].replace(" ", "_").replace("/", "-")
        ruta = os.path.join(
            carpeta_salida,
            f"Fig7_dispersion_centroide_{_nombre_seguro}_{_tag}.png")
        fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        _carpeta_datos7 = _carpeta_datos_crudos(carpeta_salida)
        _base7 = f"Fig7_dispersion_centroide_{_nombre_seguro}_{_tag}"
        with open(os.path.join(_carpeta_datos7, f"{_base7}_datos.csv"),
                  "w", newline="", encoding="utf-8") as _fh7:
            _w7 = _csv.writer(_fh7)
            _w7.writerow(["t_s", "dx_um", "dy_um", "r_um"])
            for _t7, _dx7, _dy7, _r7 in zip(tiempos, dx_signed, dy_signed, dist_eucl):
                _w7.writerow([f"{_t7:.6f}", f"{_dx7:.4f}", f"{_dy7:.4f}", f"{_r7:.4f}"])
        with open(os.path.join(_carpeta_datos7, f"{_base7}_resumen.txt"),
                  "w", encoding="utf-8") as _fh7b:
            _fh7b.write(f"rms_um: {rms:.6g}\n")
            _fh7b.write(f"dx_max_um: {dx_max:.6g}\n")
            _fh7b.write(f"dy_max_um: {dy_max:.6g}\n")
            _fh7b.write(f"dist_max_um: {dist_max:.6g}\n")
            _fh7b.write(f"sigma_rayleigh_um: {sigma_ray:.6g}\n")
            _fh7b.write(f"ks_stat: {ks_stat:.6g}\n")
            _fh7b.write(f"ks_pvalue_bootstrap: {ks_p:.6g}\n")
        print(f"  Guardada: {os.path.basename(ruta)}")


def _fig_varianza_polarizacion(haces: list[dict], carpeta_salida: str):
    """
    Para cada haz y cada condicion ESTACIONARIA (sin_turbulencia, estable),
    genera el mapa de varianza del haz sobre diferentes posiciones de
    polarizacion (replica _pol_mapa_varianza de Parte3), recalculado desde
    las imagenes crudas de la subcarpeta 'diferentes_polarizaciones_haz'.

    El transitorio NUNCA se procesa aqui (no se toma polarizacion durante
    el impulso de turbulencia) -- se omite explicitamente, no por fallo.
    Si a alguno de los dos casos estacionarios le falta la carpeta de
    polarizaciones, simplemente se omite ese caso especifico sin detener
    el resto del analisis.
    """
    _asegurar_carpeta(carpeta_salida)
    um_per_px = getattr(_P3, "TAMANO_PIXEL_UM", 2.2)

    for haz in haces:
        for clave in ("sin_turbulencia", "estable"):  # transitorio excluido a proposito
            carpeta_base = haz.get("condiciones", {}).get(clave)
            if not carpeta_base:
                continue

            carpeta_pol = _P3.encontrar_carpeta_polarizaciones(carpeta_base)
            if carpeta_pol is None:
                print(f"  [{haz['nombre']} — {clave}] Sin carpeta de "
                      f"polarizaciones, se omite.")
                continue

            try:
                imagenes, _nombres_img = _P3.cargar_imagenes_polarizacion(carpeta_pol)
            except Exception as e:
                print(f"  [{haz['nombre']} — {clave}] Error cargando "
                      f"polarizaciones: {e}")
                continue

            if len(imagenes) < 2:
                print(f"  [{haz['nombre']} — {clave}] Menos de 2 imagenes "
                      f"de polarizacion, se omite.")
                continue

            stack    = np.stack(imagenes, axis=0)
            varianza = stack.var(axis=0)

            var_norm = (varianza - varianza.min()) / max(
                varianza.max() - varianza.min(), 1e-9)
            H, W = var_norm.shape[:2]

            etiqueta = _CONDICIONES_ES.get(clave, clave)

            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(var_norm, cmap="plasma", origin="upper", aspect="equal",
                           vmin=0, vmax=1, extent=[0, W * um_per_px, H * um_per_px, 0])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Varianza normalizada $\\sigma^2/\\sigma^2_{max}$",
                           fontsize=11)
            cbar.ax.tick_params(labelsize=10)
            ax.set_title(f"Varianza del haz sobre estados de polarizacion\n"
                        f"{haz['nombre']} — {etiqueta}",
                        fontsize=13, fontweight="bold")
            ax.set_xlabel("x (\u00b5m)")
            ax.set_ylabel("y (\u00b5m)")
            fig.tight_layout()

            nombre_seguro = haz["nombre"].replace(" ", "_").replace("/", "-")
            ruta = os.path.join(
                carpeta_salida,
                f"Fig8_varianza_polarizacion_{nombre_seguro}_{clave}.png")
            fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            ruta_csv8 = os.path.join(
                _carpeta_datos_crudos(carpeta_salida),
                f"Fig8_varianza_polarizacion_{nombre_seguro}_{clave}_datos.csv")
            with open(ruta_csv8, "w", newline="", encoding="utf-8") as _fh8:
                _w8 = _csv.writer(_fh8)
                for _fila8 in varianza:
                    _w8.writerow([f"{v:.8g}" for v in _fila8])
            print(f"  Guardada: {os.path.basename(ruta)}")


def _fig_por_haz(haces: list[dict], carpeta_salida: str, campo: str,
                  ylabel: str, titulo_base: str, nombre_archivo_base: str,
                  csv_col: str, es_intensidad: bool = False,
                  stat_key: str | None = None, stat_fmt: str | None = None,
                  colores: tuple = (_PALETA[0], _PALETA[1])):
    """
    Genera, POR CADA HAZ, una figura comparando 'sin_turbulencia' vs
    'estable' (SIN incluir el transitorio) para el campo de datos
    especificado (r_um, corrs, o p_norm). Complementa las figuras que
    comparan todos los haces bajo una misma condicion.

    stat_key/stat_fmt: si se dan, se agrega el valor numerico resumen
    (ej. RMS, correlacion media, sigma_I) a la leyenda de cada serie,
    igual que ya hacen las graficas que comparan los distintos haces.

    colores: par (sin_turbulencia, con_turbulencia). Cada tipo de
    metrica (desplazamiento/correlacion/intensidad) usa su propio par,
    para distinguirlas visualmente entre si a simple vista.
    """
    _asegurar_carpeta(carpeta_salida)
    for haz in haces:
        d0 = haz.get("_datos", {}).get("sin_turbulencia")
        d1 = haz.get("_datos", {}).get("estable")
        if d0 is None and d1 is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 5.5))
        _series = [(d0, "Sin turbulencia atmosferica", colores[0]),
                   (d1, "Con turbulencia atmosferica", colores[1])]
        for datos, etiqueta, color in _series:
            if datos is None:
                continue
            _label = etiqueta
            if stat_key is not None and stat_key in datos:
                _label = f"{etiqueta}  ({stat_fmt.format(datos[stat_key])})"
            ax.plot(datos["tiempos"], datos[campo], color=color, lw=1.3,
                    alpha=0.85, label=_label)

        if es_intensidad:
            ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.5)

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{titulo_base} — {haz['nombre']}", fontweight="bold")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
        if campo == "r_um":
            ax.set_ylim(bottom=0)
        fig.tight_layout()

        nombre_seguro = haz["nombre"].replace(" ", "_").replace("/", "-")
        ruta = os.path.join(carpeta_salida,
                            f"{nombre_archivo_base}_{nombre_seguro}.png")
        fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        ruta_csv = os.path.join(_carpeta_datos_crudos(carpeta_salida),
                                os.path.basename(ruta).replace(".png", ".csv"))
        with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["condicion", "t_s", csv_col])
            for datos, etiqueta, _ in _series:
                if datos is None:
                    continue
                for t, val in zip(datos["tiempos"], datos[campo]):
                    w.writerow([etiqueta, f"{t:.6f}", f"{val:.6f}"])
        print(f"  Guardada: {os.path.basename(ruta)}")


# Pares de colores fijos por tipo de metrica (sin_turbulencia, con_turbulencia).
# Cada tipo de grafica usa un par distinto para distinguirse a simple vista.
_COLORES_CORRELACION = ("#6A1B9A", "#00838F")   # morado / verde azulado
_COLORES_INTENSIDAD  = ("#2E7D32", "#EF6C00")   # verde / naranja


def _fig_desplazamiento_por_haz(haces: list[dict], carpeta_salida: str):
    """Fig1b — Beam wander r(t) de CADA haz, contrastando sin turbulencia
    vs. con turbulencia en una misma gráfica. Complementa la Fig1, que
    compara todos los haces bajo una sola condición: aquí se ve cuánto
    empeora cada haz respecto a sí mismo."""
    _fig_por_haz(haces, carpeta_salida, "r_um",
                "Desplazamiento radial $r(t)$ (\u00b5m)",
                "Desplazamiento del centroide", "Fig1b_desplazamiento_por_haz", "r_um",
                stat_key="r_rms", stat_fmt="RMS = {:.3f} \u00b5m")


def _fig_correlacion_por_haz(haces: list[dict], carpeta_salida: str):
    """Fig2b — Correlación espacial C(t) de cada haz, sin turbulencia vs.
    con turbulencia. Es la métrica más directa de la hipótesis: cuánto
    conserva cada haz su perfil original al atravesar la turbulencia."""
    _fig_por_haz(haces, carpeta_salida, "corrs",
                "Correlacion de Pearson $C(t)$",
                "Correlacion espacial", "Fig2b_correlacion_por_haz",
                "pearson_correlation",
                stat_key="corr_media", stat_fmt="$\\overline{{C}}$ = {:.4f}",
                colores=_COLORES_CORRELACION)


def _fig_intensidad_por_haz(haces: list[dict], carpeta_salida: str):
    """Fig6b — Intensidad normalizada I(t) de cada haz, sin turbulencia
    vs. con turbulencia, con su índice de centelleo σ_I en la leyenda.
    Cuantifica cuánto se degrada la estabilidad de potencia recibida."""
    _fig_por_haz(haces, carpeta_salida, "p_norm",
                "Intensidad normalizada $I(t)/\\overline{I}$",
                "Intensidad normalizada", "Fig6b_intensidad_por_haz",
                "I_norm", es_intensidad=True,
                stat_key="sigma_I", stat_fmt="$\\sigma_I$ = {:.6f}",
                colores=_COLORES_INTENSIDAD)


def _generar_video_transitorio(haces: list[dict], carpeta_salida: str):
    """
    Genera, para cada haz, un video del caso TRANSITORIO recortado y
    centrado en el centroide de referencia (el mismo de 'sin_turbulencia',
    fijo durante todo el video -- asi se ve al haz moverse DENTRO de una
    ventana estable, en vez de perseguirlo). Se resta el dark frame si la
    carpeta es de adquisicion cruda. Incluye una barra de escala fisica.

    Todos los videos comparten el MISMO tamano de recorte en pixeles
    (el mismo 'crop_size' que usa Fig5), para poder compararlos uno al
    lado del otro con la misma escala real.
    """
    _asegurar_carpeta(carpeta_salida)
    um_per_px = getattr(_P3, "TAMANO_PIXEL_UM", 2.2)
    crop_size = _determinar_crop_size(haces)
    lado_um   = crop_size * um_per_px
    bar_um    = _escala_redonda(lado_um * 0.25)
    bar_px    = max(1, int(round(bar_um / um_per_px)))

    for haz in haces:
        datos_ref = haz.get("_datos", {}).get("sin_turbulencia")
        carpeta_trans = haz.get("condiciones", {}).get("transitorio")
        if datos_ref is None or not carpeta_trans:
            continue

        f_ref = datos_ref["frame0"].astype(np.float64)
        carpeta_ref = haz.get("condiciones", {}).get("sin_turbulencia")
        dark_ref = _buscar_dark_frame_crudo(carpeta_ref) if carpeta_ref else None
        if dark_ref is not None and dark_ref.shape == f_ref.shape:
            f_ref = np.clip(f_ref - dark_ref, 0, None)
        # Centro de recorte FIJO: centroide de referencia (no se recalcula
        # frame a frame, para ver el haz moverse dentro de una ventana fija).
        cx, cy = _P3.calcular_centroide(f_ref)

        dark_trans = _buscar_dark_frame_crudo(carpeta_trans)

        carpeta_est = _P3.encontrar_carpeta_estabilidad(carpeta_trans)
        if carpeta_est is None:
            print(f"  [{haz['nombre']}] Sin 'estabilidad_temporal' en "
                  f"transitorio, se omite el video.")
            continue
        ruta_video = _P3.encontrar_video(carpeta_est)
        if ruta_video is None:
            print(f"  [{haz['nombre']}] Sin video de transitorio, se omite.")
            continue

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print(f"  [{haz['nombre']}] No se pudo abrir el video de "
                  f"transitorio, se omite.")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        nombre_seguro = haz["nombre"].replace(" ", "_").replace("/", "-")
        ruta_out = os.path.join(carpeta_salida,
                                f"Video_transitorio_{nombre_seguro}.mp4")
        writer = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (crop_size, crop_size))

        margin  = int(crop_size * 0.05)
        y_bar   = crop_size - margin
        n_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float64)
            if dark_trans is not None and dark_trans.shape == frame.shape:
                frame = np.clip(frame - dark_trans, 0, None)

            recorte = _recortar_centrado(frame, cx, cy, crop_size)
            vis = cv2.normalize(recorte, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

            cv2.line(vis_color, (margin, y_bar), (margin + bar_px, y_bar),
                     (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(vis_color, f"{bar_um:.0f} um",
                       (margin, y_bar - 12), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(vis_color)
            n_frame += 1

        cap.release()
        writer.release()
        print(f"  Guardada: {os.path.basename(ruta_out)}  ({n_frame} frames)")


def _fig_imagenes_referencia(haces: list[dict], carpeta_salida: str):
    """
    Montaje: frame de referencia (sin turbulencia) y frame bajo turbulencia
    estable, para cada haz, en una cuadrícula N_haces x 2.
    """
    _asegurar_carpeta(carpeta_salida)
    filas_validas = []
    for haz in haces:
        f0 = haz.get("_datos", {}).get("sin_turbulencia")
        f1 = haz.get("_datos", {}).get("estable")
        if f0 is not None or f1 is not None:
            filas_validas.append((haz["nombre"], f0, f1))

    if not filas_validas:
        return

    n = len(filas_validas)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (nombre, f0, f1) in enumerate(filas_validas):
        for j, (datos, titulo) in enumerate([(f0, "Sin turbulencia atmosferica"),
                                             (f1, "Con turbulencia atmosferica")]):
            ax = axes[i, j]
            if datos is not None:
                img = datos["frame0"]
                vis = img.astype(np.float64)
                vis = (vis - vis.min()) / max(vis.max() - vis.min(), 1e-9)
                ax.imshow(vis, cmap="inferno")
            ax.set_title(f"{nombre} — {titulo}" if datos is not None
                        else f"{nombre} — {titulo} (sin datos)",
                        fontsize=11)
            ax.axis("off")

    fig.suptitle("Imagenes de referencia por haz y condicion",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    ruta = os.path.join(carpeta_salida, "Fig4_imagenes_referencia.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardada: {os.path.basename(ruta)}")


def _escala_redonda(valor_aprox: float) -> float:
    """Redondea a un valor 'bonito' (1,2,5 x 10^n) cercano al valor dado,
    para elegir la longitud de una barra de escala legible."""
    if valor_aprox <= 0:
        return 100.0
    exp  = np.floor(np.log10(valor_aprox))
    base = valor_aprox / (10 ** exp)
    if base < 1.5:
        nice = 1
    elif base < 3.5:
        nice = 2
    elif base < 7.5:
        nice = 5
    else:
        nice = 10
    return float(nice * (10 ** exp))


def _buscar_dark_frame_crudo(carpeta_base: str) -> np.ndarray | None:
    """
    Si 'carpeta_base' es la carpeta ADQUISICION cruda, busca su
    subcarpeta 'darkframe' ANIDADA (cada caso adquiere y usa su propio
    dark frame) y carga la imagen en resolucion completa (sin recortar).
    Si la carpeta ya es 'Preprocesado' (procesada por la Opcion 2), el
    dark frame ya fue restado durante el recorte, por lo que se retorna
    None (no se debe restar de nuevo).
    """
    nombre_carpeta = os.path.basename(os.path.normpath(carpeta_base))
    if nombre_carpeta == utils_carpetas.NOMBRE_PREPROCESADO:
        return None

    carpeta_dark = os.path.join(carpeta_base, utils_carpetas.NOMBRE_DARKFRAME)
    if not os.path.isdir(carpeta_dark):
        return None

    for archivo in sorted(os.listdir(carpeta_dark)):
        if archivo.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            ruta = os.path.join(carpeta_dark, archivo)
            img = utils_imagenes.leer_imagen(ruta, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img.astype(np.float64)
    return None


def _recortar_centrado(frame: np.ndarray, cx: float, cy: float,
                        size: int) -> np.ndarray:
    """
    Extrae un recorte cuadrado de lado 'size' (en px) centrado en (cx, cy),
    rellenando con ceros donde no hay datos disponibles (fuera de los
    limites originales de 'frame'). Garantiza que TODOS los recortes
    generados con el mismo 'size' tengan exactamente las mismas
    dimensiones en pixeles, sin deformar el contenido.
    """
    H, W = frame.shape[:2]
    half = size / 2.0
    x0 = int(round(cx - half)); x1 = x0 + size
    y0 = int(round(cy - half)); y1 = y0 + size

    salida = np.zeros((size, size), dtype=np.float64)
    sx0, sx1 = max(x0, 0), min(x1, W)
    sy0, sy1 = max(y0, 0), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        salida[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = frame[sy0:sy1, sx0:sx1]
    return salida


def _determinar_crop_size(haces: list[dict], factor: float = 6.0,
                           minimo: int = 80, maximo: int = 1200) -> int:
    """
    Determina un tamano de recorte (en px) comun para TODOS los haces,
    basado en 6 veces el ancho D4sigma mas grande entre todos los haces
    (margen suficiente para ver el haz completo con espacio alrededor).

    IMPORTANTE: resta el dark frame (si la carpeta es de adquisicion
    cruda) antes de estimar el ancho. Sin esto, un fondo constante
    domina el calculo de "masa" del D4sigma e infla artificialmente
    la estimacion del ancho (y por lo tanto el tamano de recorte).
    """
    anchos = []
    for haz in haces:
        datos = haz.get("_datos", {}).get("sin_turbulencia")
        carpeta_base = haz.get("condiciones", {}).get("sin_turbulencia")
        if datos is None or datos.get("frame0") is None:
            continue
        f0 = datos["frame0"].astype(np.float64)
        dark = _buscar_dark_frame_crudo(carpeta_base) if carpeta_base else None
        if dark is not None and dark.shape == f0.shape:
            f0 = np.clip(f0 - dark, 0, None)
        wx, wy, _ = _P3.calcular_ancho_haz(f0)
        anchos.append(max(wx, wy))
    if not anchos:
        return minimo
    size = int(round(max(anchos) * factor))
    return int(np.clip(size, minimo, maximo))


def _fig_imagenes_solo_referencia(haces: list[dict], carpeta_salida: str):
    """
    Montaje horizontal: solo el frame 'sin turbulencia' de cada haz, uno
    junto al otro (1 fila x N_haces). Cada haz se recorta CENTRADO en su
    propio centroide, restando el dark frame si la carpeta es Adquisicion
    cruda (no necesario si ya es Preprocesado).
    Todos los recortes comparten el MISMO tamano en pixeles -- misma
    escala fisica real automaticamente, sin deformar ningun haz.
    Incluye una barra de escala fisica y una barra de intensidad.
    """
    _asegurar_carpeta(carpeta_salida)
    validos = []
    for haz in haces:
        datos = haz.get("_datos", {}).get("sin_turbulencia")
        carpeta_base = haz.get("condiciones", {}).get("sin_turbulencia")
        if datos is not None and datos.get("frame0") is not None:
            validos.append((haz["nombre"], datos, carpeta_base))

    if not validos:
        return

    um_per_px = getattr(_P3, "TAMANO_PIXEL_UM", 2.2)
    n = len(validos)
    crop_size = _determinar_crop_size(haces)
    lado_um   = crop_size * um_per_px

    paneles = []
    for nombre, datos, carpeta_base in validos:
        f0 = datos["frame0"].astype(np.float64)

        dark = _buscar_dark_frame_crudo(carpeta_base) if carpeta_base else None
        if dark is not None and dark.shape == f0.shape:
            f0 = np.clip(f0 - dark, 0, None)

        cx, cy = _P3.calcular_centroide(f0)
        recorte = _recortar_centrado(f0, cx, cy, crop_size)
        vis = (recorte - recorte.min()) / max(recorte.max() - recorte.min(), 1e-9)
        paneles.append((nombre, vis))

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0))
    if n == 1:
        axes = [axes]

    bar_um = _escala_redonda(lado_um * 0.25)
    im = None
    for ax, (nombre, vis) in zip(axes, paneles):
        # Mismo tamano en pixeles y mismo um_per_px en todos los paneles:
        # la escala fisica real ya es identica sin necesidad de limites
        # de eje adicionales.
        im = ax.imshow(vis, cmap="inferno", aspect="equal", vmin=0, vmax=1,
                       extent=[0, lado_um, lado_um, 0])
        ax.set_facecolor("black")
        ax.set_title(nombre, fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        margin = lado_um * 0.05
        y_bar  = lado_um - margin
        ax.plot([margin, margin + bar_um], [y_bar, y_bar],
                color="white", lw=5, solid_capstyle="butt", zorder=5)
        ax.text(margin + bar_um / 2, y_bar - lado_um * 0.035,
               f"{bar_um:.0f} \u00b5m", color="white", ha="center", va="bottom",
               fontsize=11, fontweight="bold", zorder=5)

    fig.suptitle("Haces bajo prueba (sin turbulencia atmosferica)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.80, wspace=0.08, left=0.02, right=0.90)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.62])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Intensidad normalizada", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    ruta = os.path.join(carpeta_salida, "Fig5_imagenes_solo_referencia.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardada: {os.path.basename(ruta)}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main(session=None) -> str | None:
    """
    Punto de entrada de la Opción 8 — la comparación que responde la
    hipótesis central de la tesis.

    Secuencia:
      1. Se pregunta UNA sola vez el origen de datos (Preprocesado o
         Adquisicion), aplicado por igual a todos los haces: comparar un
         haz con dark frame restado contra otro sin restar invalidaría la
         comparación.
      2. Se recopilan N≥2 haces, cada uno con su carpeta raíz.
      3. `_verificar_calibraciones` contrasta el µm/px real de cada haz
         (leído de su metadata) con el de la sesión actual; si difieren,
         las magnitudes en µm no son comparables y se pide confirmación
         explícita antes de continuar.
      4. Se reprocesa el video de cada condición de cada haz, calculando
         beam wander, centelleo, correlación espacial y ancho.
      5. Se generan la tabla resumen y las ~13 familias de figuras
         (ver docstring del módulo).

    Retorna la carpeta de resultados creada, o None si se canceló en
    cualquier paso o si ningún haz pudo procesarse.
    """
    print_banner("COMPARACION DE LA RESILIENCIA DE HAZ")

    # Propagar la calibracion de pixel real al modulo Parte3 (reutilizado
    # para _to_um, calcular_centroide, calcular_ancho_haz). Sin esto, los
    # calculos en um usarian el valor por defecto de Parte3 (incorrecto).
    if session is not None:
        _P3.TAMANO_PIXEL_UM = getattr(session, "um_per_px",
                                      _P3.TAMANO_PIXEL_UM)
        _umbral = getattr(getattr(session, "analysis", None),
                          "umbral_intensidad", None)
        if _umbral is not None:
            _P3.UMBRAL_INTENSIDAD = _umbral
    print(f"  Calibracion usada: {_P3.TAMANO_PIXEL_UM:.5f} \u00b5m/px")

    origen = pedir_origen_datos()
    if not origen:
        print_error("No se selecciono origen de datos. Cancelado.")
        return None
    print(f"  Origen de datos: {origen}")

    haces = _recopilar_haces(origen)
    if not haces:
        print_error("No se recopilaron datos de haces. Cancelado.")
        return None

    if not _verificar_calibraciones(haces, _P3.TAMANO_PIXEL_UM):
        print_error("Comparación cancelada por inconsistencia de calibración.")
        return None

    # ── Cargar y calcular metricas de cada condicion de cada haz ─────────────
    print_banner("CARGANDO Y ANALIZANDO VIDEOS")

    resultados = []
    for haz in haces:
        haz["_datos"] = {}
        ref_frame_haz = None  # se define con 'sin_turbulencia' y se reutiliza
        for clave, label in _CONDICIONES:
            carpeta = haz["condiciones"].get(clave)
            if not carpeta:
                continue
            print(f"\n[{haz['nombre']} — {label}]")
            datos = _cargar_condicion(carpeta, ref_frame=ref_frame_haz)
            if datos is None:
                continue
            haz["_datos"][clave] = datos
            if clave == "sin_turbulencia":
                ref_frame_haz = datos.get("ref_frame_used")

        # W0 de referencia = ancho medio SIN turbulencia (si esta disponible)
        w0_ref = None
        if "sin_turbulencia" in haz["_datos"]:
            w0_ref = haz["_datos"]["sin_turbulencia"]["w_media"]
        else:
            razon = ("no se configuro la condicion 'sin_turbulencia' para este haz"
                      if not haz["condiciones"].get("sin_turbulencia")
                      else "fallo la carga de la condicion 'sin_turbulencia' (ver error arriba)")
            print_warn(f"[{haz['nombre']}] Sin referencia W0 ({razon}): "
                       f"'r_RMS/W0' y 'Delta W (%)' quedaran como '—' para "
                       f"todas las condiciones de este haz.")
            haz["_sin_ref_razon"] = razon

        for clave, label in _CONDICIONES:
            datos = haz["_datos"].get(clave)
            if datos is None:
                continue
            wander_norm = (datos["r_rms"] / w0_ref) if w0_ref else None
            delta_w_pct = (100.0 * (datos["w_media"] - w0_ref) / w0_ref
                          if w0_ref else None)
            resultados.append({
                "nombre":          haz["nombre"],
                "condicion_clave": clave,
                "condicion_label": label,
                "metricas": {
                    "n_frames":    datos["n_frames"],
                    "r_rms":       datos["r_rms"],
                    "wander_norm": wander_norm,
                    "sigma_I":     datos["sigma_I"],
                    "corr_media":  datos["corr_media"],
                    "w_media":     datos["w_media"],
                    "delta_w_pct": delta_w_pct,
                },
            })

    if not resultados:
        print()
        print_error("No se pudo calcular ninguna metrica. Verifica las carpetas seleccionadas.")
        return None

    # ── Carpeta de salida ──────────────────────────────────────────────────────
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    messagebox.showinfo("Carpeta de resultados",
                        "Selecciona la carpeta donde guardar la comparacion.",
                        parent=root)
    base_salida = filedialog.askdirectory(title="Carpeta de resultados", parent=root)
    root.destroy()
    if not base_salida:
        print_error("No se selecciono carpeta de salida.")
        return None

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_salida = os.path.join(base_salida, f"{fecha}_Comparacion_Resiliencia")
    os.makedirs(carpeta_salida, exist_ok=False)
    print(f"\nCarpeta de resultados: {carpeta_salida}")

    # ── Generar tabla resumen ────────────────────────────────────────────────
    print_banner("GENERANDO TABLA RESUMEN DE RESILIENCIA")
    haces_sin_ref = {h["nombre"]: h["_sin_ref_razon"]
                      for h in haces if h.get("_sin_ref_razon")}
    _generar_tabla(resultados, carpeta_salida, haces_sin_ref)

    # ── Generar figuras comparativas por condicion ───────────────────────────
    print_banner("GENERANDO FIGURAS COMPARATIVAS DE RESILIENCIA")
    lim_global_dispersion = _calcular_lim_global_dispersion(haces)
    for clave, label in _CONDICIONES:
        label_es = _CONDICIONES_ES.get(clave, label)
        _fig_desplazamiento_comparado(haces, clave, label_es, carpeta_salida)
        _fig_intensidad_comparada(haces, clave, label_es, carpeta_salida)
        _fig_correlacion_comparada(haces, clave, label_es, carpeta_salida)
        _fig_barras_resumen(resultados, clave, label_es, carpeta_salida)
        # Mapa de dispersion + distribucion de Rayleigh: solo tiene
        # sentido para condiciones estacionarias (no para el transitorio,
        # donde la posicion media del haz cambia durante la grabacion).
        if clave != "transitorio":
            _fig_dispersion_centroide(haces, clave, label_es, carpeta_salida,
                                      lim_global_dispersion)
    _fig_imagenes_referencia(haces, carpeta_salida)
    _fig_imagenes_solo_referencia(haces, carpeta_salida)
    _generar_video_transitorio(haces, carpeta_salida)
    _fig_varianza_polarizacion(haces, carpeta_salida)
    # Figuras por haz (sin turbulencia vs con turbulencia, sin transitorio)
    _fig_desplazamiento_por_haz(haces, carpeta_salida)
    _fig_correlacion_por_haz(haces, carpeta_salida)
    _fig_intensidad_por_haz(haces, carpeta_salida)

    print_banner("COMPARACION COMPLETADA")
    print(f"  Resultados en: {carpeta_salida}")
    return carpeta_salida


if __name__ == "__main__":
    main(session=None)
