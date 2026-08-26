# -*- coding: utf-8 -*-
"""
ComparacionTurbulencia.py — Comparación de mediciones de cámara de
                             turbulencia (barridos de caracterización).

Descripción:
Compara N barridos (corridas) de la Opción 4 (adquisición) + Opción 5
(preprocesado) de la cámara de turbulencia, reutilizando las funciones
físicas ya validadas de CamaraTurbulencia.py (calcular_W_fibre, despejar_r0,
calcular_Cn2, analizar_video_from_centroids) para que los resultados sean
exactamente consistentes con el análisis individual (Opción 6).

Metodología:
1) Calcula r0 y Cn2 DIRECTAMENTE desde los videos preprocesados
   (_proc.mp4) de cada barrido — nunca desde carpetas de resultados de la
   Opción 6, para no depender de un análisis previo que podría perderse
   o no existir.
2) Verifica (sin bloquear) que los parámetros físicos del montaje
   (longitud de onda, distancia focal, distancia de propagación,
   calibración de píxel) sean consistentes entre los barridos comparados.
3) Genera las salidas (todas en español, formato de publicación):
       Tabla_Parametros_Turbulencia.csv/.png
       Fig_Turbulencia_Barridos_Comparados.png

Autor: Diego Aguilar
"""

import os
import sys
import json as _json
import csv as _csv
import types as _types
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

# Reutilizamos las funciones fisicas ya validadas de CamaraTurbulencia.py.
# Los imports de hardware en ese modulo son "lazy" (solo se cargan dentro
# de main_adquisicion), por lo que importarlo aqui es seguro y NO requiere
# camara, polarizador ni DAQ conectados.
import CamaraTurbulencia as _CT
import utils_carpetas
from console_ui import print_banner, print_error, print_warn
from gui.eleccion_origen_datos_dialog import pedir_origen_datos
from gui.dialogos_comunes import pedir_carpeta, pedir_entero, pedir_texto


# =============================================================================
# ESTILO DE PUBLICACION (igual al resto del programa)
# =============================================================================

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
    "font.family":       "serif",
    "font.size":         19,
    "axes.titlesize":    23,
    "axes.labelsize":    20,
    "xtick.labelsize":   16,
    "ytick.labelsize":   16,
    "legend.fontsize":   16,
    "axes.grid":         True,
    "grid.color":        "#CCCCCC",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.4,
    "lines.markersize":  8,
    "mathtext.fontset":  "dejavuserif",
})

_PALETA = ["#1565C0", "#C62828", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F"]

_MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
_MARCADORES_SIMBOLO = {
    "o": "\u25cf", "s": "\u25a0", "^": "\u25b2", "D": "\u25c6",
    "v": "\u25bc", "P": "\u271a", "X": "\u2716", "*": "\u2605",
    "h": "\u2b21", "<": "\u25c0", ">": "\u25b6",
}
_LETRAS = "abcdefghijklmnopqrstuvwxyz"


# =============================================================================
# DIALOGOS DE ENTRADA
# =============================================================================

_asegurar_carpeta = utils_carpetas.asegurar_carpeta


# =============================================================================
# CARGA DE UN BARRIDO DESDE SU CARPETA PREPROCESADA
# =============================================================================

def _armar_session_fake(metadata: dict) -> _types.SimpleNamespace:
    """Construye un objeto de sesion 'falso' a partir de metadata_adquisicion.json,
    con la misma estructura que usa CamaraTurbulencia.main_analisis."""
    camd = metadata.get("optica", {})
    cam2 = metadata.get("camara", {})
    um_per_px = float(cam2.get("um_per_px", 2.2))

    return _types.SimpleNamespace(
        turbulence=_types.SimpleNamespace(
            wavelength_nm   = float(camd.get("wavelength_nm",   980)),
            focal_length_mm = float(camd.get("focal_length_mm", 11.0)),
            distance_m      = float(camd.get("distance_m",      1.0)),
            metodo_w0       = camd.get("metodo_w0",       "directo"),
            w_fibre_um      = float(camd.get("w_fibre_um",      0.0)),
        ),
        scale=_types.SimpleNamespace(um_per_px=um_per_px),
        # CamaraTurbulencia.analizar_video_from_centroids lee la longitud de
        # onda por session.experiment.wavelength (no por
        # session.turbulence.wavelength_nm) — sin este campo, cae siempre al
        # fallback interno de 980 nm sin importar el valor real guardado
        # arriba, mismo patron que usa CamaraTurbulencia.main_analisis.
        experiment=_types.SimpleNamespace(
            wavelength=str(camd.get("wavelength_nm", 980)) + " nm",
        ),
        _w0_fijo_m=None,
    )


def _extraer_campos_fisicos(metadata: dict) -> dict:
    """
    Extrae del metadata_adquisicion.json los parametros fisicos del montaje
    (longitud de onda, distancia focal, distancia de propagacion,
    calibracion de pixel) para poder compararlos entre barridos -- ver
    _verificar_parametros_fisicos().
    """
    camd = metadata.get("optica", {})
    cam2 = metadata.get("camara", {})
    return {
        "wavelength_nm":   camd.get("wavelength_nm"),
        "focal_length_mm": camd.get("focal_length_mm"),
        "distance_m":      camd.get("distance_m"),
        "um_per_px":       cam2.get("um_per_px"),
    }


def _verificar_parametros_fisicos(barridos: list) -> None:
    """
    Informa (sin bloquear) si la longitud de onda, distancia focal,
    distancia de propagacion o calibracion de pixel difieren entre los
    barridos comparados. A diferencia de la Opcion 8 (donde comparar haces
    de dispositivos distintos con exposicion/canal distintos puede ser
    legitimo), los barridos de esta opcion normalmente deberian venir del
    MISMO montaje optico de la camara de turbulencia -- una diferencia aqui
    suele ser mas indicativa de un error (ej. mezclar corridas con la
    optica reconfigurada) que de una eleccion deliberada, por eso se avisa
    explicitamente aunque no se bloquee la comparacion.
    """
    _campos_info = [("wavelength_nm", "Longitud de onda (nm)"),
                     ("focal_length_mm", "Distancia focal f1 (mm)"),
                     ("distance_m", "Distancia de propagacion L (m)"),
                     ("um_per_px", "Calibracion (um/px)")]
    for campo, etiqueta in _campos_info:
        valores = {b["nombre"]: (b.get("fisica") or {}).get(campo) for b in barridos}
        distintos = set(v for v in valores.values() if v is not None)
        if len(distintos) > 1:
            print_warn(f"  {etiqueta} distinto entre los barridos comparados: " +
                       ", ".join(f"'{n}'={v}" for n, v in valores.items() if v is not None))
        elif not distintos:
            print_warn(f"  No se pudo verificar '{etiqueta}' en ninguno de los barridos "
                       "comparados (metadata_adquisicion.json no encontrado o incompleto).")


def _cargar_barrido_video(carpeta: str, nombre_barrido: str, crudo: bool) -> dict | None:
    """
    Carga un barrido COMPLETO reprocesando video, desde 'Preprocesado'
    (crudo=False, video_<tag>_proc.mp4 -- ya con ROI y dark frame
    aplicados por la Opcion 5) o desde 'Adquisicion' (crudo=True,
    video_<tag>.mp4 -- crudo, sin ROI ni resta de dark frame). Lee
    metadata_adquisicion.json para conocer los casos (DeltaT, velocidad)
    y procesa cada video directamente, calculando r0 y Cn2 con las mismas
    funciones que usa la Opci\u00f3n 6.
    """
    nombre_carpeta_esperada = "Adquisicion" if crudo else "Preprocesado"
    meta_path = os.path.join(carpeta, "metadata_adquisicion.json")
    if not os.path.exists(meta_path):
        print_error(f"No se encontr\u00f3 metadata_adquisicion.json en: {carpeta}")
        print(f"    Verifica que sea la carpeta '{nombre_carpeta_esperada}' de la caracterizaci\u00f3n.")
        return None

    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = _json.load(fh)

    # Deduplicar casos (por si el metadata tiene entradas repetidas)
    _tags_vistos = set()
    casos_uniq = []
    for c in metadata.get("casos", []):
        if c.get("carpeta") and c["carpeta"] not in _tags_vistos:
            _tags_vistos.add(c["carpeta"]); casos_uniq.append(c)

    if not casos_uniq:
        print_error(f"Sin casos en metadata_adquisicion.json: {carpeta}")
        return None

    def _video_path(tag: str) -> str:
        """Ruta del video de un caso, con el sufijo `_proc` solo si se
        eligió trabajar sobre datos preprocesados (ver `crudo`)."""
        nombre = f"video_{tag}.mp4" if crudo else f"video_{tag}_proc.mp4"
        return os.path.join(carpeta, tag, nombre)

    session_fake = _armar_session_fake(metadata)
    print(f"  '{nombre_barrido}': {len(casos_uniq)} casos encontrados")

    # ── W0 desde el caso de referencia (DeltaT=0, v=0), calculado UNA vez ─────
    _ref_exacto = next((c for c in casos_uniq
                        if float(c.get("delta_T_C", 99)) == 0.0
                        and float(c.get("velocity_ms", 99)) == 0.0), None)
    if _ref_exacto is not None:
        _ref = _ref_exacto
    else:
        _ref = casos_uniq[0]
        print_warn(f"  '{nombre_barrido}': no se encontro el caso de referencia DeltaT=0, "
                   f"v=0 -- usando '{_ref['carpeta']}' (DeltaT={_ref.get('delta_T_C')}, "
                   f"v={_ref.get('velocity_ms')}) como referencia de W0 en su lugar. Si ese "
                   "caso tiene turbulencia activa, W0 (y por tanto r0/Cn2/sigma_R de todo "
                   "el barrido) quedara sesgado.")
    _tag_ref  = _ref["carpeta"]
    _mp4_ref  = _video_path(_tag_ref)
    if not os.path.exists(_mp4_ref):
        print_error(f"No se encontr\u00f3 el video de referencia: {_mp4_ref}")
        return None

    _cap_ref = cv2.VideoCapture(_mp4_ref)
    _ok_ref, _f_ref = _cap_ref.read()
    _cap_ref.release()
    if not _ok_ref or _f_ref is None:
        print_error(f"No se pudo leer el video de referencia: {_mp4_ref}")
        return None
    if _f_ref.ndim == 3:
        _f_ref = cv2.cvtColor(_f_ref, cv2.COLOR_BGR2GRAY)

    _rw0 = _CT.calcular_W_fibre(_CT._to2d(_f_ref), session_fake.scale.um_per_px)
    _w0v = _rw0[0] if isinstance(_rw0, tuple) else float(_rw0)
    session_fake._w0_fijo_m = _w0v
    print(f"    W0 de referencia = {_w0v * 1e6:.4f} \u00b5m "
          f"(DeltaT={_ref.get('delta_T_C')}, v={_ref.get('velocity_ms')})")

    # ── Procesar cada caso: calcular r0, Cn2 desde su video _proc.mp4 ─────────
    datos_barrido = {}
    for caso in casos_uniq:
        tag = caso["carpeta"]
        dt  = float(caso.get("delta_T_C", 0.0))
        vel = float(caso.get("velocity_ms", 0.0))
        t_var_s = float(caso.get("t_var_s", 25.0))

        mp4_path = _video_path(tag)
        if not os.path.exists(mp4_path):
            print_error(f"[{tag}] Video no encontrado, se omite.")
            continue

        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print_error(f"[{tag}] No se pudo abrir el video, se omite.")
            continue
        fps_real = cap.get(cv2.CAP_PROP_FPS)

        xs, ys, first_frame = [], [], None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if first_frame is None:
                first_frame = frame.copy()
            cx, cy = _CT._centroide(frame, umbral_frac=0.0)
            xs.append(cx); ys.append(cy)
        cap.release()

        if not xs:
            print_error(f"[{tag}] Sin frames leidos, se omite.")
            continue

        # Usar el fps REAL del .mp4 en vez de asumir 30 fps -- mismo fix que
        # CamaraTurbulencia.main_analisis, necesario para que t_var_s
        # (exclusion del transitorio termico) se aplique sobre el eje de
        # tiempo correcto.
        if fps_real and fps_real > 0:
            timestamps = np.arange(len(xs)) / fps_real
        else:
            print_warn(f"[{tag}] fps invalido reportado por el video ({fps_real}); "
                       "usando 30 fps de respaldo.")
            timestamps = np.arange(len(xs)) / 30.0

        resultado = _CT.analizar_video_from_centroids(
            xs, ys, first_frame, session_fake,
            timestamps=timestamps, t_var_s=t_var_s)
        if not resultado:
            print_error(f"[{tag}] No se pudo calcular r0/Cn2, se omite.")
            continue

        datos_barrido[(dt, vel)] = resultado
        print(f"    [{tag}] r0 = {resultado['r0_mm']:.4f} mm  |  "
              f"Cn2 = {resultado['Cn2']:.4e} m^-2/3")

    if not datos_barrido:
        print_error(f"No se pudo calcular ningun caso del barrido '{nombre_barrido}'.")
        return None

    return {"nombre": nombre_barrido, "datos": datos_barrido,
            "fisica": _extraer_campos_fisicos(metadata)}



def _recopilar_barridos(origen: str) -> list[dict]:
    """
    Pregunta al usuario cuantos barridos va a comparar y, para cada uno,
    pide su carpeta RAIZ de la caracterizacion (contiene Adquisicion/ y
    Preprocesado/) y la carga segun el `origen` elegido para toda la
    comparacion ("Preprocesado" o "Adquisicion").
    """
    n = pedir_entero(
        "Comparaci\u00f3n de mediciones de c\u00e1mara de turbulencia",
        "\u00bfCu\u00e1ntos barridos vas a comparar?\n"
        "(ej. 2: dos barridos de caracterizaci\u00f3n de la c\u00e1mara)",
        inicial=2, minimo=1)
    if not n:
        return []

    barridos = []
    for i in range(1, n + 1):
        nombre = pedir_texto(
            f"Barrido {i}/{n}",
            f"Nombre del barrido #{i} (ej. 'Barrido 1', 'Prueba A'):",
            inicial=f"Barrido {i}")
        if not nombre:
            nombre = f"Barrido {i}"

        carpeta = pedir_carpeta(
            f"[{nombre}] Selecciona la carpeta RAIZ de la caracterizacion\n"
            f"(contiene Adquisicion/ y Preprocesado/).")
        if not carpeta:
            print(f"  Barrido '{nombre}' omitido (sin carpeta seleccionada).")
            continue
        carpeta_raiz = utils_carpetas.normalizar_carpeta_raiz(carpeta)

        crudo = (origen == "Adquisicion")
        nombre_sub = (utils_carpetas.NOMBRE_ADQUISICION if crudo
                      else utils_carpetas.NOMBRE_PREPROCESADO)
        datos = _cargar_barrido_video(
            os.path.join(carpeta_raiz, nombre_sub), nombre, crudo=crudo)
        if datos is None:
            print(f"  Barrido '{nombre}' omitido (no se pudo procesar).")
            continue

        barridos.append(datos)

    return barridos


# =============================================================================
# ASIGNACION DE MARCADORES POR POSICION
# =============================================================================

def _asignar_marcadores(barridos: list[dict]) -> dict:
    """
    Asigna una LETRA (a, b, c, ...) y un MARCADOR a cada caso segun su
    POSICION (orden) dentro de su propio barrido -- no por coincidencia
    exacta de (DeltaT, v), ya que estos pueden variar ligeramente entre
    barridos "repetidos" (ej. 160 vs 163 °C). Se asume que todos los
    barridos tienen el mismo número de casos, en el mismo orden.

    Retorna: dict {(nombre_barrido, DeltaT, v): (letra, marcador)}
    """
    resultado = {}
    for b in barridos:
        casos_ordenados = sorted(b["datos"].keys())
        for i, (dt, vel) in enumerate(casos_ordenados):
            letra    = _LETRAS[i % len(_LETRAS)]
            marcador = _MARCADORES[i % len(_MARCADORES)]
            resultado[(b["nombre"], dt, vel)] = (letra, marcador)
    return resultado


# =============================================================================
# TABLA Y FIGURA COMPARATIVA
# =============================================================================

def _generar_tabla_turbulencia(barridos: list[dict], carpeta_salida: str, carpeta_datos: str):
    """Genera la tabla de parámetros de turbulencia por caso (todos los barridos).
    El CSV (dato numérico) va a `carpeta_datos`; el PNG (figura) a `carpeta_salida`."""
    _asegurar_carpeta(carpeta_salida)
    _asegurar_carpeta(carpeta_datos)
    asignacion = _asignar_marcadores(barridos)
    encabezado = ["Barrido", "Marcador", "$\\Delta T$ (\u00b0C)", "$v$ (m/s)",
                  "$r_0$ (mm)", "$C_n^2$ (m$^{-2/3}$)", "$\\sigma_R$"]
    filas = []
    for b in barridos:
        for (dt, vel) in sorted(b["datos"].keys()):
            m = b["datos"][(dt, vel)]
            letra, marcador = asignacion.get((b["nombre"], dt, vel), ("a", "o"))
            simbolo = _MARCADORES_SIMBOLO.get(marcador, "\u25cf")
            filas.append([
                f"{b['nombre']}{letra}", simbolo, f"{dt:.1f}", f"{vel:.2f}",
                f"{m.get('r0_mm', float('nan')):.4f}",
                f"{m.get('Cn2', float('nan')):.4e}",
                f"{m.get('sigma_R', float('nan')):.6f}",
            ])

    if not filas:
        print_error("No hay datos para generar la tabla.")
        return

    ruta_csv = os.path.join(carpeta_datos, "Tabla_Parametros_Turbulencia.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(encabezado)
        w.writerows(filas)
    print(f"  Guardada: {os.path.basename(ruta_csv)}")

    fig_h = 1.3 + 0.55 * len(filas)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=encabezado,
                      loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(16)
    tabla.auto_set_column_width(col=list(range(len(encabezado))))
    tabla.scale(1, 1.7)
    for (row, col), cell in tabla.get_celld().items():
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")
            if col == 1:
                cell.set_text_props(fontsize=20)
    fig.suptitle("Parámetros de turbulencia por caso",
                 fontsize=22, fontweight="bold", y=0.98)
    fig.tight_layout()
    ruta_png = os.path.join(carpeta_salida, "Tabla_Parametros_Turbulencia.png")
    fig.savefig(ruta_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardada: {os.path.basename(ruta_png)}")


def _fig_turbulencia_comparada(barridos: list[dict], carpeta_salida: str):
    """
    Figura de 3 paneles (r0, Cn2, sigma_R vs DeltaT) con todos los barridos
    superpuestos. Cada barrido tiene un color; cada posicion dentro del
    barrido tiene un marcador (decodificado en la tabla, para no saturar
    la leyenda).
    """
    _asegurar_carpeta(carpeta_salida)
    metricas = [("r0_mm",   "$r_0$ (mm)",           "$r_0$ vs $\\Delta T$",      False),
                ("Cn2",     "$C_n^2$ (m$^{-2/3}$)", "$C_n^2$ vs $\\Delta T$",    True),
                ("sigma_R", "$\\sigma_R$ (adim.)",  "$\\sigma_R$ vs $\\Delta T$", False)]

    asignacion = _asignar_marcadores(barridos)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for ax, (clave, ylabel, titulo, logy) in zip(axes, metricas):
        for i, b in enumerate(barridos):
            color = _PALETA[i % len(_PALETA)]
            puntos = sorted(b["datos"].items())
            for (dt, vel), val_dict in puntos:
                val = val_dict.get(clave, float("nan"))
                _letra, mk = asignacion.get((b["nombre"], dt, vel), ("a", "o"))
                plot_fn = ax.semilogy if logy else ax.plot
                plot_fn([dt], [val], marker=mk, color=color, ms=13,
                        ls="none", mec="black", mew=0.7)
            ax.plot([], [], marker="o", color=color, ms=11, ls="none",
                    label=b["nombre"])
        ax.set_xlabel("$\\Delta T$ (\u00b0C)")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo, fontweight="bold")
        ax.legend(fontsize=15, loc="best")

    fig.suptitle("Consistencia entre barridos de caracterización\n"
                 "de la cámara de turbulencia",
                 fontsize=22, fontweight="bold", y=1.05)
    fig.tight_layout()
    ruta = os.path.join(carpeta_salida, "Fig_Turbulencia_Barridos_Comparados.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardada: {os.path.basename(ruta)}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main(session=None) -> str | None:
    """
    Punto de entrada de la Opción 9 — evalúa la REPRODUCIBILIDAD de la
    cámara de turbulencia comparando varios barridos independientes.

    Secuencia:
      1. Se elige el origen de datos, aplicado a todos los barridos.
      2. Se recopilan N≥1 barridos, recalculando r₀/Cₙ²/σ_R desde los
         videos con las mismas funciones físicas de la Opción 6 (nunca se
         leen resultados ya calculados, para no depender de un análisis
         previo que pudiera faltar o estar desactualizado).
      3. `_verificar_parametros_fisicos` avisa —sin bloquear— si la
         longitud de onda, focal, distancia o calibración difieren entre
         barridos: a diferencia de la Opción 8, aquí lo esperable es que
         todos vengan del MISMO montaje, así que una diferencia suele
         indicar un error de selección de carpetas.
      4. Se genera la tabla comparativa y la figura de 3 paneles
         (r₀, Cₙ² en semilog, σ_R) frente a ΔT.

    Que los barridos se superpongan en esas curvas es lo que justifica
    usar la cámara como instrumento de referencia en la Opción 8.

    Retorna la carpeta de resultados creada, o None si se canceló.
    """
    print_banner("COMPARACION DE MEDICIONES DE CAMARA DE TURBULENCIA")

    origen = pedir_origen_datos()
    if not origen:
        print_error("No se selecciono origen de datos. Cancelado.")
        return None
    print(f"  Origen de datos: {origen}")

    barridos = _recopilar_barridos(origen)
    if not barridos:
        print_error("No se recopilaron barridos. Cancelado.")
        return None
    _verificar_parametros_fisicos(barridos)

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
    carpeta_salida = os.path.join(base_salida, f"{fecha}_Comparacion_Turbulencia")
    os.makedirs(carpeta_salida, exist_ok=False)
    # Datos numericos (CSV) separados de las graficas/videos, igual que
    # Parte3.py/CamaraTurbulencia.py (Opciones 3/6).
    carpeta_datos = os.path.join(carpeta_salida, utils_carpetas.NOMBRE_SUBCARPETA_DATOS_CRUDOS)
    os.makedirs(carpeta_datos, exist_ok=True)
    print(f"\nCarpeta de resultados: {carpeta_salida}")

    print_banner("GENERANDO TABLA Y FIGURA COMPARATIVA")
    _generar_tabla_turbulencia(barridos, carpeta_salida, carpeta_datos)
    _fig_turbulencia_comparada(barridos, carpeta_salida)

    print_banner("COMPARACION COMPLETADA")
    print(f"  Resultados en: {carpeta_salida}")
    return carpeta_salida


if __name__ == "__main__":
    main(session=None)
