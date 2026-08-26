# -*- coding: utf-8 -*-
"""
MedicionWFibre.py — Medición del radio del campo modal W_fibre del haz gaussiano.

Descripción:
Calcula W_fibre = 2·σ_prom a partir de una fotografía del haz colimado
usando el método del segundo momento espacial (D4σ / 2). Puede ejecutarse
desde el menú principal (Opción 7) o de forma independiente. El valor
calculado se propaga a session.turbulence.w_fibre_um para que la Opción 6
(análisis de la cámara de turbulencia) lo use directamente.

Metodología:
1) El usuario elige entre capturar una foto nueva con la cámara
   o cargar una imagen ya guardada.
2) Se aplica el preprocesado (resta de dark frame si está disponible,
   recorte opcional a la ROI).
3) Se calculan el centroide y el segundo momento espacial.
4) Se pide una carpeta de resultados y se genera, si se eligió una,
   una figura (`<fecha>_wfibre_imagen.png`) con la imagen del haz y las
   elipses 1σ y 2σ superpuestas, y un reporte numérico
   (`<fecha>_wfibre_reporte.txt`) con todos los valores calculados.
5) El valor de W_fibre se propaga a session.turbulence.w_fibre_um
   (independientemente de si se guardó carpeta de resultados o no).

Autor: Diego Aguilar
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime

import cv2
import numpy as np
import utils_imagenes
import utils_beam_metrics as _beam_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vmbpy

# GUI unificada de vista en vivo
import os as _os, sys as _sys
_LV_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _LV_ROOT not in _sys.path: _sys.path.insert(0, _LV_ROOT)
from gui.live_view import mostrar_captura_espacio as _lv_capture

from console_ui import print_banner, print_ok, print_error, print_warn

# ── Integracion con SessionConfig ─────────────────────────────────────────────
SESSION = None

def _scale(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.scale, key, fallback)
    return fallback

def _cam(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.camera, key, fallback)
    return fallback


# =============================================================================
# CAPTURA DE IMAGEN CON LA CÁMARA
# =============================================================================

def capturar_imagen_camara(exposure_us: int = 10_000,
                            gain_db: float = 10.0) -> np.ndarray | None:
    """
    Abre la camara en modo streaming usando gui.live_view.
    El usuario presiona ESPACIO para capturar o Q para cancelar.
    """
    with vmbpy.VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No se encontro ninguna camara Allied Vision.")
            return None

        with cams[0] as cam:
            try:
                cam.ExposureAuto.set("Off")
                cam.ExposureTime.set(exposure_us)
                cam.Gain.set(gain_db)
                _pf = _cam("pixel_format") if SESSION else "Mono8"
                if _pf:
                    try: cam.get_feature_by_name("PixelFormat").set(_pf)
                    except Exception as _e: print(f"PixelFormat: {_e}")
            except Exception as e:
                print(f"Advertencia al configurar camara: {e}")

            return _lv_capture(
                cam,
                "Medicion de W_fibre — Haz colimado",
                ["Asegurese de que el haz este correctamente colimado.",
                 "Ajuste el enfoque hasta ver el perfil mas nitido."]
            )


# =============================================================================
# CÁLCULO DE W_FIBRE
# =============================================================================

def calcular_w_fibre(img: np.ndarray,
                     um_per_px: float) -> dict:
    """
    Calcula W_fibre = 2·σ_prom del perfil de intensidad, usando
    utils_beam_metrics (compartido con CamaraTurbulencia.calcular_W_fibre,
    que mide la misma cantidad física por el método directo -- ambos deben
    dar el mismo resultado para la misma imagen).

    Centroide y segundo momento se calculan con umbral ISO 11146
    (umbral_frac=0.05): el fondo/ruido residual se descarta antes de
    ubicar el centro y antes de integrar sigma, igual que en
    CamaraTurbulencia.calcular_W_fibre.

    Retorna un dict con:
        cx, cy       : centroide [px]
        sigma_x_px,
        sigma_y_px   : desviaciones estándar espaciales [px]
        sigma_x_um,
        sigma_y_um   : ídem en µm
        w_fibre_um   : 2·σ_prom [µm]
        w_fibre_m    : ídem en metros
    """
    cx, cy   = _beam_metrics.centroide(img, umbral_frac=0.05)
    sx_px, sy_px = _beam_metrics.segundo_momento(img, cx, cy, umbral_frac=0.05)
    sigma_avg_px = (sx_px + sy_px) / 2.0
    w_fibre_um   = 2.0 * sigma_avg_px * um_per_px
    return {
        "cx": cx, "cy": cy,
        "sigma_x_px": sx_px, "sigma_y_px": sy_px,
        "sigma_x_um": sx_px * um_per_px,
        "sigma_y_um": sy_px * um_per_px,
        "w_fibre_um": w_fibre_um,
        "w_fibre_m":  w_fibre_um * 1e-6,
    }


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def guardar_figura(img: np.ndarray, resultado: dict,
                   um_per_px: float, ruta: str) -> None:
    """Guarda figura con la imagen del haz y las elipses σ superpuestas."""
    if img.dtype != np.uint8:
        vis = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        vis = img.copy()

    H, W  = vis.shape
    cx    = resultado["cx"]
    cy    = resultado["cy"]
    sx_px = resultado["sigma_x_px"]
    sy_px = resultado["sigma_y_px"]
    u     = "µm"

    # Extent en µm
    ext = [0, W * um_per_px, H * um_per_px, 0]

    fig, ax = plt.subplots(figsize=(7, 7 * H / W + 1.2), facecolor="white")
    ax.imshow(vis, cmap="inferno", origin="upper",
              aspect="equal", vmin=0, vmax=255, extent=ext)
    ax.plot(cx * um_per_px, cy * um_per_px,
            "r+", markersize=14, markeredgewidth=2)

    theta_t = np.linspace(0, 2 * np.pi, 300)
    ax.plot((cx + sx_px * np.cos(theta_t)) * um_per_px,
            (cy + sy_px * np.sin(theta_t)) * um_per_px,
            "cyan", lw=1.5, ls="--", label="1σ ellipse")
    ax.plot((cx + 2 * sx_px * np.cos(theta_t)) * um_per_px,
            (cy + 2 * sy_px * np.sin(theta_t)) * um_per_px,
            "lime", lw=1.2, ls=":", label="2σ  (D4σ / 2 = W_fibre)")

    ax.legend(fontsize=9, facecolor="white", framealpha=0.85)
    ax.set_title(
        f"Medición de W_fibre\n"
        f"W_fibre = {resultado['w_fibre_um']:.4f} {u}   "
        f"(σx = {resultado['sigma_x_um']:.3f} {u},  "
        f"σy = {resultado['sigma_y_um']:.3f} {u})",
        fontsize=10, fontweight="bold")
    ax.set_xlabel(f"x ({u})", fontsize=9)
    ax.set_ylabel(f"y ({u})", fontsize=9)
    cbar = fig.colorbar(ax.get_images()[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intens. norm.", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardada: {os.path.basename(ruta)}")


def guardar_reporte(resultado: dict, um_per_px: float,
                    ruta: str, session=None) -> None:
    """Guarda un .txt con todos los valores calculados."""
    exp  = getattr(session, "experiment", None)
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    lineas = [
        "=" * 60,
        "  MEDICIÓN DE W_FIBRE — RADIO DEL CAMPO MODAL",
        "=" * 60, "",
        f"  Fecha y hora          : {now}",
        f"  Longitud de onda      : {getattr(exp, 'wavelength', '?')}",
        f"  Tamaño de píxel       : {um_per_px:.5f} µm/px",
        "",
        "── Resultado ────────────────────────────────────────────",
        f"  σx                    : {resultado['sigma_x_um']:.4f} µm",
        f"  σy                    : {resultado['sigma_y_um']:.4f} µm",
        f"  σ_prom = (σx+σy)/2    : {(resultado['sigma_x_um']+resultado['sigma_y_um'])/2:.4f} µm",
        f"  W_fibre = 2·σ_prom    : {resultado['w_fibre_um']:.4f} µm",
        f"  W_fibre               : {resultado['w_fibre_m']:.6e} m",
        f"  Centroide (x, y)      : ({resultado['cx']:.2f}, {resultado['cy']:.2f}) px",
        "",
        "=" * 60,
    ]
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    print(f"  Guardado:  {os.path.basename(ruta)}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main(session=None) -> float | None:
    """
    Mide W_fibre a partir de una imagen del haz colimado.
    Propaga el valor a session.turbulence.w_fibre_um si session != None.
    Retorna W_fibre en µm, o None si se cancela.
    """
    global SESSION
    SESSION = session

    um_per_px = _scale("um_per_px", 2.2)
    exp_us    = _cam("exposure_time", 10_000)
    gain_db   = _cam("gain", 10.0)

    print_banner("MEDICIÓN DE W_FIBRE — RADIO DEL CAMPO MODAL")

    # ── Origen de la imagen ───────────────────────────────────────────────────
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    usar_camara = messagebox.askyesno(
        "Fuente de imagen",
        "¿Deseas capturar una nueva foto con la cámara?\n\n"
        "• SÍ  → Vista en vivo; presiona ESPACIO para capturar.\n"
        "• NO  → Selecciona una imagen ya guardada (.png / .tif / .bmp / .jpg).",
        parent=root)
    root.destroy()

    img = None
    if usar_camara:
        print("  Abriendo cámara...")
        img = capturar_imagen_camara(exposure_us=exp_us, gain_db=gain_db)
        if img is None:
            print_error("No se capturó imagen.")
            return None
        print_ok(f"Imagen capturada  ({img.shape[1]}×{img.shape[0]} px)")
    else:
        root2 = tk.Tk(); root2.withdraw(); root2.attributes("-topmost", True)
        ruta_img = filedialog.askopenfilename(
            title="Seleccionar imagen del haz",
            filetypes=[("Imágenes", "*.png *.tif *.tiff *.bmp *.jpg *.jpeg"),
                       ("Todos", "*.*")],
            parent=root2)
        root2.destroy()
        if not ruta_img:
            print_error("No se seleccionó imagen.")
            return None
        raw = utils_imagenes.leer_imagen(ruta_img, cv2.IMREAD_UNCHANGED)
        if raw is None:
            print_error(f"No se pudo leer: {ruta_img}")
            return None
        if raw.ndim == 3:
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            img = raw
        print_ok(f"Imagen cargada: {os.path.basename(ruta_img)}  "
                 f"({img.shape[1]}×{img.shape[0]} px)")

    # ── Cálculo ───────────────────────────────────────────────────────────────
    resultado  = calcular_w_fibre(img, um_per_px)
    w_fibre_um = resultado["w_fibre_um"]

    print(f"\n  σx = {resultado['sigma_x_um']:.4f} µm   "
          f"σy = {resultado['sigma_y_um']:.4f} µm")
    print(f"  W_fibre = 2·σ_prom = {w_fibre_um:.4f} µm  "
          f"= {resultado['w_fibre_m']:.4e} m")

    # ── Selección de carpeta de resultados ────────────────────────────────────
    root3 = tk.Tk(); root3.withdraw(); root3.attributes("-topmost", True)
    carpeta = filedialog.askdirectory(
        title="Seleccionar carpeta para guardar resultados de W_fibre",
        parent=root3)
    root3.destroy()

    if carpeta:
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
        guardar_figura(img, resultado, um_per_px,
                       os.path.join(carpeta, f"{fecha}_wfibre_imagen.png"))
        guardar_reporte(resultado, um_per_px,
                        os.path.join(carpeta, f"{fecha}_wfibre_reporte.txt"),
                        session=session)
    else:
        print_warn("No se seleccionó carpeta — resultados no guardados.")

    # ── Propagar a la sesión ──────────────────────────────────────────────────
    if session is not None:
        session.turbulence.w_fibre_um = w_fibre_um
        print()
        print_ok(f"W_fibre propagado a la sesión: {w_fibre_um:.4f} µm")
        print("     (disponible automáticamente en la Opción 6)")

    return w_fibre_um


if __name__ == "__main__":
    main(session=None)
