# -*- coding: utf-8 -*-
"""
Calibración de Tamaño de Píxel a partir de una captura en vivo
de la punta de una fibra óptica (Allied Vision Alvium 1800 U-501 mNIR)

Descripción:
Adquiere una imagen en tiempo real desde la cámara, permite al usuario
capturarla manualmente y calcula la escala real en µm/px a partir del
diámetro físico conocido de la punta de la fibra.

Metodología:
1) Inicializa la cámara Allied Vision mediante vmbpy (Vimba X).
2) Muestra un streaming en vivo mientras solicita al usuario que ilumine la fibra.
3) El usuario presiona ESPACIO para capturar el frame actual.
4) Segmenta el haz mediante umbralización (Otsu).
5) Detecta el contorno principal y ajusta un círculo.
6) Calcula la relación µm/px usando el diámetro real del haz.
7) Genera visualización diagnóstica y reporte numérico.

Controles durante el streaming:
    ESPACIO  →  Capturar frame y ejecutar calibración
    Q        →  Salir sin calibrar

Autor: Diego Aguilar
"""

import vmbpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

from gui.ui_style import crear_botones_confirmar_cancelar
from console_ui import print_banner, print_ok


# ── Integración con SessionConfig (inicializador main.py) ───────────────────────
# SESSION se inyecta desde main.py cuando se ejecuta el flujo completo.
# Si el script se corre de forma independiente, SESSION queda en None
# y usa los parámetros configurables definidos abajo.
SESSION = None

def _cal(key):
    """Lee un valor de SESSION.calibration si está disponible, si no del global."""
    if SESSION is not None:
        return getattr(SESSION.calibration, key)
    _defaults = {
        'beam_diameter_um': BEAM_DIAMETER_UM,
        'exposure_us':      EXPOSURE_TIME_US,
        'gain_db':          GAIN_DB,
        'save_figure':      SAVE_FIGURE,
        'show_figure':      SHOW_FIGURE,
    }
    return _defaults[key]
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# PARÁMETROS CONFIGURABLES
# =============================================================================

BEAM_DIAMETER_UM = 125.0   # Diámetro real de la punta de la fibra en micrómetros

# Parámetros de la cámara
EXPOSURE_TIME_US = 10000  # Tiempo de exposición en microsegundos
GAIN_DB          = 10.0        # Ganancia en dB

SAVE_FIGURE = None             # Ej: "resultado.png" para guardar, o None para no guardar
SHOW_FIGURE = True             # True para mostrar la figura interactiva




# =============================================================================
# DIÁLOGO DE CONFIGURACIÓN DE CALIBRACIÓN
# =============================================================================

class CalibrationSetupDialog:
    """
    Diálogo previo a la calibración que permite al usuario ajustar:
      - Diametro real del haz [um]
      - Exposicion de la camara [us]
      - Ganancia de la camara [dB]
    """

    def __init__(self, beam_um: float, exposure_us: float, gain_db: float):
        self.beam_um     = beam_um
        self.exposure_us = exposure_us
        self.gain_db     = gain_db
        self.confirmed   = False

    def show(self) -> bool:
        """Abre el diálogo y bloquea hasta confirmar o cancelar. Retorna
        True si el usuario aceptó, dejando los valores validados en
        `self.beam_um`, `self.exposure_us` y `self.gain_db`."""
        self.root = tk.Tk()
        self.root.title("Configuracion de calibracion de pixel")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        pad = {"padx": 10, "pady": 5, "sticky": "w"}

        ttk.Label(self.root,
                  text="Parametros de calibracion del tamano de pixel",
                  font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(12, 6), sticky="w")

        # Diametro del haz
        ttk.Label(self.root,
                  text="Diametro real del haz / fibra (um):").grid(
            row=1, column=0, **pad)
        self._beam_var = tk.StringVar(value=str(self.beam_um))
        ttk.Entry(self.root, textvariable=self._beam_var, width=12).grid(
            row=1, column=1, **pad)

        # Exposicion
        ttk.Label(self.root,
                  text="Exposicion de la camara (us):").grid(
            row=2, column=0, **pad)
        self._exp_var = tk.StringVar(value=str(self.exposure_us))
        ttk.Entry(self.root, textvariable=self._exp_var, width=12).grid(
            row=2, column=1, **pad)

        # Ganancia
        ttk.Label(self.root,
                  text="Ganancia de la camara (dB):").grid(
            row=3, column=0, **pad)
        self._gain_var = tk.StringVar(value=str(self.gain_db))
        ttk.Entry(self.root, textvariable=self._gain_var, width=12).grid(
            row=3, column=1, **pad)

        ttk.Label(self.root,
                  text="  Ilumine la punta de la fibra antes de presionar 'Iniciar calibracion'.",
                  foreground="gray", font=("", 8, "italic")).grid(
            row=4, column=0, columnspan=2, padx=10, pady=(8, 2), sticky="w")

        # Botones
        bf = ttk.Frame(self.root)
        bf.grid(row=5, column=0, columnspan=2, sticky="ew")
        crear_botones_confirmar_cancelar(
            bf, "Iniciar calibracion", self._confirm, self._cancel)

        self.root.update_idletasks()
        w  = self.root.winfo_width()
        h  = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")
        self.root.mainloop()
        return self.confirmed

    def _confirm(self):
        """
        Valida los tres campos antes de cerrar. El diámetro debe ser mayor
        que 0 porque es el patrón de longitud del que se despeja µm/px: un
        valor nulo o negativo produciría una escala sin sentido físico que
        se propagaría a TODAS las magnitudes en µm del proyecto.
        """
        try:
            self.beam_um     = float(self._beam_var.get().replace(",", "."))
            self.exposure_us = float(self._exp_var.get().replace(",", "."))
            self.gain_db     = float(self._gain_var.get().replace(",", "."))
            if self.beam_um <= 0:
                raise ValueError("El diametro debe ser mayor que 0.")
            if self.exposure_us <= 0:
                raise ValueError("La exposicion debe ser mayor que 0.")
            self.confirmed = True
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error de validacion", str(e), parent=self.root)

    def _cancel(self):
        self.root.destroy()


# =============================================================================
# ADQUISICION DE IMAGEN
# =============================================================================

def capture_frame_from_camera(
    exposure_us: float = EXPOSURE_TIME_US,
    gain_db: float = GAIN_DB
) -> np.ndarray:
    """
    Abre la camara Allied Vision, muestra streaming en vivo (usando
    gui.live_view) y espera a que el usuario presione ESPACIO para
    capturar un frame.

    Retorna la imagen capturada como array numpy en escala de grises,
    o None si el usuario cancelo.
    """
    import os, sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from gui.live_view import mostrar_captura_espacio

    captured_image = None

    with vmbpy.VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("No se encontro ninguna camara Allied Vision conectada.")

        with cams[0] as cam:
            print(f"  Camara: {cam.get_model()}")
            try:
                cam.ExposureAuto.set("Off")
                cam.ExposureTime.set(exposure_us)
                cam.Gain.set(gain_db)
                _pf = getattr(getattr(SESSION, "calibration", None), "pixel_format", "Mono8") if SESSION else "Mono8"
                if _pf:
                    try: cam.get_feature_by_name("PixelFormat").set(_pf)
                    except Exception as _e: print(f"PixelFormat: {_e}")
                print(f"  Exposicion: {exposure_us} us  |  Ganancia: {gain_db} dB")
            except Exception as e:
                print(f"  Advertencia al configurar camara: {e}")

            instrucciones = [
                "Ilumine la punta de la fibra optica.",
                "Asegurese de que este centrada y enfocada.",
            ]
            captured_image = mostrar_captura_espacio(
                cam,
                "Calibracion de pixel — Punta de fibra optica",
                instrucciones
            )

    return captured_image


# =============================================================================
# DETECCIÓN DEL CÍRCULO
# =============================================================================

def detect_beam_circle(gray: np.ndarray) -> tuple[float, float, float]:
    """
    Detecta el círculo de la punta de la fibra óptica en la imagen.

    Retorna: (cx, cy, radio) en píxeles.
    """

    # ── Umbralización de Otsu ──────────────────────────────────────────────────
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operaciones morfológicas para cerrar huecos y eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # ── Contornos ──────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No se encontraron contornos en la imagen capturada.")

    main_contour = max(contours, key=cv2.contourArea)

    # ── Círculo mínimo envolvente ──────────────────────────────────────────────
    (cx, cy), radius = cv2.minEnclosingCircle(main_contour)

    # ── Refinamiento con ajuste elíptico ──────────────────────────────────────
    if len(main_contour) >= 5:
        ellipse = cv2.fitEllipse(main_contour)
        r_avg = (ellipse[1][0] + ellipse[1][1]) / 4.0
        radius = (radius + r_avg) / 2.0
        cx, cy = ellipse[0]

    print(f"Centro detectado : ({cx:.1f}, {cy:.1f}) px")
    print(f"Radio detectado  : {radius:.1f} px  →  Diámetro: {2*radius:.1f} px")

    return float(cx), float(cy), float(radius)


# =============================================================================
# CALIBRACIÓN
# =============================================================================

def compute_pixel_size(radius_px: float, beam_diameter_um: float) -> float:
    """Calcula el tamaño real de un píxel en µm/px."""
    if radius_px <= 0:
        raise ValueError(
            f"Radio detectado inválido ({radius_px} px) — no se detectó un círculo "
            "válido en la imagen de calibración. Repite la captura con mejor "
            "iluminación/enfoque.")
    return beam_diameter_um / (2.0 * radius_px)


def print_report(cx, cy, radius, um_per_px, img_shape, beam_diameter_um=BEAM_DIAMETER_UM):
    """Imprime un resumen con los resultados de calibración."""
    w, h = img_shape[1], img_shape[0]
    print_banner("RESULTADO DE CALIBRACIÓN")
    print(f"  Diámetro real del haz     : {beam_diameter_um:.1f} µm")
    print(f"  Diámetro detectado        : {2*radius:.2f} px")
    print(f"  Escala: 1 px              = {um_per_px:.5f} µm")
    print(f"  Escala: 1 µm              = {1/um_per_px:.4f} px")
    print(f"  Campo de visión (imagen)  : {w*um_per_px:.1f} x {h*um_per_px:.1f} µm")
    print(f"  Centro del haz            : ({cx:.1f}, {cy:.1f}) px")
    print()


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def visualize_result(
    gray: np.ndarray,
    cx: float, cy: float, radius: float,
    um_per_px: float,
    save_path: str = None
) -> None:
    """Genera figura de diagnóstico con el círculo detectado superpuesto."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Calibración de píxel — Punta de fibra óptica", fontsize=14, fontweight="bold")

    # ── Imagen completa con círculo ───────────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax0.add_patch(plt.Circle((cx, cy), radius, color="cyan", fill=False, linewidth=2))
    ax0.plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=2)
    ax0.set_title(
        f"Haz detectado\nCentro=({cx:.0f},{cy:.0f}) px | Ø={2*radius:.1f} px",
        fontsize=10
    )
    ax0.set_xlabel("x [px]")
    ax0.set_ylabel("y [px]")

    # ── Zoom alrededor del haz ────────────────────────────────────────────────
    ax1 = axes[1]
    margin = int(radius * 1.4)
    x0 = max(0, int(cx - margin));  x1 = min(gray.shape[1], int(cx + margin))
    y0 = max(0, int(cy - margin));  y1 = min(gray.shape[0], int(cy + margin))
    crop = gray[y0:y1, x0:x1]

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    cv2.circle(crop_rgb, (int(cx - x0), int(cy - y0)), int(radius), (0, 255, 255), 2)
    cv2.drawMarker(crop_rgb, (int(cx - x0), int(cy - y0)),
                   (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

    ax1.imshow(crop_rgb)
    ax1.set_title(
        f"Zoom del haz\n{um_per_px:.4f} µm/px  |  Ø real = {2*radius*um_per_px:.1f} µm",
        fontsize=10
    )
    ax1.set_xlabel("x [px]")
    ax1.set_ylabel("y [px]")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")

    plt.show()


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def calibrate_pixel_size(
    beam_diameter_um: float = None,
    exposure_us: float = None,
    gain_db: float = None,
    save_figure: str = None,
    show_figure: bool = None,
    session=None,
) -> float | None:
    """
    Función principal: captura una imagen en vivo, detecta la punta
    de la fibra óptica y retorna µm/px.

    Cuando se llama desde el inicializador (main.py), `session` es un
    SessionConfig ya configurado: los parámetros se leen de session.calibration
    y el resultado se propaga automáticamente a session.apply_um_per_px().

    Cuando se ejecuta de forma independiente, usa los argumentos o las
    constantes globales definidas al inicio del script.

    Parámetros
    ----------
    beam_diameter_um : Diámetro real de la punta en micrómetros.
    exposure_us      : Tiempo de exposición de la cámara en µs.
    gain_db          : Ganancia de la cámara en dB.
    save_figure      : Ruta para guardar la figura diagnóstica (None = no guardar).
    show_figure      : Si True, muestra la figura interactiva tras la calibración.
    session          : SessionConfig del inicializador (opcional).

    Retorna
    -------
    um_per_px : float  — Tamaño real de cada píxel en µm.
                None   — Si el usuario salió sin capturar.
    """
    global SESSION
    if session is not None:
        SESSION = session

    # Resolver parámetros: argumento explícito > session > global
    _beam  = beam_diameter_um if beam_diameter_um is not None else _cal('beam_diameter_um')
    _exp   = exposure_us      if exposure_us      is not None else _cal('exposure_us')
    _gain  = gain_db          if gain_db          is not None else _cal('gain_db')
    _save  = save_figure      if save_figure      is not None else _cal('save_figure')
    _show  = show_figure      if show_figure      is not None else _cal('show_figure')

    # 0. Dialogo de configuracion previo a la captura
    dlg = CalibrationSetupDialog(
        beam_um=_beam, exposure_us=_exp, gain_db=_gain)
    if not dlg.show():
        print("Calibracion cancelada desde el dialogo de configuracion.")
        return None
    _beam = dlg.beam_um
    _exp  = dlg.exposure_us
    _gain = dlg.gain_db
    # Propagar valores actualizados a SessionConfig si existe
    if SESSION is not None:
        SESSION.calibration.beam_diameter_um = _beam
        SESSION.calibration.exposure_us      = _exp
        SESSION.calibration.gain_db          = _gain

    # 1. Captura en vivo
    gray = capture_frame_from_camera(exposure_us=_exp, gain_db=_gain)

    if gray is None:
        print("No se capturó ninguna imagen. Calibración cancelada.")
        return None

    # 2. Detección del haz
    cx, cy, radius = detect_beam_circle(gray)

    # 3. Cálculo de escala
    try:
        um_per_px = compute_pixel_size(radius, _beam)
    except ValueError as e:
        print(f"Error de calibración: {e}")
        return None

    # 4. Reporte
    print_report(cx, cy, radius, um_per_px, gray.shape, beam_diameter_um=_beam)

    # 5. Visualización
    if _show:
        visualize_result(gray, cx, cy, radius, um_per_px, save_path=_save)

    # 6. Propagar resultado al SessionConfig si viene del orquestador
    if SESSION is not None:
        SESSION.apply_um_per_px(um_per_px)
        print_ok(f"Tamaño de píxel propagado a SessionConfig: {um_per_px:.5f} µm/px")

    return um_per_px


# =============================================================================
# EJECUCIÓN
# =============================================================================

def main(session=None):
    """
    Punto de entrada.

    `session` (SessionConfig): si se pasa desde el inicializador, los parámetros
    se leen de session.calibration y el resultado se propaga a session.
    Retorna um_per_px (float) o None si se cancela.
    """
    um_per_px = calibrate_pixel_size(session=session)
    return um_per_px


if __name__ == "__main__":
    main()