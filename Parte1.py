# -*- coding: utf-8 -*-
"""
Sistema Automatizado de Captura de Imágenes y Video
con Control de Polarización MPC320, Cámara Alvium y Switch Óptico 4x1

Descripción:
Automatiza la adquisición de imágenes en hasta 4 etapas (cada una activable
de forma independiente en STAGES_ENABLED):
0) Dark Frame: Foto con el láser apagado 
1) Estabilidad temporal: Fotos y/o video en una misma polarización
2) Diferentes polarizaciones (haz): Fotos en diferentes polarizaciones
3) Diferentes polarizaciones (haz+fibra): Fotos en las mismas polarizaciones

Metodología:
1) Diálogo gráfico para elegir la carpeta destino.
2) Genera configuraciones de polarización.
3) Inicializa switch óptico, polarizador y cámara.
4) [Vista en vivo] Confirmación del usuario antes de iniciar.
5) ETAPA 0: Captura dark frame (láser apagado)            [si está habilitada]
6) [Vista en vivo] Confirmación para avanzar a ETAPA 1.
7) ETAPA 1: Captura imágenes y/o video en polarización fija [si está habilitada]
8) [Vista en vivo] Confirmación para avanzar a ETAPA 2.
9) ETAPA 2: Captura imágenes en diferentes polarizaciones  [si está habilitada]
10) [Vista en vivo] Confirmación para avanzar a ETAPA 3.
11) ETAPA 3: Captura imágenes (haz+fibra)                  [si está habilitada]
12) Guarda imágenes en subcarpetas organizadas y genera reporte completo.
13) Retorna el sistema a home y cierra dispositivos.

Autor: Diego Aguilar
"""

import vmbpy
import cv2
import numpy as np
import os
import json
import time
import clr
import sys
import nidaqmx
from nidaqmx.constants import LineGrouping
from datetime import datetime

import utils_carpetas as _utils_carpetas
import utils_imagenes as _utils_imagenes
from console_ui import print_banner, print_seccion, print_ok, print_error, print_warn, print_skip

# GUI unificada de vista en vivo
import os as _os_p1, sys as _sys_p1
_LV_ROOT_P1 = _os_p1.path.dirname(_os_p1.path.dirname(_os_p1.path.abspath(__file__)))
if _LV_ROOT_P1 not in _sys_p1.path: _sys_p1.path.insert(0, _LV_ROOT_P1)
from gui.live_view import mostrar_vista_en_vivo as _lv_p1, _ascii as _lv_ascii

# ── Selector de carpeta destino ────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox
# ──────────────────────────────────────────────────────────────────────────────

# ── Dialogos de raiz de dispositivo + caso (Sin turbulencia/Transitorio/Con turbulencia)
from gui.inicio_adquisicion_dialog import (
    preguntar_primera_vez, pedir_carpeta_raiz_existente, pedir_caso_unico)
# ──────────────────────────────────────────────────────────────────────────────

# ── Integración con SessionConfig (inicializador main.py) ─────────────────────
# SESSION se inyecta desde modules/acquisition.py cuando se ejecuta main.py.
# Si el script se corre de forma independiente (`python Parte1.py`), SESSION
# queda en None y el script usa los diccionarios de configuración manual
# definidos más abajo (EXPERIMENT_CONFIG, CAMERA_CONFIG, etc.).
#
# Los accesores `_exp`/`_cam`/`_stage`/... encapsulan exactamente esa
# bifurcación: cada uno lee un sub-objeto de la sesión o su diccionario
# equivalente. Gracias a ellos, el resto del archivo se escribe una sola vez
# y funciona igual en ambos modos de ejecución. Los `_filename_*` hacen lo
# mismo para la convención de nombres de archivo.
#
# ⚠ Al añadir un parámetro configurable nuevo hay que actualizar AMBOS lados
# (dataclass en config.py y diccionario aquí), o el modo standalone lanzará
# KeyError.
SESSION = None

def _exp(key):
    return SESSION.experiment.__dict__[key] if SESSION else EXPERIMENT_CONFIG[key]

def _cam(key):
    return SESSION.camera.__dict__[key] if SESSION else CAMERA_CONFIG[key]

def _stage(key):
    return SESSION.stages.__dict__[key] if SESSION else STAGES_ENABLED[key]

def _dark(key):
    return SESSION.dark_frame.__dict__[key] if SESSION else DARK_FRAME_CONFIG[key]

def _temp(key):
    return SESSION.temporal.__dict__[key] if SESSION else TEMPORAL_STABILITY_CONFIG[key]

def _scale(key):
    return SESSION.scale.__dict__[key] if SESSION else SCALE_CONFIG[key]

def _switch(key):
    return SESSION.switch.__dict__[key] if SESSION else OPTICAL_SWITCH_CONFIG[key]

def _pol_serial():
    return SESSION.polarizer.serial if SESSION else POLARIZER_SERIAL

def _overlay_text():
    return SESSION.overlay.text if SESSION else IMAGE_OVERLAY_TEXT

def _pol_positions():
    """Retorna las posiciones de polarización como dict pos1..pos10."""
    if SESSION:
        return {f'pos{i+1}': p for i, p in enumerate(SESSION.polarization.positions)}
    return POLARIZATION_POSITIONS


def _pol_enabled_indices():
    """
    Retorna lista de índices 1-based de las posiciones habilitadas.
    P01 (índice 1) siempre incluida.
    """
    if SESSION:
        enabled = getattr(SESSION.polarization, 'enabled_positions', None)
        if enabled:
            return [i + 1 for i, en in enumerate(enabled) if en]
    # Si no hay sesión o no hay campo enabled_positions → todas activas
    return list(range(1, 11))

def _filename_haz(pos_idx: int) -> str:
    if SESSION:
        return SESSION.filename_haz(SESSION.experiment.channel, pos_idx)
    return f"MMI_Haz_{EXPERIMENT_CONFIG['channel']}_P{pos_idx:02d}"

def _filename_haz_fibra(pos_idx: int) -> str:
    if SESSION:
        return SESSION.filename_haz_fibra(SESSION.experiment.channel, pos_idx)
    return f"MMI_Haz_Fibra_{EXPERIMENT_CONFIG['channel']}_P{pos_idx:02d}"

def _filename_estabilidad(i: int) -> str:
    if SESSION:
        return SESSION.filename_estabilidad(SESSION.experiment.channel, i)
    return f"MMI_Estabilidad_{EXPERIMENT_CONFIG['channel']}_T{i:02d}"

def _filename_video() -> str:
    if SESSION:
        return SESSION.filename_video(SESSION.experiment.channel)
    return f"{TEMPORAL_STABILITY_CONFIG['video_filename']}_{EXPERIMENT_CONFIG['channel']}"

def _filename_darkframe() -> str:
    if SESSION:
        return SESSION.dark_frame.filename   # siempre "darkframe"
    return f"{DARK_FRAME_CONFIG['filename']}_{EXPERIMENT_CONFIG['channel']}"
# ──────────────────────────────────────────────────────────────────────────────

# Referencias de Thorlabs
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.PolarizerCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.PolarizerCLI import *
from System import Decimal

# =============================================================================
# CONFIGURACIÓN MANUAL DEL EXPERIMENTO
# =============================================================================

EXPERIMENT_CONFIG = {
    'experiment_id': 'MMI-EXP001',
    # 'device_name' nombra la carpeta raíz del experimento:
    #   YYYYMMDD_CaracterizacionHaz_<device_name>
    'device_name': 'MMI',
    'conditions': 'longitud de la fibra NFC:: 11600 um',
    'wavelength': '980 nm',
    'temperature': '23.01',
    'current': '61.33',
    'channel': 'CH01'
}

# =============================================================================
# ACTIVACIÓN / DESACTIVACIÓN DE ETAPAS
# =============================================================================
STAGES_ENABLED = {
    'stage0_dark_frame':             True,
    'stage1_temporal_stability':     True,
    'stage2_polarization_haz':       True,
    'stage3_polarization_haz_fibra': True,
}

# =============================================================================
# CONFIGURACIÓN DE ETAPA 0 (DARK FRAME)
# =============================================================================
# subfolder_name debe coincidir con utils_carpetas.NOMBRE_DARKFRAME y con
# config.DarkFrameConfig.subfolder_name -- la carpeta real la crea
# utils_carpetas.crear_carpetas_caso() con esa constante, no este dict.
DARK_FRAME_CONFIG = {
    'subfolder_name': 'darkframe',
    'filename': 'dark_frame',
    'paddle1': 80,
    'paddle2': 80,
    'paddle3': 80,
}

# =============================================================================
# CONFIGURACIÓN DE ETAPA 1
# =============================================================================
TEMPORAL_STABILITY_CONFIG = {
    'paddle1': 80,
    'paddle2': 80,
    'paddle3': 80,
    'capture_photos': True,
    'capture_video': True,
    'num_images': 1,
    'time_interval': 1.0,
    'log_all_metadata': False,
    'video_duration': 30,
    'video_codec': 'mp4v',
    'video_format': 'mp4',
    'video_filename': 'MMI_Estabilidad_CH01_Video',
    'subfolder_name': 'estabilidad_temporal'
}

# =============================================================================
# CONFIGURACIÓN DE POLARIZACIONES (ETAPAS 2 y 3)
# =============================================================================
POLARIZATION_CONFIG = {
    'subfolder_haz': 'diferentes_polarizaciones_haz',
    'subfolder_haz_fibra': 'diferentes_polarizaciones_haz_fibra'
}

# =============================================================================
# CONFIGURACIÓN DEL SWITCH ÓPTICO
# =============================================================================
OPTICAL_SWITCH_CONFIG = {
    'device_name': 'Dev1',
    'port': 0,
    'lines': [0, 1], #[MSB,LSB]
    'channel': 1
}

# =============================================================================
# CONFIGURACIÓN DE LA CÁMARA
# =============================================================================
CAMERA_CONFIG = {
    'exposure_time': 10000,
    'gain': 10,
    'exposure_auto': 'Off',
    'pixel_format': 'Mono8',
    'video_exposure_time': None,
    'video_gain': None,
    'guardar_npz': True,
}

# =============================================================================
# CONFIGURACIÓN DEL POLARIZADOR
# =============================================================================
POLARIZER_SERIAL = "38388714"

# =============================================================================
# ANOTACIÓN EN IMÁGENES
# =============================================================================
IMAGE_OVERLAY_TEXT = 'Longitud de fibra NCF: 11600 um'

# =============================================================================
# CONFIGURACIÓN DE ESCALA
# =============================================================================
SCALE_CONFIG = {
    'enabled': True,
    'um_per_px': 2.2,
    'px_per_um': 0.4545,
    'bar_length_um': 100,
    'bar_height_px': 16,
    'margin': 35,
    'font_scale': 0.55,
    'font_thickness': 1,
    'color': (255, 255, 255),
    'outline_color': (0, 0, 0),
}

# ======================================================================


def generate_polarization_positions():
    """Genera 10 posiciones de polarización fijas"""
    positions = {
        'pos1':  {'paddle1': 0,   'paddle2': 0,   'paddle3': 0},
        'pos2':  {'paddle1': 80,  'paddle2': 80,  'paddle3': 80},
        'pos3':  {'paddle1': 160, 'paddle2': 160, 'paddle3': 160},
        'pos4':  {'paddle1': 105, 'paddle2': 30,  'paddle3': 128},
        'pos5':  {'paddle1': 26,  'paddle2': 17,  'paddle3': 154},
        'pos6':  {'paddle1': 139, 'paddle2': 91,  'paddle3': 113},
        'pos7':  {'paddle1': 50,  'paddle2': 44,  'paddle3': 68},
        'pos8':  {'paddle1': 77,  'paddle2': 55,  'paddle3': 130},
        'pos9':  {'paddle1': 71,  'paddle2': 8,   'paddle3': 16},
        'pos10': {'paddle1': 64,  'paddle2': 25,  'paddle3': 76}
    }
    return positions


POLARIZATION_POSITIONS = generate_polarization_positions()


# =============================================================================
# SELECTOR DE CARPETA DESTINO
# =============================================================================

def ask_destination_folder():
    """
    Abre un diálogo gráfico para que el usuario elija la carpeta destino
    donde se creará la carpeta principal del experimento.

    La carpeta final se construirá como:
        <ruta_elegida> / YYYYMMDD_CaracterizacionHaz_<device_name>

    Si el usuario cancela el diálogo, retorna None (el llamador decide
    cómo manejarlo — nunca debe terminar el proceso, ya que este script
    puede estar corriendo dentro de main.py y debe poder volver al menú).

    Returns
    -------
    str | None
        Ruta absoluta de la carpeta destino elegida por el usuario, o
        None si canceló.
    """
    root = tk.Tk()
    root.withdraw()                      # Ocultar ventana raíz de tkinter
    root.attributes('-topmost', True)    # Diálogo siempre al frente

    # Ejemplo del nombre que se creará
    date_prefix  = datetime.now().strftime('%Y%m%d')
    example_name = f"{date_prefix}_CaracterizacionHaz_{_exp('device_name')}"

    messagebox.showinfo(
        title="Selección de carpeta destino",
        message=(
            f"Seleccione la carpeta donde se guardará el experimento.\n\n"
            f"Se creará automáticamente la subcarpeta:\n"
            f"  {example_name}\n\n"
            f"Haga clic en OK para abrir el selector de carpeta."
        )
    )

    destination = filedialog.askdirectory(
        title="Seleccionar carpeta destino del experimento",
        mustexist=True
    )
    root.destroy()

    if not destination:
        print()
        print_error("No se seleccionó ninguna carpeta destino. "
                    "El experimento se canceló.")
        print()
        return None

    print_ok(f"Carpeta destino seleccionada: {destination}")
    print()
    return destination


def resolver_destino_y_caso(session=None):
    """
    Flujo de inicio de la adquisición: resuelve la carpeta RAÍZ del
    dispositivo (nueva o reutilizada) y el caso de turbulencia a
    adquirir en esta corrida.

    1) Pregunta si es la primera vez con este dispositivo:
       - Sí  → pide la carpeta destino (ask_destination_folder) y crea
               una raíz nueva, protegida contra sobreescritura.
       - No  → pide la carpeta raíz ya existente.
    2) Pregunta qué caso se va a adquirir (Sin turbulencia / Transitorio
       / Con turbulencia — selección única).
    3) Calcula la carpeta del caso, también protegida contra
       sobreescritura (si el caso ya existía en esta raíz, se agrega un
       sufijo numérico en vez de reutilizarla).

    Retorna (carpeta_raiz_dispositivo, caso, carpeta_caso), o None si el
    usuario cancela en cualquier paso.
    """
    primera_vez = preguntar_primera_vez()
    if primera_vez is None:
        return None

    if primera_vez:
        destino_padre = ask_destination_folder()
        if destino_padre is None:
            return None
        date_prefix = datetime.now().strftime('%Y%m%d')
        base_name   = f"{date_prefix}_CaracterizacionHaz_{_exp('device_name')}"
        carpeta_raiz = _utils_carpetas.carpeta_raiz_segura(destino_padre, base_name)
    else:
        carpeta_raiz = pedir_carpeta_raiz_existente(device_name_esperado=_exp('device_name'))
        if carpeta_raiz is None:
            print_error("No se seleccionó ninguna carpeta raíz. El experimento se canceló.")
            return None

    caso = pedir_caso_unico(carpeta_raiz=carpeta_raiz)
    if caso is None:
        print_error("No se seleccionó ningún caso. El experimento se canceló.")
        return None

    carpeta_caso = _utils_carpetas.carpeta_raiz_segura(carpeta_raiz, caso)
    return carpeta_raiz, caso, carpeta_caso


# =============================================================================
# UTILIDADES
# =============================================================================

def get_python_environment_info():
    """Obtiene información del entorno de Python y librerías utilizadas"""
    env_info = {}
    env_info['python_version'] = sys.version.split()[0]
    env_info['python_full'] = sys.version.replace('\n', ' ')
    try:
        env_info['opencv_version'] = cv2.__version__
    except:
        env_info['opencv_version'] = 'N/A'
    try:
        env_info['numpy_version'] = np.__version__
    except:
        env_info['numpy_version'] = 'N/A'
    try:
        env_info['vmbpy_version'] = vmbpy.__version__
    except:
        env_info['vmbpy_version'] = 'N/A'
    env_info['script_name'] = os.path.basename(__file__)
    env_info['platform'] = sys.platform
    return env_info


# =============================================================================
# CONTROLADOR DEL SWITCH ÓPTICO
# =============================================================================

class OpticalSwitchController:
    """
    Controlador para switch óptico de 4 canales usando NI USB-6001.
    Utiliza 2 salidas digitales para representar canales 1-4 en binario.
    """

    def __init__(self, device_name='Dev1', port=0, lines=None):
        self.device_name = device_name
        self.port = port
        self.lines = lines if lines is not None else [0, 1]
        self.task = None
        self.current_channel = None

    def connect(self):
        """
        Abre la tarea NI-DAQ sobre las 2 líneas digitales configuradas,
        agrupadas como un solo canal (`CHAN_FOR_ALL_LINES`) para poder
        escribir el número de canal como un entero en vez de bit a bit.

        Retorna True/False en lugar de lanzar excepción: el switch es
        hardware OPCIONAL y el llamador continúa el experimento con una
        advertencia si no está disponible.
        """
        try:
            self.task = nidaqmx.Task()
            line_string = (
                f"{self.device_name}/port{self.port}"
                f"/line{self.lines[0]}:{self.lines[1]}"
            )
            self.task.do_channels.add_do_chan(
                line_string,
                line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
            )
            self.task.start()
            print_ok(f"Switch óptico conectado en {line_string}")
            return True
        except Exception as e:
            print_error(f"Error al conectar switch óptico: {e}")
            return False

    def set_channel(self, channel):
        """
        Selecciona uno de los 4 canales (1-4) escribiendo `channel-1`
        (0-3) como patrón de 2 bits sobre las 2 líneas digitales
        agrupadas (`self.lines`, [MSB,LSB] según el comentario de
        OPTICAL_SWITCH_CONFIG). El orden exacto de bit (cuál línea es
        MSB y cuál LSB) tiene referencias contradictorias sin verificar
        entre archivos del proyecto -- ver config.py::OpticalSwitchConfig
        y docs/DeveloperNotes.md §6 ("posible inversión MSB/LSB"),
        pendiente de confirmar con hardware real. No se resuelve aquí.
        """
        if channel < 1 or channel > 4:
            raise ValueError("El canal debe estar entre 1 y 4")
        binary_value = channel - 1
        bit0 = binary_value & 0b01
        bit1 = (binary_value & 0b10) >> 1
        self.task.write(binary_value)
        self.current_channel = channel
        print_ok(f"Canal {channel} del switch activado (Binario: {bit1}{bit0}, Valor: {binary_value})")

    def disconnect(self):
        """Detiene y cierra la tarea NI-DAQ, liberando las líneas
        digitales. Los errores solo se advierten: dejar la tarea abierta
        impediría que el siguiente experimento tomara el dispositivo."""
        if self.task:
            try:
                self.task.stop()
                self.task.close()
                print_ok("Switch óptico desconectado")
            except Exception as e:
                print_warn(f"Error al desconectar switch óptico: {e}")

    def __enter__(self):
        """Permite usar el controlador como context manager, garantizando
        la desconexión del DAQ aunque el bloque falle."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# =============================================================================
# CONTROLADOR PRINCIPAL
# =============================================================================

class OpticalSystemController:
    """Controlador maestro para el sistema óptico"""

    def __init__(self, overlay_text='', device_root_folder=None, caso=None,
                 carpeta_caso=None):
        self.polarizer = None
        self.camera = None
        self.vmb = None
        self.optical_switch = None
        self.device_root_folder = device_root_folder
        self.caso = caso
        self.carpeta_caso = carpeta_caso
        self.save_folder = None
        self.adquisicion_folder = None
        self.subfolder_dark_frame = None
        self.subfolder_temporal = None
        self.subfolder_haz = None
        self.subfolder_haz_fibra = None
        self.captured_image = None
        self.camera_model = None
        self.polarizer_model = None
        self.image_metadata = {}
        self.overlay_text = overlay_text

    # ──────────────────────────────────────────────────────────────────────────
    # VISTA EN VIVO CON CONFIRMACIÓN
    # ──────────────────────────────────────────────────────────────────────────

    def show_live_preview_and_confirm(self, stage_title, instruction_lines=None):
        """
        Abre una ventana de vista en vivo (via gui.live_view) y espera
        confirmacion [S] o aborto [Q].
        Lanza KeyboardInterrupt si el usuario aborto.
        """
        if instruction_lines is None:
            instruction_lines = []
        ok = _lv_p1(self.camera, stage_title, instruction_lines)
        if not ok:
            raise KeyboardInterrupt("Experimento abortado por el usuario (tecla Q).")

    # ──────────────────────────────────────────────────────────────────────────
    # BARRA DE ESCALA
    # ──────────────────────────────────────────────────────────────────────────

    def _add_scale_bar(self, image):
        """
        Dibuja la barra de escala física en la esquina inferior izquierda,
        con su longitud en píxeles derivada de la calibración vigente
        (`bar_length_um * px_per_um`), de modo que representa una distancia
        real correcta en la imagen.

        Cada trazo se dibuja dos veces —primero un contorno negro más
        grueso, luego el relleno claro— para que siga siendo legible tanto
        sobre el fondo oscuro de la imagen como sobre el haz saturado.

        Trabaja sobre una COPIA: la imagen original nunca se modifica, lo
        que permite guardar el dato crudo sin anotaciones en el .npz.
        """
        if not _scale('enabled'):
            return image

        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w   = img.shape[:2]
        bar_um = _scale('bar_length_um')
        bar_px = round(bar_um * _scale('px_per_um'))
        bar_h  = _scale('bar_height_px')
        margin = _scale('margin')
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = _scale('font_scale')
        fthick = _scale('font_thickness')
        color  = _scale('color')
        outline = _scale('outline_color')

        label = f"{bar_um} um"
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)

        x1 = margin
        x2 = x1 + bar_px
        text_gap     = 4
        y_bar_bottom = h - margin
        y_bar_top    = y_bar_bottom - bar_h
        y_text       = y_bar_top - text_gap
        y_cap_top    = y_bar_top - 2
        y_cap_bottom = y_bar_bottom + 2

        def put_text_outlined(img, text, org):
            cv2.putText(img, text, org, font, fscale, outline, fthick + 2, cv2.LINE_AA)
            cv2.putText(img, text, org, font, fscale, color,   fthick,     cv2.LINE_AA)

        def put_line_outlined(img, pt1, pt2, thickness):
            cv2.line(img, pt1, pt2, outline, thickness + 2, cv2.LINE_AA)
            cv2.line(img, pt1, pt2, color,   thickness,     cv2.LINE_AA)

        y_bar_mid = (y_bar_top + y_bar_bottom) // 2
        put_line_outlined(img, (x1, y_bar_mid), (x2, y_bar_mid), bar_h)
        put_line_outlined(img, (x1, y_cap_top), (x1, y_cap_bottom), fthick + 1)
        put_line_outlined(img, (x2, y_cap_top), (x2, y_cap_bottom), fthick + 1)

        x_text = x1 + (bar_px - tw) // 2
        put_text_outlined(img, label, (x_text, y_text))
        return img

    # ──────────────────────────────────────────────────────────────────────────
    # OVERLAY DE TEXTO (esquina inferior derecha)
    # ──────────────────────────────────────────────────────────────────────────

    def _add_overlay_text(self, image):
        """
        Escribe el texto libre configurado por el usuario
        (`session.overlay.text`) en la esquina inferior derecha. Sirve para
        que la condición experimental quede legible dentro de la propia
        imagen, sin depender del JSON de metadatos.

        El texto se sanea a ASCII (`_lv_ascii`) antes de dibujarlo: las
        fuentes Hershey de OpenCV no soportan Unicode y las tildes o la ñ
        aparecerían como símbolos rotos.
        """
        overlay = _overlay_text() if SESSION else self.overlay_text
        if not overlay:
            return image

        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        font          = cv2.FONT_HERSHEY_SIMPLEX
        font_scale    = 1
        thickness     = 1
        color         = (255, 255, 255)
        outline_color = (0, 0, 0)

        overlay_safe = _lv_ascii(overlay)
        text_size, _ = cv2.getTextSize(overlay_safe, font, font_scale, thickness)
        text_w, text_h = text_size
        h, w = img.shape[:2]
        margin = 8
        x = w - text_w - margin
        y = h - margin

        cv2.putText(img, overlay_safe, (x, y), font, font_scale,
                    outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(img, overlay_safe, (x, y), font, font_scale,
                    color, thickness, cv2.LINE_AA)
        return img

    def _add_caso_dispositivo_overlay(self, image):
        """
        Escribe automáticamente "<caso> | <dispositivo>" en la esquina
        inferior derecha de TODA foto/video de esta adquisición — excepto
        el dark frame, que nunca lleva overlays (ver capture_image con
        apply_overlays=False, que nunca llega a _apply_all_overlays).
        Mismo tamaño de fuente que el overlay de texto configurable del
        usuario (_add_overlay_text), y se apila ARRIBA de ese overlay para
        no superponerse con él, esté activo o no.

        No se repite en el Preprocesado (Parte2.py) a propósito: el texto
        ya queda "quemado" en el píxel desde la adquisición, y volver a
        escribir texto sobre una imagen ya procesada podría interferir con
        el análisis cuantitativo (Opción 3).
        """
        caso_label  = _utils_carpetas.CASO_LABELS.get(self.caso, self.caso) if self.caso else None
        device_name = _exp('device_name')
        partes = [p for p in (caso_label, device_name) if p]
        if not partes:
            return image
        texto = "  |  ".join(partes)

        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        font          = cv2.FONT_HERSHEY_SIMPLEX
        font_scale    = 1        # mismo tamaño que el overlay de texto configurable
        thickness     = 1
        color         = (255, 255, 255)
        outline_color = (0, 0, 0)
        margin        = 8
        line_gap      = 6

        texto_safe = _lv_ascii(texto)
        (text_w, text_h), _ = cv2.getTextSize(texto_safe, font, font_scale, thickness)
        h, w = img.shape[:2]

        overlay_usuario = _overlay_text() if SESSION else self.overlay_text
        if overlay_usuario:
            (_, overlay_h), _ = cv2.getTextSize(_lv_ascii(overlay_usuario), font, font_scale, thickness)
            y = h - margin - overlay_h - line_gap
        else:
            y = h - margin
        x = w - text_w - margin

        cv2.putText(img, texto_safe, (x, y), font, font_scale,
                    outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(img, texto_safe, (x, y), font, font_scale,
                    color, thickness, cv2.LINE_AA)
        return img

    def _apply_all_overlays(self, image):
        """
        Compone las tres anotaciones sobre la imagen, en este orden fijo:
        texto del usuario y etiqueta caso|dispositivo (esquina inferior
        derecha, apiladas) y barra de escala (esquina inferior izquierda).

        El orden importa porque `_add_caso_dispositivo_overlay` consulta si
        existe el texto del usuario para colocarse justo encima y no
        solaparse con él.

        ⚠ Todo lo que se dibuja aquí queda QUEMADO permanentemente en el
        PNG/MP4 guardado. Esos píxeles no son señal óptica: si el ROI del
        preprocesado los incluyera, contaminarían centroide, ancho y
        energía en el análisis. Por eso `Parte2.py` verifica explícitamente
        que el ROI elegido no invada estas zonas, y por eso el dark frame y
        el respaldo .npz se guardan SIN pasar por esta función.
        """
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = self._add_overlay_text(img)
        img = self._add_caso_dispositivo_overlay(img)
        img = self._add_scale_bar(img)
        return img

    # ──────────────────────────────────────────────────────────────────────────
    # CARPETAS Y REPORTE
    # ──────────────────────────────────────────────────────────────────────────

    def create_experiment_folders(self):
        """
        Crea, dentro de la carpeta del CASO ya resuelta y protegida
        (self.carpeta_caso — ver resolver_destino_y_caso), la estructura
        estándar Adquisicion/Preprocesado/Analisis. `darkframe` queda
        ANIDADA dentro de Adquisicion (junto con estabilidad_temporal y
        polarizaciones): cada caso adquiere y guarda su propio dark
        frame, nunca compartido con otro caso del mismo dispositivo.

        La carpeta del caso ya viene desambiguada (sufijo numérico si el
        caso se repite dentro de la misma raíz de dispositivo), así que
        aquí no hace falta protección adicional a ese nivel.
        """
        self.save_folder = self.carpeta_caso
        (self.adquisicion_folder, _, _,
         self.subfolder_dark_frame) = _utils_carpetas.crear_carpetas_caso(
            self.save_folder)
        print_ok(f"Carpeta de caso creada: {self.save_folder}")

        if _stage('stage1_temporal_stability'):
            self.subfolder_temporal = os.path.join(
                self.adquisicion_folder, TEMPORAL_STABILITY_CONFIG['subfolder_name'])
            os.makedirs(self.subfolder_temporal, exist_ok=True)
            print_ok(f"Subcarpeta creada: {TEMPORAL_STABILITY_CONFIG['subfolder_name']}")

        if _stage('stage2_polarization_haz'):
            self.subfolder_haz = os.path.join(
                self.adquisicion_folder, POLARIZATION_CONFIG['subfolder_haz'])
            os.makedirs(self.subfolder_haz, exist_ok=True)
            print_ok(f"Subcarpeta creada: {POLARIZATION_CONFIG['subfolder_haz']}")

        if _stage('stage3_polarization_haz_fibra'):
            self.subfolder_haz_fibra = os.path.join(
                self.adquisicion_folder, POLARIZATION_CONFIG['subfolder_haz_fibra'])
            os.makedirs(self.subfolder_haz_fibra, exist_ok=True)
            print_ok(f"Subcarpeta creada: {POLARIZATION_CONFIG['subfolder_haz_fibra']}")

        print()


    def save_acquisition_metadata(self):
        """
        Guarda metadata_adquisicion.json dentro de la carpeta Adquisicion/
        del experimento. Permite que Parte2 y Parte3 trabajen de forma
        independiente, sin necesidad de rehacer la calibración ni conocer
        la sesión.
        """
        meta = {
            "fecha":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": _exp("experiment_id"),
            "device_name":   _exp("device_name"),
            "caso":                self.caso,
            "carpeta_dispositivo": self.device_root_folder,
            "channel":       _exp("channel"),
            "wavelength":    _exp("wavelength"),
            "conditions":    _exp("conditions"),
            "camara": {
                "exposure_time_us": _cam("exposure_time"),
                "gain_db":          _cam("gain"),
                "pixel_format":     _cam("pixel_format") if SESSION else "Mono8",
            },
            "calibracion": {
                "um_per_px":         SESSION.scale.um_per_px if SESSION else None,
                "beam_diameter_um":  SESSION.calibration.beam_diameter_um if SESSION else None,
            },
            "carpetas": {
                "dark_frame":            _dark("subfolder_name"),
                "estabilidad_temporal":  TEMPORAL_STABILITY_CONFIG["subfolder_name"],
                "polarizaciones_haz":    POLARIZATION_CONFIG["subfolder_haz"],
                "polarizaciones_fibra":  POLARIZATION_CONFIG["subfolder_haz_fibra"],
            },
            "video": {
                "duration_s":   _temp("video_duration"),
                "codec":        _temp("video_codec"),
                "format":       _temp("video_format"),
            },
        }
        ruta = os.path.join(self.adquisicion_folder, "metadata_adquisicion.json")
        with open(ruta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
        print_ok(f"metadata_adquisicion.json guardado en {self.adquisicion_folder}")

    def create_report_file(self):
        """
        Crea Reporte_Experimento.txt con el encabezado completo (info del
        experimento, etapas habilitadas, configuración de cámara/switch/
        escala, entorno de software). Las etapas van agregando sus propios
        bloques después vía add_*_to_report. Retorna `report_path`, que el
        llamador debe pasar a cada add_*_to_report/finalize_report.
        """
        report_path = os.path.join(self.adquisicion_folder, "Reporte_Experimento.txt")
        env_info    = get_python_environment_info()

        capture_modes = []
        if _temp('capture_photos'):
            capture_modes.append(
                f"Fotos ({_temp('num_images')} imágenes, "
                f"intervalo {_temp('time_interval')} s)")
        if _temp('capture_video'):
            capture_modes.append(
                f"Video ({_temp('video_duration')} s, "
                f"FPS medidos automáticamente, "
                f"codec {_temp('video_codec')}, "
                f"formato .{_temp('video_format')})")
        if not capture_modes:
            capture_modes.append("⚠ Ninguno (ambas opciones desactivadas)")

        bar_px = round(_scale('bar_length_um') * _scale('px_per_um'))

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"REPORTE DE EXPERIMENTO: {_exp('experiment_id')}\n")
            f.write("EXPERIMENTO EN 4 ETAPAS\n")
            f.write("="*80 + "\n\n")

            f.write("INFORMACIÓN DEL EXPERIMENTO:\n")
            f.write(f"  ID del Experimento: {_exp('experiment_id')}\n")
            f.write(f"  Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Carpeta principal: {self.save_folder}\n")
            f.write(f"  Caso: {_utils_carpetas.CASO_LABELS.get(self.caso, self.caso)}\n")
            f.write(f"  Carpeta raíz del dispositivo: {self.device_root_folder}\n")
            f.write(f"  Canal: {_exp('channel')}\n\n")

            f.write("CONDICIONES DEL EXPERIMENTO:\n")
            f.write(f"  {_exp('conditions')}\n\n")

            f.write("ESTRUCTURA DEL EXPERIMENTO:\n")
            f.write("  Etapas habilitadas:\n")
            for key, label in [
                ('stage0_dark_frame',             'Etapa 0 — Dark Frame'),
                ('stage1_temporal_stability',     'Etapa 1 — Estabilidad Temporal'),
                ('stage2_polarization_haz',       'Etapa 2 — Polarizaciones (Haz)'),
                ('stage3_polarization_haz_fibra', 'Etapa 3 — Polarizaciones (Haz+Fibra)'),
            ]:
                estado = 'ACTIVA ' if _stage(key) else 'OMITIDA'
                f.write(f"    [{estado}] {label}\n")
            f.write("\n")

            f.write("  ETAPA 0 - Dark Frame:\n")
            f.write("    • Captura con láser APAGADO (línea de base de ruido)\n")
            f.write(f"    • Polarización: P1={_dark('paddle1')}°, "
                    f"P2={_dark('paddle2')}°, "
                    f"P3={_dark('paddle3')}°\n")
            f.write(f"    • Subcarpeta: {_dark('subfolder_name')}\n\n")

            f.write("  ETAPA 1 - Estabilidad Temporal:\n")
            f.write(f"    • Modo(s) de captura: {' + '.join(capture_modes)}\n")
            f.write(f"    • Polarización fija: P1={_temp('paddle1')}°, "
                    f"P2={_temp('paddle2')}°, "
                    f"P3={_temp('paddle3')}°\n")
            if _temp('capture_photos'):
                log_mode = ("Todas las imágenes" if _temp('log_all_metadata')
                            else "Solo la primera imagen")
                f.write(f"    • Registro de metadatos (fotos): {log_mode}\n")
            f.write(f"    • Subcarpeta: {TEMPORAL_STABILITY_CONFIG['subfolder_name']}\n\n")

            f.write("  ETAPA 2 - Diferentes Polarizaciones (Haz):\n")
            f.write(f"    • Número de imágenes: {len(_pol_enabled_indices())}\n")
            f.write(f"    • Subcarpeta: {POLARIZATION_CONFIG['subfolder_haz']}\n\n")

            f.write("  ETAPA 3 - Diferentes Polarizaciones (Haz + Fibra):\n")
            f.write(f"    • Número de imágenes: {len(_pol_enabled_indices())}\n")
            f.write(f"    • Subcarpeta: {POLARIZATION_CONFIG['subfolder_haz_fibra']}\n\n")

            f.write("SWITCH ÓPTICO:\n")
            f.write(f"  - Dispositivo DAQ: {_switch('device_name')}\n")
            f.write(f"  - Puerto: {_switch('port')}\n")
            f.write(f"  - Canal seleccionado: {_switch('channel')}\n\n")

            if _overlay_text():
                f.write("ANOTACIÓN EN IMÁGENES:\n")
                f.write(f"  \"{_overlay_text()}\"\n\n")

            f.write("BARRA DE ESCALA:\n")
            f.write(f"  - Habilitada: {'Sí' if _scale('enabled') else 'No'}\n")
            if _scale('enabled'):
                f.write(f"  - Calibración: 1 px = {_scale('um_per_px')} µm  |  "
                        f"1 µm = {_scale('px_per_um')} px\n")
                f.write(f"  - Longitud representada: {_scale('bar_length_um')} µm "
                        f"({bar_px} px en imagen)\n")
                f.write("  - Posición: esquina inferior izquierda\n\n")

            f.write("SOFTWARE UTILIZADO:\n")
            f.write(f"  - Script: {env_info['script_name']}\n")
            f.write(f"  - Python: {env_info['python_version']}\n")
            f.write(f"  - OpenCV: {env_info['opencv_version']}\n")
            f.write(f"  - NumPy: {env_info['numpy_version']}\n")
            f.write(f"  - Vimba Python (vmbpy): {env_info['vmbpy_version']}\n")
            f.write(f"  - Plataforma: {env_info['platform']}\n\n")

            f.write("CONFIGURACIÓN DEL LÁSER:\n")
            f.write(f"  - Longitud de onda: {_exp('wavelength')}\n")
            f.write(f"  - Temperatura: {_exp('temperature')}°C\n")
            f.write(f"  - Corriente: {_exp('current')} mA\n\n")

            f.write("DISPOSITIVOS UTILIZADOS:\n")
            if self.camera_model:
                f.write(f"  - Cámara: {self.camera_model}\n")
            if self.polarizer_model:
                f.write(f"  - Polarizador: {self.polarizer_model}\n")
            f.write("\n")

            f.write("CONFIGURACIÓN DE LA CÁMARA:\n")
            f.write(f"  - Tiempo de exposición: {_cam('exposure_time')} µs\n")
            f.write(f"  - Ganancia: {_cam('gain')} dB\n")
            f.write(f"  - Modo de exposición automática: {_cam('exposure_auto')}\n")
            if _temp('capture_video'):
                exp_v  = _cam('video_exposure_time') or _cam('exposure_time')
                gain_v = _cam('video_gain')          or _cam('gain')
                f.write(f"  [Video] Exposición: {exp_v} µs | Ganancia: {gain_v} dB\n")
                f.write("  [Video] FPS: medidos automáticamente durante la grabación\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("DETALLES DE CAPTURA POR ETAPA\n")
            f.write("="*80 + "\n\n")

        print_ok("Archivo de reporte creado: Reporte_Experimento.txt")
        print()
        return report_path

    # ── Registro incremental del reporte ──────────────────────────────────────
    # Los métodos `add_*_to_report` van ANEXANDO (modo 'a') al archivo creado
    # por create_report_file, uno por captura, en vez de acumular en memoria y
    # escribir al final. Así, si el experimento se interrumpe a mitad —fallo
    # de hardware, aborto del operador—, el reporte conserva todo lo ya
    # capturado hasta ese punto en vez de perderse por completo.
    #
    # Cada método registra los parámetros específicos de su etapa
    # (polarización aplicada, timestamps, exposición/ganancia reales leídas
    # de la cámara), de modo que el .txt final permite reconstruir en qué
    # condiciones exactas se tomó cada archivo del experimento.

    def add_stage_header_to_report(self, report_path, stage_number, stage_name):
        """Escribe el separador y título de una etapa en el reporte."""
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"ETAPA {stage_number}: {stage_name.upper()}\n")
            f.write("="*80 + "\n\n")

    def get_image_metadata(self):
        """
        Lee del hardware los parámetros REALES con los que se está
        capturando (exposición, ganancia, formato de píxel, dimensiones
        del sensor), en lugar de asumir los valores configurados.

        Importa para la trazabilidad: si la cámara no aceptó un valor
        solicitado —por estar fuera de rango o no soportarlo— el reporte
        registra lo que de verdad se aplicó, no lo que se pidió. Cada
        atributo se lee por separado y cae a 'N/A' si el modelo concreto
        de cámara no lo expone.
        """
        metadata = {}
        try:
            metadata['camera_model'] = self.camera.get_model()
            metadata['camera_id']    = self.camera.get_id()
            for attr, key in [
                ('ExposureTime', 'exposure_time_us'),
                ('Gain',         'gain_db'),
                ('Width',        'width'),
                ('Height',       'height'),
                ('PixelFormat',  'pixel_format'),
                ('SensorWidth',  'sensor_width'),
                ('SensorHeight', 'sensor_height'),
            ]:
                try:
                    metadata[key] = str(getattr(self.camera, attr).get())
                except:
                    metadata[key] = 'N/A'
            metadata['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except Exception as e:
            print_warn(f"Advertencia al obtener metadatos: {e}")
        return metadata

    def add_dark_frame_to_report(self, report_path, filename):
        """Registra el dark frame con su exposición y ganancia reales —los
        parámetros que DEBEN coincidir con los de las imágenes a las que se
        le restará para que la corrección de fondo sea válida."""
        metadata = self.image_metadata.get(filename, {})
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"Archivo: {filename}.png\n")
            f.write("  Láser: APAGADO\n")
            f.write(f"  Polarización: P1={_dark('paddle1')}°, "
                    f"P2={_dark('paddle2')}°, "
                    f"P3={_dark('paddle3')}°\n")
            if metadata:
                f.write(f"  Timestamp: {metadata.get('timestamp', 'N/A')}\n")
                f.write(f"  Exposición: {metadata.get('exposure_time_us', 'N/A')} µs\n")
                f.write(f"  Ganancia: {metadata.get('gain_db', 'N/A')} dB\n")
            f.write("-"*80 + "\n\n")

    def add_temporal_stability_to_report(self, report_path, image_number, filename):
        """
        Registra una foto de la Etapa 1. Si `log_all_metadata` está
        desactivado solo detalla la PRIMERA imagen: en series largas todas
        comparten configuración, y repetirla por cada foto haría el reporte
        ilegible sin aportar información nueva.
        """
        log_all = _temp('log_all_metadata')
        if not log_all and image_number > 1:
            return
        metadata = self.image_metadata.get(filename, {})
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"[FOTO] Imagen {image_number}/{_temp('num_images')}: "
                    f"{filename}.png\n")
            f.write(f"  Polarización fija: P1={_temp('paddle1')}°, "
                    f"P2={_temp('paddle2')}°, "
                    f"P3={_temp('paddle3')}°\n")
            if metadata:
                f.write(f"  Timestamp: {metadata.get('timestamp', 'N/A')}\n")
            if not log_all:
                f.write(f"  (Nota: metadatos de imágenes 2-{_temp('num_images')} "
                        f"omitidos según configuración)\n")
            f.write("-"*80 + "\n\n")

    def add_video_info_to_report(self, report_path, filename,
                                  actual_frames, actual_duration, fps_real):
        """
        Registra el video de la Etapa 1, contrastando lo solicitado con lo
        realmente logrado (duración objetivo vs. real, fps MEDIDOS vs.
        nominales). Ese fps real es el que define el eje de tiempo de todo
        el análisis posterior, así que dejarlo asentado en el reporte
        permite verificar después si la captura fue temporalmente válida.
        """
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"[VIDEO] Archivo: {filename}.{_temp('video_format')}\n")
            f.write(f"  Duración objetivo: {_temp('video_duration')} s\n")
            f.write(f"  Duración real de captura: {actual_duration:.2f} s\n")
            f.write(f"  Frames capturados: {actual_frames}\n")
            f.write(f"  FPS medidos (tiempo real): {fps_real:.2f}\n")
            f.write(f"  FPS escritos en el archivo: {fps_real:.2f}  ← video reproducirá en tiempo real\n")
            f.write(f"  Codec: {_temp('video_codec')}\n")
            f.write(f"  Polarización: P1={_temp('paddle1')}°, "
                    f"P2={_temp('paddle2')}°, "
                    f"P3={_temp('paddle3')}°\n")
            f.write(f"  Timestamp inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-"*80 + "\n\n")

    def add_polarization_to_report(self, report_path, image_number, total_images,
                                    position_name, filename, stage_name):
        """Registra una imagen de las Etapas 2/3 junto con los tres ángulos
        de paleta aplicados, para poder asociar después cada archivo con su
        estado de polarización exacto durante el análisis."""
        pos      = _pol_positions()[position_name]
        metadata = self.image_metadata.get(filename, {})
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"Imagen {image_number}/{total_images}: {filename}.png\n")
            f.write(f"  Posición: {position_name}\n")
            f.write(f"  Paddle 1: {pos['paddle1']}°\n")
            f.write(f"  Paddle 2: {pos['paddle2']}°\n")
            f.write(f"  Paddle 3: {pos['paddle3']}°\n")
            if metadata:
                f.write(f"  Timestamp: {metadata.get('timestamp', 'N/A')}\n")
            f.write("-"*80 + "\n\n")

    def finalize_report(self, report_path, total_dark_frames,
                        total_stage1_photos, total_stage1_videos,
                        total_stage2, total_stage3):
        """
        Cierra el reporte con el conteo de archivos por etapa y la marca de
        tiempo de finalización. Ese recuento permite verificar de un
        vistazo que la adquisición produjo todo lo esperado (p. ej. 10
        imágenes por etapa de polarización) antes de pasar al preprocesado.
        """
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("RESUMEN DEL EXPERIMENTO\n")
            f.write("="*80 + "\n\n")
            f.write(f"ID del Experimento: {_exp('experiment_id')}\n")
            f.write(f"ETAPA 0 - Dark Frame: {total_dark_frames} imagen(es)\n")
            f.write("ETAPA 1 - Estabilidad Temporal:\n")
            f.write(f"  Fotos capturadas: {total_stage1_photos}\n")
            f.write(f"  Videos grabados: {total_stage1_videos}\n")
            f.write(f"Imágenes ETAPA 2 (Haz): {total_stage2}\n")
            f.write(f"Imágenes ETAPA 3 (Haz+Fibra): {total_stage3}\n")
            total_photos = (total_dark_frames + total_stage1_photos
                            + total_stage2 + total_stage3)
            f.write(f"Total de fotos: {total_photos}\n")
            f.write(f"Experimento completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "="*80 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # INICIALIZACIÓN DE DISPOSITIVOS
    # ──────────────────────────────────────────────────────────────────────────

    def initialize_optical_switch(self):
        """
        Conecta el switch óptico NI-DAQ y fija el canal configurado UNA
        sola vez (no cambia por etapa). Opcional: si falla la conexión,
        retorna False sin lanzar excepción -- el experimento continúa sin
        switch, a diferencia de cámara/polarizador (obligatorios).
        """
        print("Inicializando switch óptico...")
        self.optical_switch = OpticalSwitchController(
            device_name=_switch('device_name'),
            port=_switch('port'),
            lines=_switch('lines')
        )
        if self.optical_switch.connect():
            self.optical_switch.set_channel(_switch('channel'))
            return True
        return False

    def initialize_polarizer(self):
        """
        Conecta el controlador de polarización Thorlabs MPC320 (SDK
        Kinesis vía pythonnet/clr) y hace homing de las 3 paletas.
        Obligatorio: si falla cualquier paso, retorna False (el llamador
        aborta el experimento).

        Los tiempos de espera (`time.sleep`) y el intervalo de polling
        (`StartPolling(250)`) son valores empíricos de asentamiento
        mecánico/comunicación con el hardware real, tal como se usaban en
        el script de partida de este proyecto -- no están verificados
        contra la hoja de datos de Kinesis ni documentados por el autor
        más allá de "funcionan de forma confiable en el montaje real". No
        reducirlos sin volver a probar con el polarizador físico
        conectado: un polling/espera insuficiente puede hacer que
        `EnableDevice()`/`Home()` se ejecuten antes de que el dispositivo
        esté listo.
        """
        print("Inicializando controlador de polarización...")
        pol_serial = _pol_serial()
        try:
            DeviceManagerCLI.BuildDeviceList()
            self.polarizer = Polarizer.CreatePolarizer(pol_serial)
            self.polarizer.Connect(pol_serial)

            if not self.polarizer.IsSettingsInitialized():
                self.polarizer.WaitForSettingsInitialized(10000)

            self.polarizer.StartPolling(250)  # ms; ver docstring
            time.sleep(0.25)                  # empírico; ver docstring
            self.polarizer.EnableDevice()
            time.sleep(0.25)                  # empírico; ver docstring

            device_info = self.polarizer.GetDeviceInfo()
            self.polarizer_model = (f"{device_info.Description} "
                                    f"(S/N: {pol_serial})")
            print_ok(f"Polarizador conectado: {self.polarizer_model}")

            print("  Realizando homing de los paddles...")
            for paddle in [PolarizerPaddles.Paddle1,
                           PolarizerPaddles.Paddle2,
                           PolarizerPaddles.Paddle3]:
                self.polarizer.Home(paddle, 60000)  # timeout 60s; empírico, ver docstring
                time.sleep(1)                       # empírico; ver docstring

            print_ok("Homing completado")
            print()
            return True
        except Exception as e:
            print_error(f"Error al inicializar polarizador: {e}")
            print()
            return False

    def set_polarizer_position(self, paddle1_angle, paddle2_angle,
                                paddle3_angle, position_name="custom"):
        """
        Mueve las 3 paletas a los ángulos indicados (bloqueante: espera a
        que cada movimiento termine antes de iniciar el siguiente). El
        `time.sleep(1)` tras cada `MoveTo` es el mismo tipo de espera
        empírica de asentamiento mecánico que en `initialize_polarizer` --
        no reducir sin verificar con el hardware físico conectado.
        """
        print(f"  → Moviendo paddles a {position_name}: "
              f"P1={paddle1_angle}°, P2={paddle2_angle}°, P3={paddle3_angle}°")
        for angle, paddle in [
            (paddle1_angle, PolarizerPaddles.Paddle1),
            (paddle2_angle, PolarizerPaddles.Paddle2),
            (paddle3_angle, PolarizerPaddles.Paddle3),
        ]:
            self.polarizer.MoveTo(Decimal(angle), paddle, 60000)  # timeout 60s; empírico
            time.sleep(1)  # empírico; ver docstring de initialize_polarizer
        print_ok(f"Posición {position_name} alcanzada")

    def initialize_camera(self):
        """
        Conecta a la primera cámara Allied Vision detectada (`cams[0]`,
        sin selección por ID/serial -- se asume una sola cámara conectada)
        y aplica exposición/ganancia/formato de píxel de la sesión.
        Obligatorio: retorna False sin lanzar excepción si no hay ninguna
        cámara disponible (el llamador aborta el experimento).
        """
        print("Inicializando cámara...")
        try:
            self.vmb = vmbpy.VmbSystem.get_instance()
            self.vmb.__enter__()

            cams = self.vmb.get_all_cameras()
            if not cams:
                print_error("No se encontraron cámaras")
                print()
                return False

            self.camera = cams[0]
            self.camera.__enter__()

            self.camera_model = (f"{self.camera.get_model()} "
                                 f"(ID: {self.camera.get_id()})")
            self.camera.ExposureAuto.set(_cam('exposure_auto'))
            self.camera.ExposureTime.set(_cam('exposure_time'))
            self.camera.Gain.set(_cam('gain'))
            pf = _cam('pixel_format')
            try:
                self.camera.get_feature_by_name("PixelFormat").set(pf)
                _pf_actual = self.camera.get_feature_by_name("PixelFormat").get()
                print(f"  Formato de pixel: {_pf_actual}")
            except Exception as _pf_e:
                print_warn(f"No se pudo establecer PixelFormat ({pf}): {_pf_e}")

            print_ok(f"Cámara conectada: {self.camera_model}")
            print(f"  Exposición: {self.camera.ExposureTime.get()} µs")
            print(f"  Ganancia: {self.camera.Gain.get()} dB")
            return True
        except Exception as e:
            print_error(f"Error al inicializar cámara: {e}")
            print()
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # CAPTURA DE FOTO INDIVIDUAL
    # ──────────────────────────────────────────────────────────────────────────

    def capture_image(self, filename, save_folder, apply_overlays=True):
        """
        Captura una imagen (streaming + primer frame completo), opcionalmente
        aplica anotaciones y la guarda como PNG (+ .npz sin overlays si
        `guardar_npz` está activo). Retorna True si se guardó, False si
        hubo timeout esperando el frame o algún otro fallo.

        El timeout de 5s y el polling de 0.1s son valores empíricos (no
        verificados contra la hoja de datos de la cámara Allied Vision) --
        mismo tipo de constante sin justificar que las de
        `initialize_polarizer`/`set_polarizer_position`.
        """
        print(f"  → Capturando imagen: {filename}")
        metadata = self.get_image_metadata()

        class FrameCapture:
            """Contenedor mutable compartido entre el callback de la cámara
            (que corre en el hilo de streaming) y el bucle de espera de
            abajo. Se usa un objeto en vez de variables locales porque el
            callback no puede reasignar el ámbito exterior."""
            def __init__(self):
                self.image    = None
                self.captured = False

        frame_capture = FrameCapture()

        def frame_callback(cam, stream, frame):
            # Se queda con el PRIMER frame completo y marca `captured`;
            # los siguientes se descartan (solo se necesita una imagen).
            if (frame.get_status() == vmbpy.FrameStatus.Complete
                    and not frame_capture.captured):
                img = frame.as_numpy_ndarray()
                if len(img.shape) == 3 and img.shape[2] == 1:
                    img = img.squeeze()
                frame_capture.image    = img.copy()
                frame_capture.captured = True
            stream.queue_frame(frame)

        self.camera.start_streaming(handler=frame_callback, buffer_count=5)

        timeout    = 5    # segundos; empírico, ver docstring
        start_time = time.time()
        while not frame_capture.captured:
            if time.time() - start_time > timeout:
                print_error("Timeout esperando imagen")
                self.camera.stop_streaming()
                return False
            time.sleep(0.1)  # polling; empírico, ver docstring

        self.camera.stop_streaming()
        self.image_metadata[filename] = metadata

        # Guardar tambien el dato crudo sin perdida (.npz), igual que ya se
        # hace con el video y con el dark frame de la Opcion 4 -- se guarda
        # el frame SIN overlays (scale bar/texto quemado), para que sirva
        # como respaldo reprocesable. Se omite si el usuario desactivo
        # guardar_npz en el setup (ahorra tiempo/espacio).
        if _cam('guardar_npz'):
            try:
                npz_path = os.path.join(save_folder, f"{filename}.npz")
                np.savez_compressed(npz_path, frame=frame_capture.image)
                print_ok(f"Datos crudos guardados: {filename}.npz (sin pérdida)")
            except Exception as _e:
                print_warn(f"No se pudo guardar {filename}.npz: {_e}")

        if apply_overlays:
            final_image = self._apply_all_overlays(frame_capture.image)
        else:
            img = frame_capture.image
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            final_image = img

        filepath = os.path.join(save_folder, f"{filename}.png")
        _utils_imagenes.guardar_imagen(filepath, final_image)
        print_ok(f"Imagen guardada: {filename}.png")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # GRABACIÓN DE VIDEO
    # ──────────────────────────────────────────────────────────────────────────

    def record_video(self, filename, save_folder):
        """
        Graba un video durante video_duration segundos en TIEMPO REAL.
        Los FPS se calculan midiendo el tiempo entre el primer y último frame.
        """
        duration  = _temp('video_duration')
        codec_str = _temp('video_codec')
        fmt       = _temp('video_format')

        video_exposure   = _cam('video_exposure_time')
        video_gain       = _cam('video_gain')
        restore_exposure = None
        restore_gain     = None

        if video_exposure is not None:
            try:
                restore_exposure = self.camera.ExposureTime.get()
                self.camera.ExposureTime.set(video_exposure)
                print(f"  [Video] Exposición ajustada a {video_exposure} µs")
            except Exception as e:
                print_warn(f"No se pudo ajustar exposición para video: {e}")

        if video_gain is not None:
            try:
                restore_gain = self.camera.Gain.get()
                self.camera.Gain.set(video_gain)
                print(f"  [Video] Ganancia ajustada a {video_gain} dB")
            except Exception as e:
                print_warn(f"No se pudo ajustar ganancia para video: {e}")

        print(f"\n  → Iniciando grabación de video: {filename}.{fmt}")
        print(f"     Duración objetivo : {duration} s")
        print("     FPS               : se medirán automáticamente (tiempo real)")
        print(f"     Codec             : {codec_str}")

        frames_buffer = []
        frame_shape   = [None]

        def video_callback(cam, stream, frame):
            # Callback MINIMO: solo copia el frame crudo y el timestamp.
            # Los overlays se aplican despues, al escribir el MP4.
            # Esto permite que la camara opere a su FPS maximo real.
            if frame.get_status() == vmbpy.FrameStatus.Complete:
                t   = time.time()
                img = frame.as_numpy_ndarray().copy()
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img.squeeze()
                if frame_shape[0] is None:
                    frame_shape[0] = (img.shape[1], img.shape[0])
                frames_buffer.append((t, img))
            stream.queue_frame(frame)

        self.camera.start_streaming(handler=video_callback, buffer_count=10)

        print("  ⏺ Grabando", end='', flush=True)
        t_start  = time.time()
        last_dot = -1
        # Aviso de memoria RAM estimada -- NO se modifica el callback de
        # captura (video_callback), que debe permanecer minimo para no
        # limitar el FPS real de la camara. Este chequeo corre en este bucle
        # de sondeo (ya existente, fuera de la ruta critica) cada ~20ms, y
        # solo imprime un aviso una vez si la grabacion parece que va a usar
        # mucha RAM -- no bloquea ni cambia el comportamiento de la captura.
        _RAM_AVISO_BYTES = 2 * 1024**3  # 2 GB
        _ram_avisada = False
        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration:
                break
            current_sec = int(elapsed)
            if current_sec > last_dot:
                print('.', end='', flush=True)
                last_dot = current_sec
            if not _ram_avisada and frame_shape[0] is not None and frames_buffer:
                _bytes_por_frame = frames_buffer[-1][1].nbytes
                _estimado = len(frames_buffer) * _bytes_por_frame
                if _estimado > _RAM_AVISO_BYTES:
                    print(f"\n  ⚠  Este video ya esta usando ~{_estimado/1024**3:.1f} GB de RAM "
                          "y sigue grabando -- si el sistema se queda sin memoria, considera "
                          "reducir la duracion o la resolucion en proximas capturas.")
                    _ram_avisada = True
            time.sleep(0.02)

        t_stop          = time.time()
        self.camera.stop_streaming()
        actual_duration = t_stop - t_start
        actual_frames   = len(frames_buffer)
        print_ok(f"({actual_frames} frames en {actual_duration:.2f} s)")

        if actual_frames < 2 or frame_shape[0] is None:
            print_error("Frames insuficientes. Video no guardado.")
            self._restore_camera_params(restore_exposure, restore_gain)
            return False, actual_frames, actual_duration, 0.0

        t_first = frames_buffer[0][0]
        t_last  = frames_buffer[-1][0]
        span    = t_last - t_first

        if span <= 0:
            print_warn("No se pudo calcular FPS real (span=0). Usando 1.0 fps de respaldo.")
            fps_real = 1.0
        else:
            fps_real = (actual_frames - 1) / span

        print(f"     FPS medidos (tiempo real): {fps_real:.4f}")

        filepath = os.path.join(save_folder, f"{filename}.{fmt}")
        fourcc   = cv2.VideoWriter_fourcc(*codec_str)
        writer   = cv2.VideoWriter(filepath, fourcc, fps_real, frame_shape[0])

        if not writer.isOpened():
            print_error("No se pudo abrir VideoWriter. Verifica codec/formato.")
            writer.release()
            self._restore_camera_params(restore_exposure, restore_gain)
            return False, actual_frames, actual_duration, fps_real

        font          = cv2.FONT_HERSHEY_SIMPLEX
        font_scale    = 1
        thickness     = 1
        color         = (255, 255, 255)
        outline_color = (0, 0, 0)
        margin        = 8

        sample_text = f"{actual_frames}/{actual_frames}"
        (tw, th), _ = cv2.getTextSize(sample_text, font, font_scale, thickness)
        user_text_offset = (th + margin * 2) if self.overlay_text else 0

        for idx, (_, frm) in enumerate(frames_buffer, start=1):
            # Aplicar overlays aqui (fuera del callback) para no limitar FPS
            frame_out = self._apply_all_overlays(frm)
            counter_txt = f"{idx}/{actual_frames}"
            (cw, ch), _ = cv2.getTextSize(counter_txt, font, font_scale, thickness)
            h, w = frame_out.shape[:2]
            x    = w - cw - margin
            y    = h - margin - user_text_offset
            cv2.putText(frame_out, counter_txt, (x, y), font, font_scale,
                        outline_color, thickness + 2, cv2.LINE_AA)
            cv2.putText(frame_out, counter_txt, (x, y), font, font_scale,
                        color, thickness, cv2.LINE_AA)
            writer.write(frame_out)
        writer.release()

        # Guardar frames crudos sin pérdida en .npz (conserva bit-depth
        # original) — opcional, ver guardar_npz en el setup.
        if _cam('guardar_npz'):
            try:
                npz_path = os.path.join(save_folder, f"{filename}.npz")
                timestamps = np.array([t for t, _ in frames_buffer])
                raw_frames = np.array([frm for _, frm in frames_buffer])
                np.savez(npz_path,
                         frames=raw_frames,
                         timestamps=timestamps,
                         fps=fps_real)
                print_ok(f"Datos crudos guardados: {filename}.npz  (sin pérdida)")
            except Exception as _e:
                print_warn(f"No se pudo guardar .npz: {_e}")

        print_ok(f"Video guardado: {filename}.{fmt}")
        print(f"     Frames totales   : {actual_frames}")
        print(f"     Duración real    : {actual_duration:.2f} s")
        print(f"     FPS del archivo  : {fps_real:.4f}  ← tiempo real garantizado")

        self._restore_camera_params(restore_exposure, restore_gain)
        return True, actual_frames, actual_duration, fps_real

    def _restore_camera_params(self, restore_exposure, restore_gain):
        """
        Restaura exposición y ganancia a los valores que tenían antes de
        grabar el video, si `record_video` los había cambiado
        (`video_exposure_time`/`video_gain`).

        Es lo que garantiza la CONSISTENCIA EXPERIMENTAL entre etapas: sin
        esta restauración, las imágenes de polarización capturadas después
        del video quedarían con los ajustes del video, y sus intensidades
        no serían comparables con las del dark frame ni entre sí.
        """
        if restore_exposure is not None:
            try:
                self.camera.ExposureTime.set(restore_exposure)
                print(f"  [Video] Exposición restaurada a {restore_exposure} µs")
            except Exception as e:
                print_warn(f"No se pudo restaurar exposición: {e}")
        if restore_gain is not None:
            try:
                self.camera.Gain.set(restore_gain)
                print(f"  [Video] Ganancia restaurada a {restore_gain} dB")
            except Exception as e:
                print_warn(f"No se pudo restaurar ganancia: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 0 — DARK FRAME
    # ──────────────────────────────────────────────────────────────────────────

    def run_stage0_dark_frame(self, report_path):
        """
        Etapa 0: captura 1 imagen con el láser apagado (dark frame),
        en la posición de polarización configurada para esta etapa.
        Retorna 1 si se capturó correctamente, 0 si falló.
        """
        print_banner("ETAPA 0: DARK FRAME (láser apagado)")
        self.add_stage_header_to_report(report_path, 0, "Dark Frame (láser apagado)")

        self.set_polarizer_position(
            _dark('paddle1'),
            _dark('paddle2'),
            _dark('paddle3'),
            "dark frame"
        )
        time.sleep(0.5)  # empírico; ver docstring de initialize_polarizer

        filename = _filename_darkframe()
        if self.capture_image(filename, self.subfolder_dark_frame, apply_overlays=False):
            self.add_dark_frame_to_report(report_path, filename)
            print()
            print_ok("ETAPA 0 completada — Dark frame guardado.")
            print()
            return 1
        else:
            print()
            print_error("ETAPA 0 — No se pudo capturar el dark frame.")
            print()
            return 0

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 1 — ESTABILIDAD TEMPORAL
    # ──────────────────────────────────────────────────────────────────────────

    def run_stage1_temporal_stability(self, report_path):
        """
        Etapa 1: captura fotos y/o graba video en una posición de
        polarización fija, para caracterizar la estabilidad temporal del
        haz. Retorna (total_fotos, total_videos) capturados con éxito.
        """
        print_banner("ETAPA 1: ESTABILIDAD TEMPORAL")

        capture_photos = _temp('capture_photos')
        capture_video  = _temp('capture_video')

        if not capture_photos and not capture_video:
            print_warn("Ambas opciones de captura están desactivadas. Etapa 1 omitida.")
            return 0, 0

        if capture_photos:
            print(f"  • Fotos: {_temp('num_images')} imágenes, "
                  f"intervalo {_temp('time_interval')} s")
        if capture_video:
            print(f"  • Video: {_temp('video_duration')} s, "
                  f"FPS medidos automáticamente")
        print(f"  • Polarización: P1={_temp('paddle1')}°, "
              f"P2={_temp('paddle2')}°, "
              f"P3={_temp('paddle3')}°\n")

        self.add_stage_header_to_report(report_path, 1, "Estabilidad Temporal")
        self.set_polarizer_position(
            _temp('paddle1'),
            _temp('paddle2'),
            _temp('paddle3'),
            "estabilidad temporal"
        )
        time.sleep(0.5)  # empírico; ver docstring de initialize_polarizer

        total_photos = 0
        total_videos = 0

        if capture_photos:
            print_seccion("CAPTURA DE FOTOS")
            for i in range(1, _temp('num_images') + 1):
                print(f"\n  Foto {i}/{_temp('num_images')}")
                filename = _filename_estabilidad(i)
                if self.capture_image(filename, self.subfolder_temporal):
                    self.add_temporal_stability_to_report(report_path, i, filename)
                    total_photos += 1
                if i < _temp('num_images'):
                    print(f"  → Esperando {_temp('time_interval')} s...")
                    time.sleep(_temp('time_interval'))

        if capture_video:
            print_seccion("GRABACIÓN DE VIDEO")
            vid_filename = _filename_video()
            success, actual_frames, actual_duration, fps_real = self.record_video(
                vid_filename, self.subfolder_temporal
            )
            if success:
                self.add_video_info_to_report(report_path, vid_filename,
                                               actual_frames, actual_duration, fps_real)
                total_videos = 1

        print()
        print_ok("ETAPA 1 completada")
        print()
        return total_photos, total_videos

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 2 — POLARIZACIONES (HAZ)
    # ──────────────────────────────────────────────────────────────────────────

    def run_stage2_polarization_haz(self, report_path):
        """
        Etapa 2: captura 1 imagen del haz por cada posición de
        polarización habilitada (hasta 10). Retorna el número de
        posiciones habilitadas (no necesariamente el número de imágenes
        realmente capturadas con éxito -- ver `capture_image`).
        """
        print_banner("ETAPA 2: DIFERENTES POLARIZACIONES (HAZ)")
        self.add_stage_header_to_report(report_path, 2,
                                        "Diferentes Polarizaciones (Haz)")
        enabled_indices = _pol_enabled_indices()
        pol_positions   = _pol_positions()
        total_enabled   = len(enabled_indices)

        print(f"  Posiciones habilitadas: {total_enabled}  "
              f"({', '.join(f'P{idx:02d}' for idx in enabled_indices)})")

        for capture_num, pos_idx in enumerate(enabled_indices, 1):
            pos_name = f'pos{pos_idx}'
            print_seccion(f"CAPTURA {capture_num}/{total_enabled}  (P{pos_idx:02d})")
            pos = pol_positions[pos_name]
            self.set_polarizer_position(
                pos['paddle1'], pos['paddle2'], pos['paddle3'], pos_name)
            time.sleep(0.5)  # empírico; ver docstring de initialize_polarizer
            filename = _filename_haz(pos_idx)
            if self.capture_image(filename, self.subfolder_haz):
                self.add_polarization_to_report(
                    report_path, capture_num, total_enabled, pos_name, filename, "Haz")
            time.sleep(0.3)  # empírico; ver docstring de initialize_polarizer

        print()
        print_ok("ETAPA 2 completada")
        print()
        return total_enabled

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 3 — POLARIZACIONES (HAZ + FIBRA)
    # ──────────────────────────────────────────────────────────────────────────

    def run_stage3_polarization_haz_fibra(self, report_path):
        """
        Etapa 3: igual que la Etapa 2, pero con la fibra insertada
        manualmente en el arreglo óptico (inserción entre etapas,
        indicada al usuario en el mensaje de confirmación previo). Retorna
        el número de posiciones habilitadas.
        """
        print_banner("ETAPA 3: DIFERENTES POLARIZACIONES (HAZ + FIBRA)")
        print()
        print_ok("Iniciando ETAPA 3...")
        print()
        self.add_stage_header_to_report(report_path, 3,
                                        "Diferentes Polarizaciones (Haz + Fibra)")
        enabled_indices = _pol_enabled_indices()
        pol_positions   = _pol_positions()
        total_enabled   = len(enabled_indices)

        print(f"  Posiciones habilitadas: {total_enabled}  "
              f"({', '.join(f'P{idx:02d}' for idx in enabled_indices)})")

        for capture_num, pos_idx in enumerate(enabled_indices, 1):
            pos_name = f'pos{pos_idx}'
            print_seccion(f"CAPTURA {capture_num}/{total_enabled}  (P{pos_idx:02d})")
            pos = pol_positions[pos_name]
            self.set_polarizer_position(
                pos['paddle1'], pos['paddle2'], pos['paddle3'], pos_name)
            time.sleep(0.5)  # empírico; ver docstring de initialize_polarizer
            filename = _filename_haz_fibra(pos_idx)
            if self.capture_image(filename, self.subfolder_haz_fibra):
                self.add_polarization_to_report(
                    report_path, capture_num, total_enabled, pos_name, filename, "Haz+Fibra")
            time.sleep(0.3)  # empírico; ver docstring de initialize_polarizer

        print()
        print_ok("ETAPA 3 completada")
        print()
        return total_enabled

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENTO PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────

    def run_experiment(self):
        """
        Orquesta el experimento completo de adquisición del haz óptico:
        crea las carpetas, conecta el hardware, ejecuta las 4 etapas
        habilitadas en orden y cierra el reporte.

        Secuencia y por qué ese orden:
          0. **Dark frame** (láser apagado) — primero, para tener la línea
             de base de ruido del sensor antes de cualquier medición.
          1. **Estabilidad temporal** — polarización fija; produce el video
             del que sale el dato central de la tesis.
          2. **Polarizaciones del haz** — barrido de estados de paleta.
          3. **Polarizaciones del haz + fibra** — mismo barrido con la
             fibra ya insertada en el arreglo.

        Entre etapas se abre una vista previa en vivo que exige
        confirmación del operador ([S]/[Q]): la Etapa 0 requiere apagar el
        láser manualmente y la Etapa 3 requiere insertar la fibra, dos
        acciones físicas que el software no puede realizar ni verificar
        por sí mismo. Abortar en cualquiera de esos puntos lanza
        KeyboardInterrupt y detiene el experimento de forma limpia.

        Cada etapa puede desactivarse por separado (`StagesConfig`), lo que
        permite repetir solo una parte sin volver a capturar todo.

        Retorna True si el experimento llegó al final; False si el hardware
        obligatorio no pudo inicializarse o si ocurrió un error durante la
        captura (en cuyo caso se imprime el traceback completo). La
        liberación del hardware NO ocurre aquí sino en `cleanup()`, que el
        llamador invoca siempre desde un bloque `finally`.
        """
        print_banner("INICIANDO EXPERIMENTO AUTOMATIZADO - 4 ETAPAS")

        bar_px = round(_scale('bar_length_um') * _scale('px_per_um'))

        print("Configuración del experimento:")
        print(f"  ID: {_exp('experiment_id')}")
        print(f"  Condiciones: {_exp('conditions')}")
        print(f"  Canal: {_exp('channel')}")
        print(f"  Longitud de onda: {_exp('wavelength')}")
        print(f"  Temperatura: {_exp('temperature')}°C")
        print(f"  Corriente: {_exp('current')} mA")
        print(f"  Switch óptico - Canal: {_switch('channel')}")
        print(f"  Carpeta raíz del dispositivo: {self.device_root_folder}")
        print(f"  Caso: {_utils_carpetas.CASO_LABELS.get(self.caso, self.caso)}")
        print()
        print("  Etapas habilitadas:")
        for key, label in [
            ('stage0_dark_frame',             'Etapa 0 — Dark Frame'),
            ('stage1_temporal_stability',     'Etapa 1 — Estabilidad Temporal'),
            ('stage2_polarization_haz',       'Etapa 2 — Polarizaciones (Haz)'),
            ('stage3_polarization_haz_fibra', 'Etapa 3 — Polarizaciones (Haz+Fibra)'),
        ]:
            estado = '✓ ACTIVA ' if _stage(key) else '✗ OMITIDA'
            print(f"    [{estado}] {label}")
        print()
        if _temp('capture_photos'):
            print(f"    Fotos: {_temp('num_images')} imgs, "
                  f"intervalo {_temp('time_interval')} s")
        if _temp('capture_video'):
            print(f"    Video: {_temp('video_duration')} s, "
                  f"FPS medidos automáticamente, "
                  f"codec {_temp('video_codec')}")
        if _overlay_text():
            print(f"  Anotación en imágenes: \"{_overlay_text()}\"")
        if _scale('enabled'):
            print(f"  Barra de escala: {_scale('bar_length_um')} µm "
                  f"({bar_px} px) | 1 px = {_scale('um_per_px')} µm")
        else:
            print("  Barra de escala: desactivada")
        print()

        self.create_experiment_folders()

        if not self.initialize_optical_switch():
            print_warn("No se pudo inicializar el switch óptico. Continuando sin él...")
        if not self.initialize_polarizer():
            return False
        if not self.initialize_camera():
            return False

        self.save_acquisition_metadata()
        report_path = self.create_report_file()

        print_banner("TODOS LOS DISPOSITIVOS LISTOS")

        try:
            etapas_activas = [
                lbl for key, lbl in [
                    ('stage0_dark_frame',             'ETAPA 0 (Dark Frame)'),
                    ('stage1_temporal_stability',     'ETAPA 1 (Estabilidad Temporal)'),
                    ('stage2_polarization_haz',       'ETAPA 2 (Haz)'),
                    ('stage3_polarization_haz_fibra', 'ETAPA 3 (Haz+Fibra)'),
                ]
                if _stage(key)
            ]
            primera_etapa_label = (etapas_activas[0] if etapas_activas
                                   else "el experimento")

            self.show_live_preview_and_confirm(
                stage_title="Verificación Inicial del Sistema",
                instruction_lines=[
                    "Verifique que la imagen se vea correctamente.",
                    "Confirme alineación óptica y demás condiciones experimentales.",
                    f"Presione [S] cuando esté listo para iniciar {primera_etapa_label}.",
                ]
            )

            # ── ETAPA 0 ───────────────────────────────────────────────────────
            total_dark_frames = 0
            if _stage('stage0_dark_frame'):
                self.show_live_preview_and_confirm(
                    stage_title="ETAPA 0 — Dark Frame",
                    instruction_lines=[
                        "APAGUE el láser antes de continuar.",
                        "Se capturará 1 imagen de fondo (sin señal).",
                        "Presione [S] cuando el láser esté apagado y esté listo.",
                    ]
                )
                total_dark_frames = self.run_stage0_dark_frame(report_path)
            else:
                print()
                print_skip("ETAPA 0 omitida (desactivada).")
                print()

            # ── ETAPA 1 ───────────────────────────────────────────────────────
            total_stage1_photos, total_stage1_videos = 0, 0
            if _stage('stage1_temporal_stability'):
                self.show_live_preview_and_confirm(
                    stage_title="Avanzar a ETAPA 1 — Estabilidad Temporal",
                    instruction_lines=[
                        "Encienda el láser nuevamente si lo apagó.",
                        "Verifique la señal en la imagen antes de continuar.",
                        "Presione [S] para iniciar la captura de estabilidad temporal.",
                    ]
                )
                total_stage1_photos, total_stage1_videos = \
                    self.run_stage1_temporal_stability(report_path)
            else:
                print()
                print_skip("ETAPA 1 omitida (desactivada).")
                print()

            # ── ETAPA 2 ───────────────────────────────────────────────────────
            total_stage2 = 0
            if _stage('stage2_polarization_haz'):
                self.show_live_preview_and_confirm(
                    stage_title="Avanzar a ETAPA 2 — Diferentes Polarizaciones (Haz)",
                    instruction_lines=[
                        f"Se capturarán {len(_pol_enabled_indices())} imágenes en diferentes polarizaciones.",
                        "Verifique que el sistema esté en condiciones correctas.",
                        "Presione [S] para iniciar la ETAPA 2.",
                    ]
                )
                total_stage2 = self.run_stage2_polarization_haz(report_path)
            else:
                print()
                print_skip("ETAPA 2 omitida (desactivada).")
                print()

            # ── ETAPA 3 ───────────────────────────────────────────────────────
            total_stage3 = 0
            if _stage('stage3_polarization_haz_fibra'):
                self.show_live_preview_and_confirm(
                    stage_title="Avanzar a ETAPA 3 — Diferentes Polarizaciones (Haz + Fibra)",
                    instruction_lines=[
                        "Realice las modificaciones necesarias en el arreglo físico.",
                        f"Se capturarán {len(_pol_enabled_indices())} imágenes con haz + fibra.",
                        "Presione [S] cuando el arreglo esté listo para iniciar la ETAPA 3.",
                    ]
                )
                total_stage3 = self.run_stage3_polarization_haz_fibra(report_path)
            else:
                print()
                print_skip("ETAPA 3 omitida (desactivada).")
                print()

            self.finalize_report(report_path, total_dark_frames,
                                 total_stage1_photos, total_stage1_videos,
                                 total_stage2, total_stage3)

            print_banner("EXPERIMENTO COMPLETADO EXITOSAMENTE")
            print(f"\nID del experimento: {_exp('experiment_id')}")
            print(f"  - Dark frame: {total_dark_frames}")
            print(f"  - Estabilidad temporal (fotos): {total_stage1_photos}")
            print(f"  - Estabilidad temporal (videos): {total_stage1_videos}")
            print(f"  - Haz: {total_stage2}")
            print(f"  - Haz+Fibra: {total_stage3}")
            print(f"\nUbicación: {self.save_folder}")
            print("Reporte generado: Reporte_Experimento.txt\n")
            return True

        except Exception as e:
            print()
            print_error(f"Error durante el experimento: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # LIMPIEZA
    # ──────────────────────────────────────────────────────────────────────────

    def return_polarizer_to_home(self):
        """
        Devuelve las 3 paletas a su posición de referencia (home) al
        terminar el experimento, de modo que la siguiente sesión arranque
        siempre desde el mismo estado mecánico conocido en lugar de heredar
        la última polarización aplicada.

        Los errores solo se advierten, no se propagan: esto se ejecuta
        durante la limpieza y un fallo aquí no debe impedir que se liberen
        los demás dispositivos.
        """
        if self.polarizer:
            try:
                print("\nRegresando polarizador a posición home...")
                for paddle in [PolarizerPaddles.Paddle1,
                               PolarizerPaddles.Paddle2,
                               PolarizerPaddles.Paddle3]:
                    self.polarizer.Home(paddle, 60000)
                    time.sleep(1)
                print_ok("Polarizador en posición home")
            except Exception as e:
                print_warn(f"Error al hacer homing final: {e}")

    def cleanup(self):
        """
        Libera todo el hardware y cierra las ventanas de OpenCV. `main()`
        lo invoca desde un bloque `finally`, de modo que se ejecuta tanto
        si el experimento terminó bien como si falló o el operador lo
        abortó.

        Cada dispositivo se cierra en su propio try/except: un fallo al
        cerrar la cámara no debe impedir que se libere el polarizador o el
        switch. Dejar cualquiera de ellos tomado obligaría a reconectar el
        equipo físicamente antes del siguiente experimento, así que aquí se
        prioriza liberar todo lo posible sobre propagar el error.
        """
        print("\nCerrando dispositivos...")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            print_warn(f"Error al cerrar ventanas de OpenCV: {e}")
        if self.camera:
            try:
                self.camera.__exit__(None, None, None)
                print_ok("Cámara cerrada")
            except Exception as e:
                print_warn(f"Error al cerrar la cámara: {e}")
        if self.vmb:
            try:
                self.vmb.__exit__(None, None, None)
            except Exception as e:
                print_warn(f"Error al cerrar Vimba: {e}")
        self.return_polarizer_to_home()
        if self.polarizer:
            try:
                self.polarizer.StopPolling()
                self.polarizer.Disconnect()
                print_ok("Polarizador desconectado")
            except Exception as e:
                print_warn(f"Error al desconectar el polarizador: {e}")
        if self.optical_switch:
            self.optical_switch.disconnect()
        print("\n¡Sistema apagado correctamente!\n")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main(session=None):
    """
    Punto de entrada principal. Se comporta igual tanto si se llama desde
    el inicializador (main.py, con `session` ya configurado) como en modo
    independiente (`python Parte1.py`, `session=None`): siempre pregunta
    primera-vez-vs-raíz-existente y qué caso adquirir — ver
    resolver_destino_y_caso().

    Retorna la ruta de la carpeta RAÍZ DEL DISPOSITIVO (contiene los 3
    casos), o None si hay error o el usuario cancela, lo que permite al
    inicializador propagar esa ruta a Parte2 y Parte3.
    """
    global SESSION
    SESSION = session

    # ── 1. Raíz del dispositivo + caso a adquirir ─────────────────────────────
    resuelto = resolver_destino_y_caso(session)
    if resuelto is None:
        return None
    carpeta_raiz, caso, carpeta_caso = resuelto

    # ── 2. Inicializar el controlador ─────────────────────────────────────────
    overlay = _overlay_text()
    controller = OpticalSystemController(
        overlay_text=overlay,
        device_root_folder=carpeta_raiz,
        caso=caso,
        carpeta_caso=carpeta_caso,
    )

    try:
        success = controller.run_experiment()
        if not success:
            print()
            print_warn("El experimento no se completó correctamente")
            return None
        if session is not None:
            session.caso_actual = caso
        return controller.device_root_folder    # ← retornamos la carpeta RAÍZ del dispositivo

    except KeyboardInterrupt:
        print()
        print()
        print_warn("Experimento interrumpido por el usuario")
        return None
    except Exception as e:
        print()
        print_error(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()