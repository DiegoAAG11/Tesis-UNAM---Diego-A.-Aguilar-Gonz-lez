# -*- coding: utf-8 -*-
"""
CamaraTurbulencia.py — Caracterización de una cámara de turbulencia óptica.

Descripción:
Calcula los tres parámetros de turbulencia estándar en FSO a partir del
beam wander de un haz gaussiano colimado:

    Cₙ²  — coeficiente de estructura del índice de refracción  [m^(-2/3)]
    r₀   — parámetro de Fried (longitud de coherencia)          [m]
    σ_R  — varianza de Rytov (índice de scintillation débil)   [adim.]

Ecuaciones (enlace horizontal, theta = 0):
    W₀      = f₁·λ / (π·W_fibre)  — W_fibre introducido manualmente
    <r_c^2> = sigma_cx^2 + sigma_cy^2  (suma de varianzas, no media)
    ⟨r_c²⟩ = 0.54·L²·(λ/2W₀)²·(2W₀/r₀)^(5/3)
    r₀     = [0.4234·k²·Cₙ²·L]^(-3/5)
    σ_R    = 1.23·Cₙ²·k^(7/6)·L^(11/6)

Metodología:
Implementada como 3 funciones independientes en este mismo archivo,
correspondientes a las Opciones 4/5/6 del menú principal:
1) Adquisición (main_adquisicion): dark frame + grabación de video por
   cada caso experimental (ΔT, velocidad de ventiladores), caso por caso.
2) Preprocesado (main_preprocesado): aplica ROI + resta de dark frame a
   cada video crudo, un frame a la vez.
3) Análisis (main_analisis): extrae los centroides del video preprocesado
   (o crudo), calcula la varianza del beam wander excluyendo el
   transitorio térmico inicial, y despeja r₀, Cₙ² y σ_R con las
   ecuaciones de arriba.

Puede ejecutarse de forma independiente o invocado desde main.py vía
modules/turbulencia_adquisicion.py y modules/turbulencia_analisis.py.
Cuando se invoca con SESSION != None, lee todos los parámetros de
SESSION.turbulence / SESSION.camera / SESSION.experiment.

Autor: Diego Aguilar
"""

import os
import time
import math
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime

import utils_carpetas as _utils_carpetas
import utils_imagenes as _utils_imagenes
import utils_beam_metrics as _beam_metrics
from console_ui import print_banner, print_ok, print_error, print_warn

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Los imports de hardware son perezosos — solo se cargan en main_adquisicion.
# Esto permite que el preprocesado y el analisis corran sin los drivers de
# camara/polarizador instalados.
vmbpy              = None  # se carga bajo demanda
DeviceManagerCLI   = None
Polarizer          = None
PolarizerPaddles   = None
Decimal            = None

# GUI unificada de vista en vivo
import os as _os, sys as _sys
_LV_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _LV_ROOT not in _sys.path: _sys.path.insert(0, _LV_ROOT)


def _load_hardware_libs():
    """
    Importa las librerias de hardware (vmbpy, Thorlabs, NI-DAQmx).
    Solo se llama desde main_adquisicion, nunca al importar el modulo.
    Retorna True si todas las librerias estan disponibles, False si alguna falla.
    """
    global vmbpy, DeviceManagerCLI, Polarizer, PolarizerPaddles, Decimal
    ok = True

    try:
        import vmbpy as _vmbpy
        vmbpy = _vmbpy
    except Exception as e:
        print(f"  Advertencia: vmbpy no disponible: {e}")
        ok = False

    try:
        import clr as _clr
        _kinesis = "C:\\Program Files\\Thorlabs\\Kinesis\\"
        _clr.AddReference(_kinesis + "Thorlabs.MotionControl.DeviceManagerCLI.dll")
        _clr.AddReference(_kinesis + "Thorlabs.MotionControl.GenericMotorCLI.dll")
        _clr.AddReference(_kinesis + "ThorLabs.MotionControl.PolarizerCLI.dll")
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI as _DMC
        from Thorlabs.MotionControl.PolarizerCLI import Polarizer as _P, PolarizerPaddles as _PP
        from System import Decimal as _Dec
        DeviceManagerCLI = _DMC
        Polarizer = _P
        PolarizerPaddles = _PP
        Decimal = _Dec
    except Exception as e:
        print(f"  Advertencia: Thorlabs Kinesis no disponible: {e}")
        # Non-fatal for acquisition (polarizer optional)

    try:
        from gui.live_view import mostrar_vista_en_vivo as _lv
        global _lv_confirm
        _lv_confirm = _lv
    except Exception as e:
        print(f"  Advertencia: live_view no disponible: {e}")
        ok = False

    return ok


_lv_confirm = None  # set by _load_hardware_libs


# ── Integracion con SessionConfig ─────────────────────────────────────────────
# SESSION se inyecta desde modules/turbulencia_adquisicion.py y
# modules/turbulencia_analisis.py cuando el script corre bajo main.py. Si se
# ejecuta de forma independiente (`python CamaraTurbulencia.py`), SESSION
# queda en None y cada accesor devuelve su valor de respaldo.
#
# Los seis accesores de abajo existen precisamente para que ese doble modo de
# ejecución no obligue a salpicar el archivo de comprobaciones
# `if SESSION is not None` : cada uno lee un sub-objeto distinto de la sesión
# (turbulence, experiment, camera, scale, switch, polarizer) y acepta un
# `fallback` explícito en el punto de uso.
SESSION = None

def _turb(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.turbulence, key, fallback)
    return fallback

def _exp(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.experiment, key, fallback)
    return fallback

def _cam(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.camera, key, fallback)
    return fallback

def _scale(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.scale, key, fallback)
    return fallback

def _switch_cfg(key, fallback):
    if SESSION is not None:
        return getattr(SESSION.switch, key, fallback)
    return fallback

def _pol_serial():
    if SESSION is not None:
        return SESSION.polarizer.serial
    return "38388714"


# =============================================================================
# CONTROLADOR DE HARDWARE
# =============================================================================
# La configuración del experimento NO se pide aquí: el diálogo
# (gui/turbulence_dialog.py) lo invoca main.py::step_turbulencia_adquisicion
# antes de llamar a main_adquisicion, de modo que todo el barrido comparta
# exactamente los mismos parámetros ópticos.


class TurbController:
    """
    Controlador unificado del hardware del experimento de turbulencia:
    cámara Allied Vision, controlador de polarización Thorlabs y switch
    óptico NI-DAQ.

    Se usa como *context manager*, de forma que la conexión y la
    liberación de los tres dispositivos queden garantizadas incluso si el
    experimento falla a mitad de camino:

        with TurbController(session) as ctl:
            ctl.set_polarizer(p1, p2, p3)
            ctl.record_video(...)

    Política de disponibilidad deliberadamente asimétrica: la CÁMARA es
    obligatoria (sin ella no hay dato, `_init_camera` lanza excepción),
    mientras que polarizador y DAQ son OPCIONALES — si no están
    conectados se degradan a "no disponible" con advertencia y sus
    métodos se vuelven no-ops. Esto permite trabajar en montajes reducidos
    (un solo canal, polarización fija manualmente) sin modificar código.

    Es el equivalente, para el pipeline de turbulencia, de
    `Parte1.py::OpticalSystemController` en el pipeline del haz óptico;
    son dos implementaciones separadas porque los experimentos tienen
    ciclos de vida distintos (aquí un solo dark frame por sesión y un
    bucle abierto de casos, allá 4 etapas fijas).
    """
    def __init__(self, session=None):
        self.session   = session
        self.vmb       = None
        self.camera    = None
        self.polarizer = None
        self.daq_task  = None

    def __enter__(self):
        """Conecta los tres dispositivos al entrar al bloque `with`. El
        orden importa: DAQ y polarizador primero (opcionales, degradan sin
        abortar) y la cámara al final, para no dejar dispositivos a medio
        inicializar si esta última —obligatoria— falla."""
        self._init_daq()
        self._init_polarizer()
        self._init_camera()
        return self

    def __exit__(self, *_):
        """Libera todo el hardware al salir del bloque `with`, tanto en
        salida normal como por excepción — evita dejar la cámara o el DAQ
        tomados si el experimento se interrumpe."""
        self._close()

    def _init_daq(self):
        """
        Conecta el switch óptico NI-DAQ, si está disponible. Opcional en
        todo el archivo: cualquier fallo (driver no instalado, dispositivo
        no conectado) deja `self.daq_task = None` con una advertencia, sin
        lanzar excepción -- `set_channel` más abajo ya maneja ese caso
        retornando de inmediato.
        """
        try:
            import nidaqmx as _nidaqmx
            from nidaqmx.constants import LineGrouping as _LG
        except Exception:
            print("  DAQ no disponible (NI-DAQmx no instalado).")
            return
        dev   = _switch_cfg("device_name", "Dev1")
        port  = _switch_cfg("port", 0)
        lines = _switch_cfg("lines", [0, 1])
        try:
            self.daq_task = _nidaqmx.Task()
            line_str = ", ".join(
                f"{dev}/port{port}/line{ln}" for ln in lines)
            self.daq_task.do_channels.add_do_chan(
                line_str, line_grouping=_LG.CHAN_PER_LINE)
            self.daq_task.start()
            print_ok("DAQ inicializado")
        except Exception as e:
            print_warn(f"DAQ no disponible: {e}")
            self.daq_task = None

    def _init_polarizer(self):
        """
        Conecta el polarizador Thorlabs MPC320 y hace homing de las 3
        paletas -- opcional en este pipeline (a diferencia de Parte1.py,
        donde es obligatorio): cualquier fallo deja `self.polarizer =
        None` con advertencia, sin abortar la sesión.

        Los `time.sleep`/`StartPolling(250)`/timeouts son valores
        empíricos de asentamiento mecánico/comunicación con el hardware
        real, mismo tipo de constante sin verificar contra hoja de datos
        que en Parte1.py::OpticalSystemController.initialize_polarizer
        (ver docstring allá) -- no reducir sin probar con el polarizador
        físico conectado.
        """
        try:
            DeviceManagerCLI.BuildDeviceList()
            serial = _pol_serial()
            self.polarizer = Polarizer.CreatePolarizer(serial)
            self.polarizer.Connect(serial)
            if not self.polarizer.IsSettingsInitialized():
                self.polarizer.WaitForSettingsInitialized(10000)
            self.polarizer.StartPolling(250)  # ms; empírico, ver docstring
            time.sleep(0.5)                   # empírico, ver docstring
            self.polarizer.EnableDevice()
            info = self.polarizer.GetDeviceInfo()
            print_ok(f"Polarizador conectado: {info.Description}")
            for paddle in [PolarizerPaddles.Paddle1,
                           PolarizerPaddles.Paddle2,
                           PolarizerPaddles.Paddle3]:
                self.polarizer.Home(paddle, 60000)  # timeout 60s; empírico
                time.sleep(1)                       # empírico, ver docstring
        except Exception as e:
            print_warn(f"Polarizador no disponible: {e}")
            self.polarizer = None

    def _init_camera(self):
        """
        Conecta a la primera cámara Allied Vision detectada (`cams[0]`,
        sin selección por ID/serial) y aplica exposición/ganancia/formato
        de píxel. Obligatorio: lanza `RuntimeError` si no hay ninguna
        cámara conectada (a diferencia de `_init_daq`/`_init_polarizer`,
        que degradan a "no disponible" sin abortar).
        """
        self.vmb    = vmbpy.VmbSystem.get_instance()
        self.vmb.__enter__()
        cams = self.vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("No se encontro ninguna camara Allied Vision.")
        self.camera = cams[0]
        self.camera.__enter__()
        exp  = _cam("exposure_time", 10000)
        gain = _cam("gain", 10.0)
        try:
            self.camera.ExposureAuto.set("Off")
            self.camera.ExposureTime.set(exp)
            self.camera.Gain.set(gain)
            pf = _turb("pixel_format", "Mono8") if SESSION else "Mono8"
            if not pf: pf = "Mono8"
            try:
                self.camera.get_feature_by_name("PixelFormat").set(pf)
            except Exception as _e:
                print(f"  Advertencia: no se pudo establecer PixelFormat ({pf}): {_e}")
            print(f"Camara inicializada  |  exp={exp} us  gain={gain} dB  fmt={pf}")
        except Exception as e:
            print_warn(f"Error al configurar camara: {e}")

    def _close(self):
        """
        Libera los tres dispositivos, cada uno en su propio bloque
        try/except: un fallo al cerrar la cámara no debe impedir que se
        liberen el DAQ y el polarizador. Dejar cualquiera de ellos tomado
        obligaría a reconectar el hardware físicamente antes del siguiente
        experimento, así que aquí se prioriza liberar todo lo posible
        sobre propagar el error.
        """
        try:
            if self.camera:   self.camera.__exit__(None, None, None)
            if self.vmb:      self.vmb.__exit__(None, None, None)
        except Exception as e:
            print_warn(f"Error al cerrar camara: {e}")
        try:
            if self.daq_task:
                self.daq_task.stop(); self.daq_task.close()
                print_ok("DAQ cerrado")
        except Exception as e:
            print_warn(f"Error al cerrar DAQ: {e}")
        try:
            if self.polarizer:
                self.polarizer.StopPolling()
                self.polarizer.Disconnect()
                print_ok("Polarizador desconectado")
        except Exception as e:
            print_warn(f"Error al cerrar polarizador: {e}")

    def set_channel(self, channel_str: str):
        """
        Selecciona uno de los 4 canales del switch óptico (acepta "CH01"
        .. "CH04" o un entero) escribiendo un patrón de 2 bits sobre las
        2 líneas digitales de `self.daq_task` (`CHAN_PER_LINE`, ver
        `_init_daq`). No hace nada si el DAQ no está disponible (ver
        docstring de `_init_daq`).

        El orden de bit (`mapping` de abajo) es una implementación
        independiente de `Parte1.py::OpticalSwitchController.set_channel`
        (que usa `CHAN_FOR_ALL_LINES` + un entero, no una lista de
        booleanos por línea) -- ambos codifican el mismo canal 1-4, pero
        no se verificó en esta revisión que asignen el mismo bit físico a
        MSB/LSB que la otra implementación. Ver la misma ambigüedad
        MSB/LSB sin resolver documentada en
        config.py::OpticalSwitchConfig y docs/DeveloperNotes.md §6.
        """
        if self.daq_task is None:
            return
        try:
            # Extrae el numero del string tipo "CH01" o acepta int
            n = int(str(channel_str).replace("CH", "").replace("ch", ""))
            mapping = {1: [False, False], 2: [True, False],
                       3: [False, True],  4: [True, True]}
            self.daq_task.write(mapping.get(n, [False, False]))
        except Exception as e:
            print_warn(f"Error al cambiar canal: {e}")

    def set_polarizer(self, p1: int, p2: int, p3: int):
        """
        Mueve las 3 paletas a los ángulos indicados. No hace nada si el
        polarizador no está disponible (ver docstring de
        `_init_polarizer`). El `time.sleep(1)` tras cada `MoveTo` es la
        misma espera empírica de asentamiento mecánico documentada en
        `_init_polarizer` -- no reducir sin hardware real conectado.
        """
        if self.polarizer is None:
            return
        try:
            for angle, paddle in [
                    (p1, PolarizerPaddles.Paddle1),
                    (p2, PolarizerPaddles.Paddle2),
                    (p3, PolarizerPaddles.Paddle3)]:
                self.polarizer.MoveTo(Decimal(angle), paddle, 60000)
                time.sleep(1)
        except Exception as e:
            print_warn(f"Error al mover polarizador: {e}")

    def live_preview(self, title: str, instructions: list) -> bool:
        """
        Muestra la cámara en vivo con instrucciones superpuestas y espera
        confirmación del operador: [S] continuar, [Q] abortar.

        Es el punto de control experimental del pipeline: se invoca antes
        del dark frame y antes de cada grabación, para que el operador
        verifique con sus propios ojos que el haz está centrado, enfocado
        y sin saturar, y que las condiciones de turbulencia (ΔT,
        ventiladores) ya se estabilizaron. Ninguna de esas comprobaciones
        puede automatizarse de forma fiable, y capturar sin ellas produce
        datos inservibles que solo se detectarían al analizar.

        Retorna True si el operador confirmó, False si abortó — en cuyo
        caso el llamador cancela el experimento completo.
        """
        from gui.live_view import _ascii as _asc
        window = _asc(title)
        shared = {"frame": None, "running": True}

        def cb(cam, stream, frame):
            # Callback de streaming: guarda el último frame recibido en
            # `shared` para que el bucle de la ventana lo muestre. Se
            # descarta el frame anterior sin procesarlo (la vista previa
            # solo necesita mostrar el más reciente, no todos).
            if frame.get_status() == vmbpy.FrameStatus.Complete and shared["running"]:
                img = frame.as_numpy_ndarray()
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img.squeeze()
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                shared["frame"] = img.copy()
            stream.queue_frame(frame)

        self.camera.start_streaming(handler=cb, buffer_count=5)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 900, 700)
        print(f"\n  [{window}]  [S] Continuar   [Q] Abortar experimento")
        font   = cv2.FONT_HERSHEY_SIMPLEX
        result = True

        while True:
            if shared["frame"] is not None:
                disp = shared["frame"].copy()
                h, w = disp.shape[:2]
                ov   = disp.copy()
                ph   = 42 + 26 * max(len(instructions), 1)
                cv2.rectangle(ov, (0, 0), (w, ph), (20, 20, 20), -1)
                cv2.addWeighted(ov, 0.65, disp, 0.35, 0, disp)
                y = 30
                cv2.putText(disp, _asc(title), (8, y), font, 0.72,
                            (0, 200, 255), 2, cv2.LINE_AA); y += 30
                for ln in instructions:
                    cv2.putText(disp, _asc(ln), (8, y), font, 0.54,
                                (50, 200, 50), 1, cv2.LINE_AA); y += 24
                cv2.rectangle(disp, (0, h-38), (w, h), (20, 20, 20), -1)
                cv2.putText(disp, "  [S] Continuar      [Q] Abortar experimento",
                            (8, h-14), font, 0.60,
                            (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow(window, disp)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("s"), ord("S")):
                break
            elif key in (ord("q"), ord("Q"), 27):
                result = False; break

        shared["running"] = False
        self.camera.stop_streaming()
        cv2.destroyWindow(window); cv2.waitKey(1)
        return result

    def capture_frame(self) -> np.ndarray:
        """
        Captura una sola imagen: arranca el streaming, se queda con el
        primer frame completo que llegue y lo detiene.

        Se usa para el dark frame de la sesión. Retorna el array en crudo
        (sin normalizar ni anotar) o None si no llegó ningún frame en 5 s
        — respaldo empírico para no bloquear indefinidamente si la cámara
        deja de entregar.
        """
        captured = [None]
        def cb(cam, stream, frame):
            if frame.get_status() == vmbpy.FrameStatus.Complete:
                img = frame.as_numpy_ndarray()
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img.squeeze()
                captured[0] = img.copy()
            stream.queue_frame(frame)
        self.camera.start_streaming(handler=cb, buffer_count=3)
        t0 = time.time()
        while captured[0] is None and time.time() - t0 < 5.0:
            time.sleep(0.02)
        self.camera.stop_streaming()
        return captured[0]

    def record_video(self, filepath: str,
                     duration_s: int, codec: str, fmt: str) -> list:
        """
        Graba el video de un caso experimental durante `duration_s`
        segundos y lo escribe en `filepath`. Retorna la lista de
        (timestamp, frame) capturados, que el llamador usa para calcular
        el fps REAL y guardar el respaldo .npz.

        Arquitectura productor/consumidor (decisión de diseño crítica):
        el callback de la cámara es deliberadamente MÍNIMO —solo serializa
        el frame a bytes y lo encola—, y un hilo trabajador aparte
        reconstruye los arrays NumPy. Si el callback hiciera ese trabajo,
        bloquearía el hilo de adquisición de la cámara y se PERDERÍAN
        frames, reduciendo el fps efectivo.

        Por qué importa científicamente: el fps efectivo determina la
        resolución temporal del beam wander. Perder frames no solo
        produce menos datos, sino que sesga la estadística del vagabundeo
        hacia las componentes lentas, alterando la varianza que alimenta
        el cálculo de r₀ y Cₙ².
        """
        buf = []; shape_ref = [None]

        import threading as _thr
        _copy_queue = []
        _lock = _thr.Lock()

        def cb(cam, stream, frame):
            # Callback MINIMO: convierte a bytes en C y encola para copia fuera del GIL
            if frame.get_status() == vmbpy.FrameStatus.Complete:
                t   = time.time()
                raw = frame.as_numpy_ndarray()
                # tobytes() es mas rapido que copy() para serializar en el callback
                with _lock:
                    _copy_queue.append((t, raw.tobytes(), raw.shape, raw.dtype))
            stream.queue_frame(frame)

        def worker():
            """Hilo separado: reconstruye arrays numpy fuera del callback."""
            while shared_run[0] or _copy_queue:
                with _lock:
                    items = _copy_queue[:]
                    del _copy_queue[:len(items)]
                for t, raw_bytes, shp, dt in items:
                    arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shp)
                    if arr.ndim == 3 and arr.shape[2] == 1:
                        arr = arr.squeeze()
                    buf.append((t, arr))
                    if shape_ref[0] is None:
                        shape_ref[0] = arr.shape
                if not items:
                    time.sleep(0.001)

        shared_run = [True]
        _worker_thread = _thr.Thread(target=worker, daemon=True)
        _worker_thread.start()
        # buffer_count alto: permite que la camara siga enviando
        # mientras el worker procesa en paralelo
        self.camera.start_streaming(handler=cb, buffer_count=25)
        print("  ⏺ Grabando", end="", flush=True)
        t0 = time.time(); last_dot = -1
        while time.time() - t0 < duration_s:
            sec = int(time.time() - t0)
            if sec > last_dot:
                print(".", end="", flush=True); last_dot = sec
            time.sleep(0.02)
        self.camera.stop_streaming()
        shared_run[0] = False          # señal al worker para terminar
        _worker_thread.join(timeout=30.0)  # esperar a que drene la cola
        if _worker_thread.is_alive():
            print("  Advertencia: worker timeout — puede haber frames perdidos")

        n = len(buf)
        print_ok(f"({n} frames en {time.time()-t0:.1f} s)")
        if n < 2 or shape_ref[0] is None:
            print_error("Frames insuficientes.")
            return []

        span = buf[-1][0] - buf[0][0]
        fps  = (n - 1) / span if span > 0 else 1.0
        H, W = shape_ref[0]
        writer = cv2.VideoWriter(filepath,
                                  cv2.VideoWriter_fourcc(*codec),
                                  fps, (W, H))
        frames_gray = []
        for idx, (_, img) in enumerate(buf):
            frames_gray.append(img)
            vis = (cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)
                   if img.dtype != np.uint8 else img.copy())
            bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr, f"{idx+1}/{n}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(bgr)
        writer.release()
        print(f"     Guardado: {os.path.basename(filepath)}")
        # Guardar frames crudos sin perdida (opcional, ver guardar_npz en el
        # setup — ocupa mas espacio/tiempo).
        # np.save (no comprimido) es mucho mas rapido que savez_compressed
        # y garantiza que los datos se graben incluso en archivos grandes
        if _cam("guardar_npz", True):
            npz_path = filepath.rsplit(".", 1)[0] + ".npz"
            try:
                np.savez(npz_path,
                         frames=np.array(frames_gray, dtype=frames_gray[0].dtype),
                         fps=np.float64(fps),
                         timestamps=np.array([t for t, _ in buf], dtype=np.float64))
                print(f"     Guardado: {os.path.basename(npz_path)} (datos crudos, {n} frames)")
            except Exception as _npz_err:
                print(f"     ADVERTENCIA: no se pudo guardar .npz: {_npz_err}")
                print("     El .mp4 sí fue guardado correctamente.")
        return frames_gray


# =============================================================================
# FUNCIONES DE ANALISIS
# =============================================================================

def _to2d(frame):
    """Convierte cualquier frame a 2D float64 para calculos de momentos."""
    img = frame.copy()
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img.squeeze()
        else:
            img = cv2.cvtColor(img.astype(np.uint8)
                               if img.dtype != np.uint8
                               else img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def _umbralizar(img, frac=0.05):
    """Ver utils_beam_metrics.umbralizar (misma función, reexportada aquí)."""
    return _beam_metrics.umbralizar(img, frac=frac)


def _centroide(frame, umbral_frac=0.05):
    """
    Centroide ponderado por intensidad (ver utils_beam_metrics.centroide).
    Para beam wander (seguimiento de posicion) usa umbral_frac=0 (sin umbral).
    Para W0 (medicion de tamano) usa umbral_frac=0.05 (ISO 11146).
    """
    return _beam_metrics.centroide(_to2d(frame), umbral_frac=umbral_frac)


def _segundo_momento(frame, cx, cy, umbral_frac=0.05):
    """
    Segundo momento espacial con umbral ISO 11146 (ver
    utils_beam_metrics.segundo_momento). Solo integra pixeles por encima
    de frac*I_max para evitar que el fondo oscuro o los anillos de
    difraccion inflen sigma.
    """
    return _beam_metrics.segundo_momento(_to2d(frame), cx, cy, umbral_frac=umbral_frac)



def calcular_W_fibre(first_frame, um_per_px):
    """W0 = 2*sigma_prom*delta [m] (metodo directo). Retorna (W0_m, detalle)."""
    cx, cy    = _centroide(first_frame)
    sx, sy    = _segundo_momento(first_frame, cx, cy)
    sigma_avg = (sx + sy) / 2.0
    W0_m = 2.0 * sigma_avg * um_per_px * 1e-6
    return W0_m, {"cx_px": cx, "cy_px": cy, "sx_px": sx, "sy_px": sy,
                   "sx_um": sx * um_per_px, "sy_um": sy * um_per_px,
                   "W0_um": W0_m * 1e6}


def graficar_w0_directo(frame, detalle, um_per_px, ruta, ruta_txt=None):
    """Figura de verificacion de W0 con calidad de publicacion."""
    import matplotlib.pyplot as plt

    frm = frame.copy()
    if frm.ndim == 3:
        frm = frm.squeeze() if frm.shape[2] == 1 else cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    vis = cv2.normalize(frm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    H, W = vis.shape[0], vis.shape[1]

    cx    = float(detalle.get("cx_px", W/2))
    cy    = float(detalle.get("cy_px", H/2))
    sx    = float(detalle.get("sx_px", 0))
    sy    = float(detalle.get("sy_px", 0))
    W0_um = float(detalle.get("W0_um", 0))

    if cx == 0 and cy == 0 or (sx == 0 and sy == 0):
        cx, cy = _centroide(frm.astype(np.float64))
        sx, sy = _segundo_momento(frm.astype(np.float64), cx, cy)
        W0_um  = 2.0 * (sx + sy) / 2.0 * um_per_px

    r1s    = (sx + sy) / 2.0
    r2s    = max(r1s * 2.0, 1.0)
    margin = max(int(r2s * 2.5), 30)
    zx1 = max(0, int(cx) - margin); zy1 = max(0, int(cy) - margin)
    zx2 = min(W, int(cx) + margin); zy2 = min(H, int(cy) + margin)
    zoom = vis[zy1:zy2, zx1:zx2]

    theta = np.linspace(0, 2*np.pi, 360)
    plt.rcParams.update({"figure.dpi": 300, "font.family": "serif", "font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")

    for ax, img, ex1, ey1, ex2, ey2 in [
        (ax1, vis, 0, 0, W*um_per_px, H*um_per_px),
        (ax2, zoom, zx1*um_per_px, zy1*um_per_px, zx2*um_per_px, zy2*um_per_px)
    ]:
        ax.imshow(img, cmap="inferno", origin="upper", aspect="equal",
                  vmin=0, vmax=255, extent=[ex1, ex2, ey2, ey1])
        ax.plot(cx*um_per_px, cy*um_per_px, "r+", ms=12, mew=2, zorder=5)
        ax.plot((cx + sx*np.cos(theta))*um_per_px,
                (cy + sy*np.sin(theta))*um_per_px,
                "--", color="cyan", lw=1.5, label=r"1$\sigma$", zorder=4)
        ax.plot((cx + 2*sx*np.cos(theta))*um_per_px,
                (cy + 2*sy*np.sin(theta))*um_per_px,
                "-", color="lime", lw=2.0,
                label=r"$W_0$ = 2$\sigma$ = " + f"{W0_um:.2f} µm", zorder=4)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    cx_um = cx * um_per_px; cy_um = cy * um_per_px
    ax1.set_title("Imagen completa  |  Centro = (" + f"{cx_um:.1f}, {cy_um:.1f}) µm")
    ax2.set_title("Zoom del haz  |  " + r"$W_0$ = " + f"{W0_um:.3f} µm" +
                  r"  |  $\delta$ = " + f"{um_per_px:.5f} µm/px")
    ax1.legend(loc="upper right", framealpha=0.85, fontsize=10)

    fig.suptitle(r"Medición de $W_0$ - método directo (D4$\sigma$)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    # Companero numerico de la figura W0 (carpeta separada si se indica)
    _ruta_txt = ruta_txt if ruta_txt else ruta.rsplit(".", 1)[0] + "_datos.txt"
    os.makedirs(os.path.dirname(os.path.abspath(_ruta_txt)), exist_ok=True)
    with open(_ruta_txt, "w", encoding="utf-8") as _fh:
        _fh.write("Medición de W₀ - método directo (D4σ)\n")
        _fh.write(f"cx_px       = {cx:.4f}\n")
        _fh.write(f"cy_px       = {cy:.4f}\n")
        _fh.write(f"sigma_x_px  = {sx:.4f}\n")
        _fh.write(f"sigma_y_px  = {sy:.4f}\n")
        _fh.write(f"um_per_px   = {um_per_px:.5f}\n")
        _fh.write(f"W0_um       = {W0_um:.4f}\n")
    print(f"  Figura W0 guardada: {os.path.basename(ruta)}")


def calcular_W0(focal_m, lam_m, W_fibre_m):
    """
    W0 = f1 * lambda / (pi * W_fibre)  -- método indirecto (propagación
    gaussiana), ver MathematicalReference.md §15.

    Parámetros: focal_m (distancia focal de la lente colimadora, [m]),
    lam_m (longitud de onda, [m]), W_fibre_m (radio del campo modal
    medido con MedicionWFibre.py, [m]). Retorna W0 en metros. No valida
    W_fibre_m > 0 -- un valor 0 produce ZeroDivisionError, responsabilidad
    del llamador (ver validación en gui/turbulence_dialog.py).
    """
    return focal_m * lam_m / (math.pi * W_fibre_m)


def despejar_r0(r_c2_px2, um_per_px, W0_m, lam_m, L_m):
    """
    Despeja r0 de la formula de beam wander:
        <r_c^2> = 0.54*L^2*(lambda/2W0)^2*(2W0/r0)^(5/3)
    Ver MathematicalReference.md §16.

    Parámetros: r_c2_px2 (varianza del beam wander en px², suma de
    varianzas x+y), um_per_px (calibración de píxel), W0_m/lam_m/L_m
    (radio de referencia, longitud de onda, distancia de propagación, en
    metros). Retorna r0 en metros.

    No lanza excepción. Si el beam wander es indistinguible de cero (sin
    turbulencia medible) o el numerador no es positivo (W0 inválido),
    retorna `float("inf")` en vez de fallar -- ver el caso especial que
    distingue esto de un error real en
    `analizar_video_from_centroids` (Cn2=0.0 vs Cn2=NaN).
    """
    m_per_px = um_per_px * 1e-6
    r_c2_m2  = r_c2_px2 * m_per_px**2
    if r_c2_m2 <= 1e-30:   # sin beam wander medible → r0 → infinito
        return float("inf")
    numer = 0.54 * L_m**2 * (lam_m / (2*W0_m))**2 * (2*W0_m)**(5/3)
    if numer <= 0:
        return float("inf")
    return float((numer / r_c2_m2)**(3/5))


def calcular_Cn2(r0_m, lam_m, L_m):
    """
    Cn2 = r0^(-5/3) / (0.4234 * k^2 * L)  -- parámetro de estructura del
    índice de refracción [m^(-2/3)], ver MathematicalReference.md §17.
    Parámetros en metros. Si r0_m es `inf` (sin turbulencia medible, ver
    `despejar_r0`), retorna 0.0 (ver el manejo explícito de este caso
    especial en `analizar_video_from_centroids`, no aquí).
    """
    k = 2 * math.pi / lam_m
    return r0_m**(-5/3) / (0.4234 * k**2 * L_m)


def calcular_sigma_R(Cn2, lam_m, L_m):
    """
    sigma_R = 1.23 * Cn2 * k^(7/6) * L^(11/6) -- varianza de Rytov
    (índice de scintillation débil, adimensional), ver
    MathematicalReference.md §18. Cn2 en m^(-2/3), lam_m/L_m en metros.
    """
    k = 2 * math.pi / lam_m
    return 1.23 * Cn2 * k**(7/6) * L_m**(11/6)


def analizar_video_from_centroids(xs, ys, first_frame, session=None,
                                   timestamps=None, t_var_s=25.0):
    """
    Procesa los centroides (xs, ys) ya extraidos de un video (por el
    llamador, con umbral_frac=0.0 via _centroide) y retorna el diccionario
    de parametros de turbulencia (W0, r0, Cn2, sigma_R, ...).

    Recibe los centroides ya calculados -en vez de los frames crudos- para
    que el llamador pueda recorrer el video una sola vez; ver el bucle de
    lectura en main_analisis (mas abajo en este archivo) y
    ComparacionTurbulencia._cargar_barrido_video.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size == 0:
        return {}

    lam_m     = float(getattr(getattr(session, "experiment", None),
                               "wavelength", "980").replace(" nm","")) * 1e-9
    focal_m   = getattr(getattr(session, "turbulence", None),
                         "focal_length_mm", 11.0) * 1e-3
    L_m       = getattr(getattr(session, "turbulence", None),
                         "distance_m", 1.0)
    um_per_px = getattr(getattr(session, "scale", None), "um_per_px", 2.2)

    # W0: elegir metodo segun configuracion
    metodo_w0 = getattr(getattr(session, "turbulence", None), "metodo_w0", "w_fibre")
    if metodo_w0 == "directo":
        # Usar W0 fijo de referencia si esta disponible (calculado de DeltaT=0, v=0)
        _w0_fijo = getattr(session, "_w0_fijo_m", None)
        if _w0_fijo is not None and _w0_fijo > 0:
            W_fibre_m = _w0_fijo
        else:
            _res_wf   = calcular_W_fibre(first_frame, um_per_px)
            W_fibre_m = _res_wf[0] if isinstance(_res_wf, tuple) else float(_res_wf)
        W0_m = W_fibre_m
        print(f"     W₀ (directo D4σ) = {W0_m*1e6:.4f} µm")
        # Figura W0 se genera externamente desde el frame de referencia
    else:
        w_fibre_um = getattr(getattr(session, "turbulence", None), "w_fibre_um", 0.0)
        if not w_fibre_um or w_fibre_um <= 0:
            raise ValueError(
                "W_fibre no definido. Introduce el valor en Parámetros del haz o usa el método directo.")
        W_fibre_m = w_fibre_um * 1e-6
        W0_m      = calcular_W0(focal_m, lam_m, W_fibre_m)
        print(f"     W_fibre = {w_fibre_um:.4f} µm  |  W₀ = {W0_m*1e6:.4f} µm")

    # Varianza del beam wander, excluyendo el transitorio inicial (t_var_s),
    # ver §6.2 de la documentacion tecnica: distintos casos ΔT/velocidad
    # pueden tardar tiempos distintos en alcanzar el regimen termico estable.
    N = len(xs)
    if timestamps is not None and len(timestamps) == N:
        t = np.asarray(timestamps, dtype=np.float64) - float(timestamps[0])
    else:
        t = np.arange(N) / 30.0   # fallback 30 fps, igual que graficar_desplazamiento_temporal

    mask = t >= t_var_s
    if not mask.any():
        print(f"     Aviso: t_var_s={t_var_s:.1f}s excede la duracion del video "
              f"({t[-1]:.1f}s); se usa el video completo para la varianza.")
        mask = np.ones(N, dtype=bool)

    var_x = float(np.var(xs[mask]))
    var_y = float(np.var(ys[mask]))
    r_c2  = var_x + var_y

    r0_m       = despejar_r0(r_c2, um_per_px, W0_m, lam_m, L_m)
    # Sin beam wander medible (mismo umbral que usa despejar_r0 internamente
    # para decidir r0->infinito) corresponde fisicamente a Cn2->0 (turbulencia
    # nula), no a un valor indefinido -- se reporta 0.0, no NaN, para no
    # confundirlo con un fallo de medicion en los CSV que consume la Opcion 9.
    _r_c2_m2_check = r_c2 * (um_per_px * 1e-6) ** 2
    if _r_c2_m2_check <= 1e-30:
        Cn2     = 0.0
        sigma_R = 0.0
    elif math.isfinite(r0_m):
        Cn2     = calcular_Cn2(r0_m, lam_m, L_m)
        sigma_R = calcular_sigma_R(Cn2, lam_m, L_m) if math.isfinite(Cn2) else float("nan")
    else:
        # r0 no finito por otra razon (ej. W0 invalido) -- se mantiene NaN
        # como indicador de error real, distinto del caso "sin turbulencia".
        Cn2     = float("nan")
        sigma_R = float("nan")

    return {
        "W_fibre_um": W_fibre_m * 1e6,
        "W0_um":      W0_m * 1e6,
        "metodo_w0":  metodo_w0,
        "r_c2_um2":   r_c2 * (um_per_px**2),
        "var_x_um2":  var_x * (um_per_px**2),
        "var_y_um2":  var_y * (um_per_px**2),
        "r0_mm":      r0_m * 1e3,
        "Cn2":        Cn2,
        "sigma_R":    sigma_R,
        "n_frames":   N,
        "xs_px":      xs,
        "ys_px":      ys,
    }


# =============================================================================
# GRAFICAS Y REPORTE
# =============================================================================

def _color_vel(i, n):
    """
    Asigna un color al i-ésimo valor de velocidad de aire de un barrido de
    `n` velocidades, tomándolo del colormap perceptualmente uniforme
    `plasma`. Así, en las gráficas r₀/Cₙ²/σ_R vs ΔT, el color codifica la
    velocidad de forma ordenada y legible incluso en escala de grises.
    """
    # Recorta el extremo alto (amarillo claro) de plasma para que todos
    # los puntos contrasten bien sobre el fondo blanco de las graficas.
    frac = i / max(n - 1, 1)
    return cm.plasma(0.85 * frac)[:3]


def generar_graficas(results, carpeta_graficas, carpeta_datos):
    """
    Genera las 3 gráficas r0/Cn2/sigma_R vs ΔT (una velocidad por color,
    ver `_color_vel`), con CSV hermano en `carpeta_datos` para cada una.
    `results` es el dict anidado {delta_T: {velocity: metricas_dict}}
    devuelto por `analizar_video_from_centroids` para cada caso.
    """
    delta_T_list  = sorted(results.keys())
    velocity_list = sorted({v for dt in results for v in results[dt]})
    n_vel         = len(velocity_list)

    specs = [
        ("01_r0_vs_deltaT.png",     "r0_mm",     r"$r_0$  (mm)",
         r"Parametro de Fried  $r_0$  vs  $\Delta T$", False),
        ("02_Cn2_vs_deltaT.png",    "Cn2",        "$C_n^2$  (m$^{-2/3}$)",
         r"Coeficiente de estructura del indice de refraccion  $C_n^2$  vs  $\Delta T$", True),
        ("03_sigmaR_vs_deltaT.png", "sigma_R",   r"$\sigma_R$  (adim.)",
         r"Varianza de Rytov  $\sigma_R$  vs  $\Delta T$", False),
    ]
    for fname, key, ylabel, title, logy in specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, vel in enumerate(velocity_list):
            dts = [dt for dt in delta_T_list if vel in results[dt]]
            ys  = [results[dt][vel].get(key, float("nan")) for dt in dts]
            c   = _color_vel(i, n_vel)
            if logy:
                ax.semilogy(dts, ys, "o", color=c, ms=7,
                            label=f"{vel:.1f} m/s")
            else:
                ax.plot(dts, ys, "o", color=c, ms=7,
                        label=f"{vel:.1f} m/s")
        if key == "sigma_R":
            ax.axhline(1.0, color="red", ls="--", lw=1.2,
                       label=r"$\sigma_R$ = 1  (limite Rytov)")
        ax.set_xlabel(r"$\Delta T$ (°C)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(title="Velocidad del aire", fontsize=10)
        ax.grid(True, ls="--", alpha=0.6, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(carpeta_graficas, fname),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        # Guardar CSV con los datos de la gráfica, en carpeta separada
        csv_fname = fname.replace(".png", ".csv")
        import csv as _csv
        with open(os.path.join(carpeta_datos, csv_fname), "w",
                  newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["delta_T_C"] + [f"vel_{v:.1f}_ms" for v in velocity_list])
            for dt in delta_T_list:
                row_vals = [dt]
                for vel in velocity_list:
                    row_vals.append(
                        results[dt][vel].get(key, float("nan"))
                        if vel in results.get(dt, {}) else float("nan"))
                w.writerow([f"{v:.8g}" for v in row_vals])
        print(f"  Guardada: {fname}")




# Alias del helper compartido (ver utils_carpetas.py) — se mantiene el
# nombre original para no tocar sus ~15 sitios de uso en este archivo.
_carpeta_salida_segura = _utils_carpetas.carpeta_salida_segura


def graficar_desplazamiento_temporal(m, timestamps, um_per_px,
                                      delta_T, velocity, carpeta, carpeta_datos, tag,
                                      t_ref_s=1.0):
    """
    Grafica el desplazamiento radial del centroide vs tiempo para un caso.
    Desplazamiento radial r(t) = sqrt((cx(t)-<cx>)^2 + (cy(t)-<cy>)^2) en um.
    """
    xs = np.asarray(m.get("xs_px", []))
    ys = np.asarray(m.get("ys_px", []))
    if xs.size == 0:
        print("  [" + tag + "] Sin datos de centroide para graficar")
        return

    N = len(xs)
    if timestamps is not None and len(timestamps) == N:
        t = np.asarray(timestamps, dtype=float) - float(timestamps[0])
    else:
        t = np.arange(N) / 30.0   # fallback 30 fps

    # Referencia: centroide promedio de los frames en [0, t_ref_s]
    if timestamps is not None and len(timestamps) == N and t_ref_s > 0:
        _t_rel  = np.asarray(timestamps, float) - float(timestamps[0])
        _mask_r = _t_rel <= t_ref_s
        cx_ref  = float(xs[_mask_r].mean()) if _mask_r.any() else float(xs[0])
        cy_ref  = float(ys[_mask_r].mean()) if _mask_r.any() else float(ys[0])
    else:
        n_ref  = max(1, int(t_ref_s * 30))  # approx frames if no timestamps
        cx_ref = float(xs[:n_ref].mean())
        cy_ref = float(ys[:n_ref].mean())
    r_px   = np.sqrt((xs - cx_ref)**2 + (ys - cy_ref)**2)
    r_um   = r_px * um_per_px

    r_rms = float(np.sqrt(np.mean(r_um**2)))
    r_max = float(r_um.max())

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.plot(t, r_um, lw=0.7, color="steelblue", alpha=0.75, label="r(t)")
    ax.fill_between(t, 0, r_um, alpha=0.12, color="steelblue")
    ax.axhline(r_rms, color="crimson", ls="--", lw=1.6,
               label=r"$r_{RMS}$ = " + f"{r_rms:.3f} µm")

    ax.set_xlabel("Tiempo (s)", fontsize=12)
    ax.set_ylabel("Desplazamiento radial r(t)  (µm)", fontsize=12)
    titulo = ("Desplazamiento radial del centroide vs tiempo  |  "
              r"$\Delta T$ = " + f"{delta_T:.1f}" + " °C  v=" + f"{velocity:.2f}" + " m/s"
              "  N=" + str(N) + r"  $r_{max}$=" + f"{r_max:.3f}" + " µm"
              "  ref=0-" + f"{t_ref_s:.1f}" + "s")
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    fname = "04_wander_temporal_" + tag + ".png"
    fpath = os.path.join(carpeta, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Guardada: " + fname)

    # CSV con serie temporal completa, en carpeta separada de la grafica
    import csv as _csv
    dx_um = (xs - cx_ref) * um_per_px
    dy_um = (ys - cy_ref) * um_per_px
    csv_path = os.path.join(carpeta_datos, fname.replace(".png", ".csv"))
    with open(csv_path, "w",
              newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["t_s", "cx_px", "cy_px", "dx_um", "dy_um", "r_um"])
        for ti, xi, yi, dxi, dyi, ri in zip(t, xs, ys, dx_um, dy_um, r_um):
            w.writerow([f"{ti:.6f}", f"{xi:.4f}", f"{yi:.4f}",
                        f"{dxi:.4f}", f"{dyi:.4f}", f"{ri:.4f}"])


def graficar_dispersion_centroide_wander(m, timestamps, um_per_px,
                                          delta_T, velocity, carpeta, carpeta_datos, tag,
                                          t_var_s=25.0):
    """
    Mapa de dispersión del centroide (scatter Δx vs Δy coloreado por
    tiempo) + histograma de r(t) con ajuste de distribución Rayleigh y
    prueba de bondad de ajuste Kolmogorov-Smirnov (valor p corregido por
    bootstrap paramétrico -- ver Parte3.ks_pvalue_bootstrap_rayleigh).
    Mismo diseño que Parte3.graficar_desplazamiento_centroide, aplicado
    aquí al beam wander de un caso de la cámara de turbulencia.

    Útil para verificar visualmente/estadísticamente que la turbulencia
    produce un patrón de dispersión aproximadamente Rayleigh en 2D --
    hipótesis física estándar detrás de la fórmula de beam wander
    (⟨r_c²⟩ = 0.54·L²·(λ/2W₀)²·(2W₀/r₀)^(5/3)) que usa este programa para
    despejar r₀/Cₙ².

    Usa el MISMO subconjunto de frames (t >= t_var_s) y la MISMA
    referencia (promedio del centroide sobre ese subconjunto) que
    analizar_video_from_centroids -- para que esta figura verifique la
    hipótesis Rayleigh sobre los mismos datos que efectivamente
    determinan r0/Cn2/sigma_R. Antes se referenciaba al promedio de los
    primeros segundos e incluía el video completo, lo que mezclaba el
    asentamiento térmico/mecánico inicial (no aleatorio) con el jitter de
    turbulencia en régimen estable ya excluido de r0/Cn2 -- verificado
    contra datos reales: para un caso con turbulencia fuerte, las
    excursiones más grandes del centroide (hasta ~400 µm, muy por encima
    del RMS en régimen) ocurrían todas dentro de los primeros t_var_s
    segundos, sesgando el ajuste Rayleigh y el test KS con una dinámica
    que no es la turbulencia que el resto del análisis está midiendo.
    """
    xs = np.asarray(m.get("xs_px", []))
    ys = np.asarray(m.get("ys_px", []))
    if xs.size < 2:
        print("  [" + tag + "] Sin datos de centroide suficientes para el mapa de dispersión")
        return

    N = len(xs)
    if timestamps is not None and len(timestamps) == N:
        t = np.asarray(timestamps, dtype=float) - float(timestamps[0])
    else:
        t = np.arange(N) / 30.0

    # Mismo criterio de exclusión del transitorio térmico inicial que
    # analizar_video_from_centroids (ver esa función para el detalle) --
    # si t_var_s excede la duración del video, se usa el video completo.
    mask = t >= t_var_s
    if not mask.any():
        print(f"     Aviso: t_var_s={t_var_s:.1f}s excede la duracion del video "
              f"({t[-1]:.1f}s); se usa el video completo para el mapa de dispersion.")
        mask = np.ones(N, dtype=bool)

    xs_m = xs[mask]; ys_m = ys[mask]; t_m = t[mask]
    cx_ref = float(xs_m.mean())
    cy_ref = float(ys_m.mean())

    dx = (xs_m - cx_ref) * um_per_px
    dy = (ys_m - cy_ref) * um_per_px
    dist_eucl = np.sqrt(dx ** 2 + dy ** 2)
    rms      = float(np.sqrt(np.mean(dist_eucl ** 2)))
    dx_max   = float(np.max(np.abs(dx)))
    dy_max   = float(np.max(np.abs(dy)))
    dist_max = float(dist_eucl.max())
    lim = max(np.abs(np.concatenate([dx, dy])).max(), 1e-6) * 1.15

    # ── Ajuste de Rayleigh (MLE) + KS con valor p corregido por bootstrap ────
    sigma_ray = float(np.sqrt(np.mean(dist_eucl ** 2) / 2.0))
    try:
        from scipy.stats import kstest
        import Parte3 as _P3
        ks_stat, _ = kstest(dist_eucl, "rayleigh", args=(0, sigma_ray))
        ks_p = _P3.ks_pvalue_bootstrap_rayleigh(dist_eucl, sigma_ray, ks_stat)
    except Exception:
        ks_stat, ks_p = float("nan"), float("nan")

    from scipy.stats import rayleigh as rayleigh_dist
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (ax_sc, ax_hi) = plt.subplots(1, 2, figsize=(14, 7),
                                        gridspec_kw={"wspace": 0.35})

    # ── Panel izquierdo: scatter Δx vs Δy ────────────────────────────────────
    sc = ax_sc.scatter(dx, dy, c=t_m, cmap="viridis", s=40, zorder=3,
                       label="Frames analizados")
    ax_sc.scatter(0, 0, color="red", s=140, zorder=4, marker="*",
                  label=f"Referencia (media, t≥{t_var_s:.1f}s)")
    div = make_axes_locatable(ax_sc)
    cax = div.append_axes("right", size="5%", pad=0.12)
    fig.colorbar(sc, cax=cax).set_label("Tiempo (s)", fontsize=11)
    ax_sc.annotate(
        f"RMS        = {rms:.3f} µm\n"
        f"Δx máx.    = {dx_max:.3f} µm\n"
        f"Δy máx.    = {dy_max:.3f} µm\n"
        f"Dist. máx. = {dist_max:.3f} µm",
        xy=(0.04, 0.96), xycoords="axes fraction",
        fontsize=11, fontweight="bold", color="#111111", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.92))
    ax_sc.set_xlim(-lim, lim);  ax_sc.set_ylim(-lim, lim)
    ax_sc.set_aspect('equal', adjustable='box')
    ax_sc.set_title(f"Mapa de dispersión del centroide\n(t ≥ {t_var_s:.1f}s, transitorio térmico excluido)",
                    fontsize=12, fontweight="bold")
    ax_sc.set_xlabel("$\\Delta x$ (µm)", fontsize=12)
    ax_sc.set_ylabel("$\\Delta y$ (µm)", fontsize=12)
    ax_sc.legend(fontsize=11, loc="lower right")
    ax_sc.grid(True, linestyle="--", alpha=0.6)

    # ── Panel derecho: histograma de r(t) + curva Rayleigh ───────────────────
    n_bins = max(15, min(50, len(dist_eucl) // 20))
    ax_hi.hist(dist_eucl, bins=n_bins, density=True,
              color="#4C72B0", edgecolor="#CCCCCC", linewidth=0.6,
              alpha=0.80, label="Distribución empírica")
    r_plot  = np.linspace(0, dist_eucl.max() * 1.15, 400)
    pdf_ray = rayleigh_dist.pdf(r_plot, scale=sigma_ray)
    ax_hi.plot(r_plot, pdf_ray, color="#DD4444", lw=2.2,
              label=f"Ajuste Rayleigh\n$\\sigma_R$ = {sigma_ray:.3f} µm")
    ax_hi.axvline(sigma_ray, color="#DD4444", lw=1.2, ls="--", alpha=0.7)
    ax_hi.axvline(rms, color="#888888", lw=1.2, ls=":", alpha=0.8,
                 label=f"RMS = {rms:.3f} µm")

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
        bbox=dict(boxstyle="round,pad=0.35", fc="#eeeeee", ec="#aaaaaa", alpha=0.9))

    ax_hi.set_title(
        "Distribución estadística del beam wander\n"
        "$r(t) = \\sqrt{\\Delta x^2 + \\Delta y^2}$",
        fontsize=12, fontweight="bold")
    ax_hi.set_xlabel("Distancia radial $r$ (µm)", fontsize=12)
    ax_hi.set_ylabel("Densidad de probabilidad", fontsize=12)
    # loc="upper right" (no "upper left"): el pico de la distribucion
    # Rayleigh siempre cae en la zona baja-media de r, nunca en la cola
    # derecha -- "upper left" quedaba encima de las barras mas altas del
    # histograma, tapando la forma real de la distribucion. La cola
    # derecha (r grande) es, por construccion del eje (xlim hasta
    # dist_eucl.max()*1.15), siempre de densidad baja/nula, así que esa
    # esquina queda libre para cualquier caso ΔT/velocidad.
    ax_hi.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax_hi.grid(True, linestyle="--", alpha=0.5)
    ax_hi.set_xlim(left=0)

    titulo_extra = (r"$\Delta T$ = " + f"{delta_T:.1f}" + " °C  v="
                    + f"{velocity:.2f}" + " m/s")
    fig.suptitle("Mapa de dispersión y bondad de ajuste Rayleigh  —  " + titulo_extra,
                fontsize=14, fontweight="bold", y=1.02)

    fname = "05_dispersion_rayleigh_" + tag + ".png"
    fpath = os.path.join(carpeta, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Guardada: " + fname)

    resumen_path = os.path.join(carpeta_datos, fname.replace(".png", "_resumen.txt"))
    with open(resumen_path, "w", encoding="utf-8") as fh:
        fh.write(f"rms_um: {rms:.6g}\n")
        fh.write(f"dx_max_um: {dx_max:.6g}\n")
        fh.write(f"dy_max_um: {dy_max:.6g}\n")
        fh.write(f"dist_max_um: {dist_max:.6g}\n")
        fh.write(f"sigma_rayleigh_um: {sigma_ray:.6g}\n")
        fh.write(f"ks_stat: {ks_stat:.6g}\n")
        fh.write(f"ks_pvalue_bootstrap: {ks_p:.6g}\n")
        fh.write(f"t_var_s: {t_var_s:.6g}\n")
        fh.write(f"n_frames_incluidos: {int(mask.sum())}\n")
        fh.write(f"n_frames_totales: {N}\n")


def guardar_info_experimento(carpeta, session=None):
    """
    Genera 00_info_experimento.txt con todos los parámetros de configuración
    del experimento tal como fueron configurados por el usuario.
    Equivalente al reporte de cabecera que generan Parte1 y Parte2.
    """
    from datetime import datetime
    exp  = getattr(session, "experiment",  None)
    turb = getattr(session, "turbulence",  None)
    sc   = getattr(session, "scale",       None)
    cam  = getattr(session, "camera",      None)
    pol  = getattr(session, "polarizer",   None)

    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    lineas = [
        "=" * 72,
        "  INFORMACIÓN DEL EXPERIMENTO — CARACTERIZACIÓN DE CÁMARA DE TURBULENCIA",
        "=" * 72, "",
        f"  Fecha y hora          : {now}",
        "",
        "── Experimento ─────────────────────────────────────────────────────────",
        f"  Canal del switch      : {getattr(exp, 'channel', '?')}",
        f"  Longitud de onda      : {getattr(exp, 'wavelength', '?')}",
        f"  Temperatura del láser : {getattr(exp, 'temperature', '?')} °C",
        f"  Corriente del láser   : {getattr(exp, 'current', '?')} mA",
        "",
        "── Parámetros ópticos del haz ──────────────────────────────────────────",
        f"  Focal de la lente f₁  : {getattr(turb, 'focal_length_mm', '?')} mm",
        f"  Distancia L           : {getattr(turb, 'distance_m', '?')} m",
        f"  Tamaño de píxel       : {getattr(sc, 'um_per_px', '?'):.5f} µm/px",
        "",
        "── Adquisición ─────────────────────────────────────────────────────────",
        f"  Exposición            : {getattr(cam, 'exposure_time', '?')} µs",
        f"  Ganancia              : {getattr(cam, 'gain', '?')} dB",
        f"  Duración del video    : {getattr(turb, 'video_duration_s', '?')} s",
        f"  Codec de video        : {getattr(turb, 'video_codec', '?')}",
        "",
        "── Polarización (posición fija) ────────────────────────────────────────",
        f"  Paddle 1              : {getattr(turb, 'paddle1', '?')}°",
        f"  Paddle 2              : {getattr(turb, 'paddle2', '?')}°",
        f"  Paddle 3              : {getattr(turb, 'paddle3', '?')}°",
        f"  Serial del polarizador: {getattr(pol, 'serial', '?')}",
        "",
        "── Barrido experimental ─────────────────────────────────────────────────",
        f"  ΔT analizados         : {getattr(turb, 'delta_T_list', '?')}",
        f"  Velocidades analizadas: {getattr(turb, 'velocity_list', '?')}",
        f"  Total de combinaciones: "
            f"{len(getattr(turb,'delta_T_list',[]))*len(getattr(turb,'velocity_list',[]))}",
        "",
        "=" * 72,
    ]
    ruta = os.path.join(carpeta, "00_info_experimento.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    print("  Guardado: 00_info_experimento.txt")


def guardar_reporte_txt(results, carpeta, session=None):
    """
    Guarda 00_resultados_numericos.txt: tabla de ancho fijo, pensada para
    lectura humana, con todos los parámetros ópticos de la sesión y una
    fila por caso (ΔT, velocidad, W_fibre, W0, r_c², r0, Cn2, sigma_R,
    n_frames). Ver `guardar_reporte_csv` para el equivalente máquina-
    legible que consume la Opción 9.
    """
    exp = getattr(session, "experiment", None)
    turb = getattr(session, "turbulence", None)
    sc   = getattr(session, "scale", None)

    lineas = [
        "=" * 72,
        "  CARACTERIZACIÓN DE CÁMARA DE TURBULENCIA ÓPTICA",
        "=" * 72, "",
        f"  Longitud de onda : {getattr(exp, 'wavelength', '?')}",
        f"  Temperatura      : {getattr(exp, 'temperature', '?')} °C",
        f"  Corriente        : {getattr(exp, 'current', '?')} mA",
        f"  Focal f₁         : {getattr(turb, 'focal_length_mm', '?')} mm",
        f"  Distancia L      : {getattr(turb, 'distance_m', '?')} m",
        f"  µm/px            : {getattr(sc, 'um_per_px', '?'):.5f}",
        "",
        f"  {'ΔT (°C)':>9}  {'Vel (m/s)':>10}  "
        f"{'W_fibre(µm)':>12}  {'W₀(µm)':>9}  "
        f"{'rc²(µm²)':>10}  {'r₀(mm)':>8}  "
        f"{'Cₙ²(m^-2/3)':>14}  {'σ_R':>10}  {'N':>7}",
        "  " + "-" * 96,
    ]
    for dt in sorted(results.keys()):
        for vel in sorted(results[dt].keys()):
            m = results[dt][vel]
            lineas.append(
                f"  {dt:>9.1f}  {vel:>10.2f}  "
                f"{m.get('W_fibre_um', float('nan')):>12.4f}  "
                f"{m.get('W0_um', float('nan')):>9.4f}  "
                f"{m.get('r_c2_um2', float('nan')):>10.4f}  "
                f"{m.get('r0_mm', float('nan')):>8.4f}  "
                f"{m.get('Cn2', float('nan')):>14.4e}  "
                f"{m.get('sigma_R', float('nan')):>10.6f}  "
                f"{m.get('n_frames', 0):>7d}"
            )
    lineas += ["", "=" * 72, "  FIN DEL REPORTE", "=" * 72]
    ruta = os.path.join(carpeta, "00_resultados_numericos.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    print("  Guardado: 00_resultados_numericos.txt")


def guardar_reporte_csv(results, carpeta):
    """
    Complemento agregable en CSV de guardar_reporte_txt: una fila por caso
    con todos los escalares de turbulencia (W_fibre, W0, r_c2, r0, Cn2,
    sigma_R, n_frames). A diferencia del .txt (ancho fijo, pensado para
    lectura humana), este CSV existe para que la Opcion 9 (comparacion de
    mediciones de camara de turbulencia) pueda leer estos datos sin volver
    a abrir ningun video.
    """
    import csv as _csv
    ruta = os.path.join(carpeta, "00_resultados_numericos.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["delta_T_C", "velocity_ms", "tag", "W_fibre_um", "W0_um",
                    "r_c2_um2", "r0_mm", "Cn2", "sigma_R", "n_frames"])
        for dt in sorted(results.keys()):
            for vel in sorted(results[dt].keys()):
                m = results[dt][vel]
                w.writerow([
                    f"{dt:.8g}", f"{vel:.8g}", m.get("tag", ""),
                    f"{m.get('W_fibre_um', float('nan')):.8g}",
                    f"{m.get('W0_um', float('nan')):.8g}",
                    f"{m.get('r_c2_um2', float('nan')):.8g}",
                    f"{m.get('r0_mm', float('nan')):.8g}",
                    f"{m.get('Cn2', float('nan')):.8g}",
                    f"{m.get('sigma_R', float('nan')):.8g}",
                    m.get("n_frames", 0),
                ])
    print("  Guardado: 00_resultados_numericos.csv")


# =============================================================================
# SELECCIÓN DE ROI Y PREPROCESADO DE FRAMES
# =============================================================================
# La selección de ROI se hace con Parte2.SelectorRecorte -- la MISMA
# interfaz que usa el preprocesado del haz óptico (Opción 2) -- para no
# mantener dos interfaces distintas para la misma tarea. Ver su uso en
# main_preprocesado() más abajo.

import json as _json
import types as _types

def _guardar_metadata_json(carpeta, metadata):
    """
    Escribe `metadata_adquisicion.json` en `carpeta`. Es el archivo del
    que depende toda la trazabilidad del pipeline de turbulencia: las
    Opciones 5, 6 y 9 leen de él los parámetros ópticos y la lista de
    casos, sin volver a preguntar nada al usuario.

    Se reescribe tras CADA caso (no solo al final del barrido) para que
    una sesión interrumpida conserve el registro de lo ya adquirido.
    """
    import math
    def _clean(v):
        """Convierte NaN a null recursivamente: JSON estándar no admite
        NaN, y escribirlo produciría un archivo que otras herramientas no
        pueden leer."""
        if isinstance(v, float) and math.isnan(v): return None
        if isinstance(v, dict):  return {k: _clean(val) for k, val in v.items()}
        if isinstance(v, list):  return [_clean(i) for i in v]
        return v
    ruta = os.path.join(carpeta, "metadata_adquisicion.json")
    with open(ruta, "w", encoding="utf-8") as f:
        _json.dump(_clean(metadata), f, ensure_ascii=False, indent=2)

def main_adquisicion(session=None):
    """
    Opcion 4 — Adquisicion: dark frame + grabacion de videos caso por caso.
    Guarda .mp4 + .npz crudo en Adquisicion/. No hace preprocesado ni analisis.
    """
    global SESSION
    SESSION = session
    from gui.turbulencia_caso_dialog import pedir_caso

    print_banner("ADQUISICION — CAMARA DE TURBULENCIA (caso por caso)")

    # Cargar librerias de hardware (camara, polarizador, DAQ)
    if not _load_hardware_libs():
        print("  Error: librerias de hardware no disponibles. Verificar instalacion.")
        return None

    # ── Carpeta raiz ──────────────────────────────────────────────────────────
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    base = filedialog.askdirectory(
        title="Carpeta de destino para la caracterizacion", parent=root)
    root.destroy()
    if not base:
        print("Carpeta de destino no seleccionada."); return None

    fecha        = datetime.now().strftime("%Y%m%d")
    nombre_base  = f"{fecha}_CaracterizacionCamaraTurbulencia"
    carpeta_raiz = _utils_carpetas.carpeta_raiz_segura(base, nombre_base)
    carpeta, _, _ = _utils_carpetas.crear_triple_raiz(carpeta_raiz)
    # Nota: carpeta (Adquisicion) siempre es nueva (exist_ok=False dentro de
    # crear_triple_raiz). Preprocesado/Analisis se crean vacias junto con
    # ella en este flujo, por lo que no hay riesgo de colision aqui.
    print(f"Carpeta raiz: {carpeta_raiz}")

    # ── Metadatos iniciales ───────────────────────────────────────────────────
    um_per_px = _scale("um_per_px", 2.2)
    exp       = _cam("exposure_time", 10000)
    gain      = _cam("gain", 10.0)
    pf        = _cam("pixel_format", "Mono8")
    codec     = _turb("video_codec", "mp4v")
    fmt       = _turb("video_format", "mp4")
    duration  = _turb("video_duration_s", 30)
    paddle1   = _turb("paddle1", 80)
    paddle2   = _turb("paddle2", 80)
    paddle3   = _turb("paddle3", 80)

    metadata = {
        "fecha":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experimento": {
            "channel":     _exp("channel",     "?"),
            "temperature": _exp("temperature", "?"),
            "current":     _exp("current",     "?"),
        },
        "optica":     {
            "wavelength_nm":   _exp("wavelength", "980").replace(" nm",""),
            "focal_length_mm": _turb("focal_length_mm", 11.0),
            "distance_m":      _turb("distance_m",      1.0),
            "metodo_w0":       _turb("metodo_w0",       "directo"),
            "w_fibre_um":      _turb("w_fibre_um",      0.0),
            "w0_um":           None,
        },
        "camara":     {
            "exposure_time_us": exp,
            "gain_db":          gain,
            "pixel_format":     pf,
            "um_per_px":        um_per_px,
            "guardar_npz":      _cam("guardar_npz", True),
        },
        "adquisicion": {
            "paddle1":          paddle1,
            "paddle2":          paddle2,
            "paddle3":          paddle3,
            "video_duration_s": duration,
            "video_codec":      codec,
            "video_format":     fmt,
        },
        "darkframe":  "darkframe.png",
        "roi":        None,
        "casos":      [],
    }

    ctrl = TurbController(session)
    with ctrl:
        if ctrl.camera is None:
            print("No se pudo inicializar la camara."); return None

        # Fijar la polarizacion y el canal del switch optico a los valores
        # configurados -- antes de este fix, el polarizador se quedaba en su
        # posicion Home() y el switch en el estado de la operacion anterior,
        # pese a que la metadata reportaba estos valores como aplicados.
        # set_polarizer()/set_channel() ya manejan el caso de hardware no
        # conectado sin lanzar excepcion (retornan de inmediato si
        # ctrl.polarizer/ctrl.daq_task son None, igual que el resto del
        # controlador) -- por eso la adquisicion sigue funcionando aunque el
        # switch optico no este presente en el arreglo.
        ctrl.set_polarizer(paddle1, paddle2, paddle3)
        if ctrl.polarizer is not None:
            print_ok(f"Polarizador movido a P1={paddle1}°  P2={paddle2}°  P3={paddle3}°")

        canal = _exp("channel", "?")
        ctrl.set_channel(canal)
        if ctrl.daq_task is not None:
            print_ok(f"Canal del switch óptico fijado a {canal}")

        # ── FASE 1: Ajuste del haz y dark frame ───────────────────────────────
        print("\n" + "═"*68)
        print("  FASE 1 — AJUSTE Y DARK FRAME")
        print("═"*68)

        if not ctrl.live_preview(
                "Ajuste del haz gaussiano",
                [f"Polarización: P1={paddle1}  P2={paddle2}  P3={paddle3}",
                 "Asegúrate de que el haz esté centrado y enfocado."]):
            return None

        # Dark frame (antes de alterar temperatura/velocidad) — mismo flujo
        # que la Opcion 1 (Parte1.py::run_stage0_dark_frame): una sola
        # ventana [S]/[Q], captura automatica del primer frame completo
        # tras presionar [S] (sin messagebox Si/No ni ventana [ESPACIO]
        # aparte, para que la interfaz de dark frame sea igual en ambas
        # opciones de adquisicion).
        #
        # LIMITACION DE DISENO CONOCIDA (documentada, no un bug): a
        # diferencia del pipeline del haz optico -- donde cada caso
        # (SinTurbulencia/Transitorio/ConTurbulencia) tiene su propio dark
        # frame, ver utils_carpetas.crear_carpetas_caso() -- aqui se captura
        # UN SOLO dark frame al inicio de FASE 1 y se reutiliza para TODOS
        # los casos DT<x>_vel<y> del resto de la sesion (que puede abarcar
        # horas, con muchas combinaciones ΔT/velocidad). Esto es consecuencia
        # de que el pipeline de turbulencia no tiene la capa de "casos" del
        # pipeline optico -- no hay una carpeta por caso donde anidar un dark
        # frame propio. Si la temperatura del sensor deriva de forma
        # significativa durante una sesion larga, el dark frame inicial
        # podria dejar de representar bien el offset/ruido de fondo hacia el
        # final. Recomendacion: mencionar esta limitacion metodologica en la
        # tesis si se reportan sesiones de caracterizacion muy largas.
        if not ctrl.live_preview(
                "Dark Frame — Apaga el laser antes de continuar",
                ["APAGUE el laser antes de continuar.",
                 "Se capturara 1 imagen de fondo (sin senal).",
                 "Presiona [S] cuando el laser este apagado y listo."]):
            return None
        dark_raw = ctrl.capture_frame()
        if dark_raw is not None:
            _utils_imagenes.guardar_imagen(os.path.join(carpeta, "darkframe.png"), dark_raw)
            if _cam("guardar_npz", True):
                np.savez_compressed(os.path.join(carpeta, "darkframe.npz"),
                                    dark_frame=dark_raw)
                print("  Dark frame guardado (PNG + NPZ).")
            else:
                print("  Dark frame guardado (PNG).")
        else:
            print_warn("No se pudo capturar el dark frame.")

        # ── FASE 2: Grabacion de videos caso por caso ─────────────────────────
        print("\n" + "═"*68)
        print("  FASE 2 — GRABACION DE VIDEOS")
        print("═"*68)

        n_caso = 0; dt_prev = 0.0; vel_prev = 0.0
        t_ref_prev = 1.0; t_var_prev = 25.0

        while True:
            n_caso += 1
            resultado = pedir_caso(n_caso, dt_prev, vel_prev, t_ref_prev, t_var_prev)
            if resultado is None:
                print("\n  Usuario finalizo la grabacion de casos.")
                break
            delta_T, velocity, t_ref_s, t_var_s = resultado
            dt_prev = delta_T; vel_prev = velocity
            t_ref_prev = t_ref_s; t_var_prev = t_var_s

            print(f"\n[Caso {n_caso}]  DeltaT={delta_T}C  v={velocity}m/s")
            tag_base = f"DT{delta_T}_vel{velocity}"
            tag      = tag_base
            # PROTECCION CRITICA: si ya existe un video con este mismo tag
            # (ej. mismo DeltaT/velocidad introducido dos veces), NUNCA
            # sobreescribir — se anexa un sufijo para preservar ambos.
            _dup = 2
            while os.path.exists(os.path.join(carpeta, tag,
                                               f"video_{tag}.{fmt}")):
                tag = f"{tag_base}_rep{_dup}"
                _dup += 1
            if tag != tag_base:
                print(f"  AVISO: ya existia un video para {tag_base}.")
                print(f"  Este caso se guardara como '{tag}' para NO "
                      f"sobreescribir el anterior.")
            subcarpeta = os.path.join(carpeta, tag)
            os.makedirs(subcarpeta, exist_ok=True)
            filepath   = os.path.join(subcarpeta, f"video_{tag}.{fmt}")

            if not ctrl.live_preview(
                    f"Caso {n_caso}: DeltaT={delta_T}C  v={velocity}m/s",
                    ["Asegúrate de que las condiciones sean estables."]):
                break

            try:
                ctrl.record_video(filepath, duration, codec, fmt)
            except Exception as _rec_err:
                print(f"  ERROR grabando caso {n_caso}: {_rec_err}")
                print("  El caso sera omitido pero los anteriores estan guardados.")
                continue  # intentar el siguiente caso

            metadata["casos"].append({
                "n": n_caso, "delta_T_C": delta_T, "velocity_ms": velocity,
                "carpeta": tag, "video": f"video_{tag}.{fmt}",
                "t_ref_s": t_ref_s, "t_var_s": t_var_s,
            })
            _guardar_metadata_json(carpeta, metadata)  # guardar tras cada caso

    print_banner(f"ADQUISICION COMPLETADA — {carpeta_raiz}")
    return carpeta_raiz


def main_preprocesado(session=None, carpeta_adq=None):
    """
    Preprocesado: lee .mp4 de adquisicion, aplica ROI + dark frame,
    guarda _proc.mp4 en Preprocesado/. Rapido: un frame a la vez.
    """
    global SESSION; SESSION = session
    print("\n" + "═"*68)
    print("  FASE: PREPROCESADO — CAMARA DE TURBULENCIA")
    print("  (ROI + resta dark frame  |  fuente: .mp4)")
    print("═"*68)

    # ── Seleccion de carpeta raiz ─────────────────────────────────────────────
    if carpeta_adq is None:
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        messagebox.showinfo("Seleccionar carpeta",
            "Selecciona la carpeta raiz de la caracterizacion\n"
            "(la que contiene Adquisicion/, Preprocesado/ y Analisis/).",
            parent=root)
        carpeta_adq = filedialog.askdirectory(
            title="Carpeta raiz de la caracterizacion", parent=root)
        root.destroy()
    if not carpeta_adq:
        return None

    # ── Detectar estructura ───────────────────────────────────────────────────
    _bname = os.path.basename(os.path.abspath(carpeta_adq))
    if _bname == "Adquisicion":
        _raiz = os.path.dirname(carpeta_adq)
    elif os.path.exists(os.path.join(carpeta_adq, "Adquisicion")):
        _raiz = carpeta_adq
        carpeta_adq = os.path.join(carpeta_adq, "Adquisicion")
    else:
        _raiz = carpeta_adq

    carpeta_preproc = _carpeta_salida_segura(os.path.join(_raiz, "Preprocesado"))
    os.makedirs(carpeta_preproc, exist_ok=True)

    # ── Leer metadata ─────────────────────────────────────────────────────────
    meta_path = os.path.join(carpeta_adq, "metadata_adquisicion.json")
    if not os.path.exists(meta_path):
        print(f"  No se encontro metadata_adquisicion.json en: {carpeta_adq}")
        return None
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = _json.load(fh)

    # ── Seleccion de casos (DT_vel) ───────────────────────────────────────────
    # Igual que Parte2.py (Opcion 2): siempre se pregunta explicitamente con
    # cuales casos trabajar, incluso si solo hay uno disponible.
    casos_unicos = list({c["carpeta"]: c for c in metadata.get("casos", [])
                         if c.get("carpeta")}.values())
    _tags_disponibles = [c["carpeta"] for c in casos_unicos]
    if not _tags_disponibles:
        print_error("La adquisicion no tiene ningun caso (DT_vel) guardado.")
        return None
    from gui.seleccion_casos_dialog import pedir_casos_multiples
    _tags_elegidos = pedir_casos_multiples(_tags_disponibles,
                                            preseleccionados=_tags_disponibles)
    if not _tags_elegidos:
        print_error("No se selecciono ningun caso. Preprocesado cancelado.")
        return None
    _tags_elegidos_set = set(_tags_elegidos)
    casos_unicos = [c for c in casos_unicos if c["carpeta"] in _tags_elegidos_set]

    # ── Dark frame ────────────────────────────────────────────────────────────
    # Preprocesado trabaja unicamente con .mp4/.png; el .npz es solo un
    # respaldo crudo dentro de Adquisicion y no se usa en el flujo normal.
    dark_raw = None
    _dp = os.path.join(carpeta_adq, "darkframe.png")
    if os.path.exists(_dp):
        dark_raw = _utils_imagenes.leer_imagen(_dp, cv2.IMREAD_UNCHANGED)
        if dark_raw is not None:
            if dark_raw.ndim == 3:
                dark_raw = dark_raw.squeeze()
            print("  Dark frame cargado: darkframe.png")
    else:
        print_warn("No se encontro darkframe.png — se continua sin restar dark frame.")

    # ── ROI ───────────────────────────────────────────────────────────────────
    # Se pide SIEMPRE al usuario, igual que Parte2.py (Opcion 2): la ROI no
    # se cachea entre corridas de preprocesado, para que el manejo del ROI
    # sea exactamente igual que en la caracterizacion del haz optico.
    roi = None
    dark_recortado = None
    _primer_frame = None
    if casos_unicos:
        _vid0 = os.path.join(carpeta_adq, casos_unicos[0]["carpeta"],
                              f"video_{casos_unicos[0]['carpeta']}.mp4")
        if os.path.exists(_vid0):
            _cap0 = cv2.VideoCapture(_vid0)
            _ret0, _fr0 = _cap0.read(); _cap0.release()
            if _ret0:
                _primer_frame = cv2.cvtColor(_fr0, cv2.COLOR_BGR2GRAY) if _fr0.ndim==3 else _fr0
                print("  Selecciona la ROI en la ventana:")
                from Parte2 import SelectorRecorte
                roi = SelectorRecorte(_primer_frame).seleccionar()
    if roi is None:
        print("  Sin ROI — preprocesado cancelado")
        return None

    x1, y1, x2, y2 = roi
    print(f"  ROI: ({x1},{y1}) -> ({x2},{y2})  [{x2-x1}x{y2-y1} px]")
    if dark_raw is not None:
        _dk = dark_raw[y1:y2, x1:x2]
        dark_recortado = (_dk.squeeze() if _dk.ndim==3 else _dk).astype(np.float32)

    # Guardar ROI en metadata
    metadata["roi"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    _guardar_metadata_json(carpeta_adq,    metadata)
    _guardar_metadata_json(carpeta_preproc, metadata)

    # ── Procesar cada caso: .mp4 → _proc.mp4 (frame a frame, minima RAM) ─────
    n_ok = 0
    for caso in casos_unicos:
        tag    = caso["carpeta"]
        mp4_in = os.path.join(carpeta_adq, tag, f"video_{tag}.mp4")
        if not os.path.exists(mp4_in):
            print(f"  [{tag}] .mp4 no encontrado"); continue

        sub_out = os.path.join(carpeta_preproc, tag)
        os.makedirs(sub_out, exist_ok=True)
        mp4_out = os.path.join(sub_out, f"video_{tag}_proc.mp4")

        cap = cv2.VideoCapture(mp4_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        wr  = None; n = 0

        print(f"  [{tag}]", end="", flush=True)
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Aplicar ROI
            frame = frame[y1:y2, x1:x2]
            # Restar dark frame
            if dark_recortado is not None:
                frame = np.clip(frame.astype(np.float32) - dark_recortado,
                                0, None).astype(np.uint8)
            # Inicializar writer con el tamano del primer frame procesado
            if wr is None:
                h, w = frame.shape[:2]
                wr = cv2.VideoWriter(mp4_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            wr.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            n += 1

        cap.release()
        if wr: wr.release()
        print(f" OK ({n} frames) -> {os.path.basename(mp4_out)}")
        n_ok += 1

    print(f"\n  Completado: {n_ok}/{len(casos_unicos)} casos en {carpeta_preproc}")
    return carpeta_preproc


def main_analisis(session=None, carpeta_adq=None):
    """
    Análisis de la cámara de turbulencia.
    Pide la carpeta RAÍZ de la caracterización y pregunta explícitamente si
    se desea trabajar con Adquisicion o Preprocesado — el análisis usa los
    frames tal como están en la carpeta elegida, sin aplicar ROI ni dark
    frame adicional (eso es responsabilidad exclusiva de la Opción 5).
    """
    global SESSION
    SESSION = session

    if carpeta_adq is None:
        from gui.eleccion_carpeta_dialog import pedir_eleccion_carpeta

        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        messagebox.showinfo(
            "Carpeta raíz",
            "Selecciona la carpeta RAÍZ de la caracterización\n"
            "(contiene Adquisicion/, Preprocesado/ y Analisis/).",
            parent=root)
        carpeta_raiz = filedialog.askdirectory(
            title="Carpeta raíz de la caracterización", parent=root)
        root.destroy()
        if not carpeta_raiz:
            return None
        carpeta_raiz = _utils_carpetas.normalizar_carpeta_raiz(carpeta_raiz)

        eleccion = pedir_eleccion_carpeta(carpeta_raiz)
        if not eleccion:
            return None
        carpeta_adq = os.path.join(carpeta_raiz, eleccion)
    if not carpeta_adq:
        return None

    # ── Leer metadata ─────────────────────────────────────────────────────────
    meta_path = os.path.join(carpeta_adq, "metadata_adquisicion.json")
    if not os.path.exists(meta_path):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        messagebox.showerror("Error",
            f"No se encontró metadata_adquisicion.json en:\n{carpeta_adq}\n\n"
            "Asegúrate de seleccionar una carpeta _Adq o _Preproc válida.",
            parent=root)
        root.destroy()
        return None

    with open(meta_path, "r", encoding="utf-8") as fh:
        import json as _json
        metadata = _json.load(fh)

    print(f"  Carpeta: {carpeta_adq}")
    print(f"  Casos:   {len(metadata.get('casos', []))}")

    # ── Seleccion de casos (DT_vel) ───────────────────────────────────────────
    # Igual que Parte3.py (Opcion 3): siempre se pregunta explicitamente con
    # cuales casos trabajar, incluso si solo hay uno disponible.
    _tags_disponibles = list({c["carpeta"]: None for c in metadata.get("casos", [])
                              if c.get("carpeta")}.keys())
    if not _tags_disponibles:
        print_error("No hay ningun caso (DT_vel) en la metadata.")
        return None
    from gui.seleccion_casos_dialog import pedir_casos_multiples
    _tags_elegidos = pedir_casos_multiples(_tags_disponibles,
                                            preseleccionados=_tags_disponibles)
    if not _tags_elegidos:
        print_error("No se selecciono ningun caso. Analisis cancelado.")
        return None
    _tags_elegidos_set = set(_tags_elegidos)
    metadata["casos"] = [c for c in metadata.get("casos", [])
                         if c.get("carpeta") in _tags_elegidos_set]

    # ── Parámetros ópticos desde metadata ─────────────────────────────────────
    camd = metadata.get("optica", {})
    cam2 = metadata.get("camara", {})
    um_per_px = float(cam2.get("um_per_px", _turb("um_per_px", 2.2)))

    _exp_meta = metadata.get("experimento", {})
    _cam_meta = metadata.get("camara", {})
    _adq_meta = metadata.get("adquisicion", {})
    _delta_T_list  = sorted({c["delta_T_C"]  for c in metadata.get("casos", []) if "delta_T_C"  in c})
    _velocity_list = sorted({c["velocity_ms"] for c in metadata.get("casos", []) if "velocity_ms" in c})
    session_fake = _types.SimpleNamespace(
        turbulence=_types.SimpleNamespace(
            wavelength_nm   = float(camd.get("wavelength_nm",   980)),
            focal_length_mm = float(camd.get("focal_length_mm", 11.0)),
            distance_m      = float(camd.get("distance_m",      1.0)),
            metodo_w0       = camd.get("metodo_w0",       "directo"),
            w_fibre_um      = float(camd.get("w_fibre_um",      0.0)),
            paddle1          = _adq_meta.get("paddle1",          "?"),
            paddle2          = _adq_meta.get("paddle2",          "?"),
            paddle3          = _adq_meta.get("paddle3",          "?"),
            video_duration_s = _adq_meta.get("video_duration_s", "?"),
            video_codec      = _adq_meta.get("video_codec",      "?"),
            delta_T_list     = _delta_T_list,
            velocity_list    = _velocity_list,
        ),
        scale=_types.SimpleNamespace(um_per_px=um_per_px),
        experiment=_types.SimpleNamespace(
            wavelength  = str(camd.get("wavelength_nm", "?")) + " nm",
            channel     = _exp_meta.get("channel",     metadata.get("channel", "?")),
            temperature = _exp_meta.get("temperature", metadata.get("temperature", "?")),
            current     = _exp_meta.get("current",     metadata.get("current", "?")),
        ),
        camera=_types.SimpleNamespace(
            exposure_time = _cam_meta.get("exposure_time_us", "?"),
            gain          = _cam_meta.get("gain_db",          "?"),
            pixel_format  = _cam_meta.get("pixel_format",     "?"),
        ),
        polarizer=_types.SimpleNamespace(serial="?"),
        _w0_fig_path=None,
    )

    # ── Carpeta de resultados ─────────────────────────────────────────────────
    # Buscar la carpeta raiz (puede que el usuario haya seleccionado Adquisicion/)
    _bname = os.path.basename(os.path.abspath(carpeta_adq))
    if _bname in ("Adquisicion", "Preprocesado"):
        _carpeta_raiz = os.path.dirname(os.path.abspath(carpeta_adq))
    elif os.path.exists(os.path.join(carpeta_adq, "Adquisicion")):
        _carpeta_raiz = carpeta_adq
        carpeta_adq   = os.path.join(carpeta_adq, "Adquisicion")
    else:
        _carpeta_raiz = carpeta_adq
    carpeta_out = _carpeta_salida_segura(os.path.join(_carpeta_raiz, "Analisis"))
    os.makedirs(carpeta_out, exist_ok=True)
    # Carpeta independiente para los datos numericos (CSV/TXT), separada de
    # las graficas/videos — misma filosofia que Parte3._carpeta_datos_crudos.
    carpeta_datos = os.path.join(carpeta_out, _utils_carpetas.NOMBRE_SUBCARPETA_DATOS_CRUDOS)
    os.makedirs(carpeta_datos, exist_ok=True)
    print(f"  Resultados: {carpeta_out}")
    print(f"  Datos numericos: {carpeta_datos}")

    # ── Analizar cada caso ────────────────────────────────────────────────────
    # ── Deduplicar y calcular W0 antes del analisis ─────────────────────
    _tags_seen = set()
    casos_uniq = []
    for _c in metadata.get("casos", []):
        if _c.get("carpeta") and _c["carpeta"] not in _tags_seen:
            _tags_seen.add(_c["carpeta"]); casos_uniq.append(_c)

    # Detectar carpeta de adquisicion una sola vez
    _adq_folder_base = carpeta_adq
    if os.path.basename(os.path.abspath(carpeta_adq)) == "Preprocesado":
        _adq_folder_base = os.path.join(
            os.path.dirname(os.path.abspath(carpeta_adq)), "Adquisicion")

    # W0 desde caso de referencia (DeltaT=0, v=0) — se reutiliza en todos
    session_fake._w0_fijo_m = None
    _ref_exacto = next((c for c in casos_uniq
                        if float(c.get("delta_T_C", 9)) == 0.0
                        and float(c.get("velocity_ms", 9)) == 0.0), None)
    if _ref_exacto is not None:
        _ref = _ref_exacto
    elif casos_uniq:
        _ref = casos_uniq[0]
        print_warn(f"  No se encontro el caso de referencia DeltaT=0, v=0 -- usando "
                   f"'{_ref['carpeta']}' (DeltaT={_ref.get('delta_T_C')}, "
                   f"v={_ref.get('velocity_ms')}) como referencia de W0 en su lugar. "
                   "Si ese caso tiene turbulencia activa, W0 (y por tanto r0/Cn2/sigma_R "
                   "de TODOS los casos analizados) quedara sesgado.")
    else:
        _ref = None
    if _ref:
        _tr   = _ref["carpeta"]
        _mref = os.path.join(carpeta_adq, _tr, f"video_{_tr}_proc.mp4")
        if not os.path.exists(_mref):
            _mref = os.path.join(_adq_folder_base, _tr, f"video_{_tr}.mp4")
        if os.path.exists(_mref):
            _cap0 = cv2.VideoCapture(_mref)
            _ok0, _f0 = _cap0.read(); _cap0.release()
            if _ok0 and _f0 is not None:
                if _f0.ndim == 3: _f0 = cv2.cvtColor(_f0, cv2.COLOR_BGR2GRAY)
                _rw = calcular_W_fibre(_to2d(_f0), um_per_px)
                _w0v = _rw[0] if isinstance(_rw, tuple) else float(_rw)
                _w0d = _rw[1] if isinstance(_rw, tuple) and isinstance(_rw[1], dict) else {}
                session_fake._w0_fijo_m = _w0v
                if _ref_exacto is not None:
                    print(f"  W0 = {_w0v*1e6:.4f} um  (ref: DeltaT=0, v=0)")
                else:
                    print(f"  W0 = {_w0v*1e6:.4f} um  (ref: '{_tr}', DeltaT={_ref.get('delta_T_C')}, "
                          f"v={_ref.get('velocity_ms')} -- SIN caso DeltaT=0/v=0 disponible)")
                try:
                    graficar_w0_directo(
                        _to2d(_f0), _w0d, um_per_px,
                        os.path.join(carpeta_out, "W0_medicion_directa.png"),
                        ruta_txt=os.path.join(carpeta_datos, "W0_medicion_directa_datos.txt"))
                except Exception as _ew: print(f"  Adv W0 fig: {_ew}")

    resultados = []
    for caso in casos_uniq:
        tag = caso["carpeta"]
        _adq_folder = _adq_folder_base
        # Buscar _proc.mp4 primero, si no el .mp4 de adquisicion
        _mp4_proc = os.path.join(carpeta_adq, tag, f"video_{tag}_proc.mp4")
        _mp4_raw  = os.path.join(_adq_folder, tag, f"video_{tag}.mp4")
        if os.path.exists(_mp4_proc):
            video_path = _mp4_proc
            print(f"  [{tag}] video preprocesado")
        elif os.path.exists(_mp4_raw):
            video_path = _mp4_raw
            print(f"  [{tag}] video crudo")
        else:
            print(f"  [{tag}] video no encontrado"); continue

        # VideoCapture: leer frame a frame sin cargar todo en RAM
        _cap_ma = cv2.VideoCapture(video_path)
        if not _cap_ma.isOpened():
            print(f"  [{tag}] no se pudo abrir"); continue
        _fps_real = _cap_ma.get(cv2.CAP_PROP_FPS)
        _xs_ma, _ys_ma, _first_ma = [], [], None
        while True:
            _ret_ma, _fr_ma = _cap_ma.read()
            if not _ret_ma: break
            if _fr_ma.ndim == 3:
                _fr_ma = cv2.cvtColor(_fr_ma, cv2.COLOR_BGR2GRAY)
            if _first_ma is None: _first_ma = _fr_ma
            _cx_i, _cy_i = _centroide(_fr_ma, umbral_frac=0.0)
            _xs_ma.append(_cx_i); _ys_ma.append(_cy_i)
        _cap_ma.release()
        if not _xs_ma:
            print(f"  [{tag}] sin frames"); continue
        print(f"  [{tag}] {len(_xs_ma)} frames leidos")
        # Usar el fps REAL reportado por el .mp4 en vez de asumir 30 fps --
        # con fps incorrecto, t_var_s (exclusion del transitorio termico
        # inicial) se aplicaria sobre un eje de tiempo equivocado,
        # contaminando var_x/var_y/r_c2 y por tanto r0/Cn2/sigma_R.
        if _fps_real and _fps_real > 0:
            _timestamps_ma = np.arange(len(_xs_ma)) / _fps_real
        else:
            print_warn(f"  [{tag}] fps invalido reportado por el video ({_fps_real}); "
                       "usando 30 fps de respaldo.")
            _timestamps_ma = np.arange(len(_xs_ma)) / 30.0
        _t_var_s = float(caso.get("t_var_s", 25.0))
        _t_ref_s = float(caso.get("t_ref_s",  1.0))
        m = analizar_video_from_centroids(
            _xs_ma, _ys_ma, _first_ma, session_fake,
            timestamps=_timestamps_ma, t_var_s=_t_var_s)
        if m:
            m["delta_T"]  = caso["delta_T_C"]
            m["velocity"] = caso["velocity_ms"]
            m["tag"]      = tag
            resultados.append(m)
            graficar_desplazamiento_temporal(
                m, _timestamps_ma, um_per_px,
                caso["delta_T_C"], caso["velocity_ms"],
                carpeta_out, carpeta_datos, tag, t_ref_s=_t_ref_s)
            graficar_dispersion_centroide_wander(
                m, _timestamps_ma, um_per_px,
                caso["delta_T_C"], caso["velocity_ms"],
                carpeta_out, carpeta_datos, tag, t_var_s=_t_var_s)

    if not resultados:
        print("  No se obtuvieron resultados. Verifica que los videos existan.")
        return None

    # Convertir lista a dict anidado {delta_T: {velocity: m}} -- si dos casos
    # seleccionados comparten (delta_T, velocity) -- p. ej. una repeticion
    # "_rep2" del mismo ΔT/velocidad -- solo sobrevive el que se procese
    # ultimo; se avisa explicitamente para que no se pierda de forma
    # silenciosa cual de los dos quedo fuera.
    casos_dict = {}
    for _m in resultados:
        _dt  = _m.get("delta_T",  0.0)
        _vel = _m.get("velocity", 0.0)
        if _vel in casos_dict.get(_dt, {}):
            _tag_anterior = casos_dict[_dt][_vel].get("tag", "?")
            print_warn(f"  Caso duplicado para DeltaT={_dt}, v={_vel}: "
                       f"'{_tag_anterior}' fue reemplazado por '{_m.get('tag', '?')}' "
                       "(mismo DeltaT/velocidad, solo se conserva el ultimo procesado).")
        casos_dict.setdefault(_dt, {})[_vel] = _m

    # ── Generar graficas y reporte ───────────────────────────────────
    generar_graficas(casos_dict, carpeta_out, carpeta_datos)
    guardar_reporte_txt(casos_dict, carpeta_datos, session_fake)
    guardar_reporte_csv(casos_dict, carpeta_datos)
    guardar_info_experimento(carpeta_datos, session_fake)
    print("  Analisis completado: " + carpeta_out)
    return carpeta_out

