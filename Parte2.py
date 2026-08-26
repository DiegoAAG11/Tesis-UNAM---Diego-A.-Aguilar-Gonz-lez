# -*- coding: utf-8 -*-
"""
Analizador de Haz de Fibra Óptica
con Recorte Masivo de Imágenes y Video

Descripción:
Automatiza el recorte y reorganización de imágenes y videos
provenientes de un experimento óptico previamente adquirido.

Metodología:
1) Solicita únicamente la carpeta RAÍZ DEL DISPOSITIVO (contiene los
   casos Sin turbulencia/Transitorio/Con turbulencia, cada uno con su
   propio Adquisicion/Preprocesado/Analisis, con darkframe anidada
   dentro de Adquisicion) mediante GUI.
2) Pregunta siempre con cuáles de los casos EXISTENTES se desea trabajar
   (selección múltiple — no todos los casos existen necesariamente).
3) Por cada caso elegido: verifica sus subcarpetas dentro de Adquisicion/
   (solo estabilidad es obligatoria), selecciona una imagen de referencia
   (_T01) desde estabilidad_temporal, y define una región de recorte
   mediante GUI — el rectángulo se pide UNA sola vez (primer caso) y se
   reutiliza para los demás casos elegidos.
4) Aplica el mismo recorte a imágenes y video en:
   - Diferentes polarizaciones (Haz)               [opcional]
   - Diferentes polarizaciones (Haz + Fibra)        [opcional]
   - Estabilidad temporal                           [obligatoria]
5) Resta el dark frame PROPIO de cada caso (nunca el de otro caso) a
   cada imagen recortada para eliminar ruido.
6) Guarda todo en <raiz>/<caso>/Preprocesado y copia ahí el reporte
   experimental de ese caso.
7) Los archivos ORIGINALES no son modificados en ningún momento.

Autor: Diego Aguilar
Versión: 3.0
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import messagebox, filedialog

from gui.ui_style import crear_botones_confirmar_cancelar
from gui.seleccion_casos_dialog import pedir_casos_multiples
from pathlib import Path
import cv2
import numpy as np

import config
import utils_carpetas
import utils_imagenes
from console_ui import print_banner, print_ok, print_error, print_warn, print_info


# =============================================================================
# PARÁMETROS CONFIGURABLES
# =============================================================================

NOMBRE_REPORTE = "Reporte_Experimento"

SUBCARPETA_HAZ           = "diferentes_polarizaciones_haz"
SUBCARPETA_HAZ_FIBRA     = "diferentes_polarizaciones_haz_fibra"
SUBCARPETA_ESTABILIDAD   = "estabilidad_temporal"

EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXTENSIONES_VIDEO  = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# =============================================================================
# SELECCIÓN INTERACTIVA DE CARPETAS
# =============================================================================

def seleccionar_carpeta_raiz(session=None) -> Path | None:
    """
    Pide al usuario la carpeta RAÍZ DEL DISPOSITIVO (la que contiene los
    casos Sin turbulencia/Transitorio/Con turbulencia) — nunca hay que
    navegar manualmente hasta una subcarpeta.

    Si `session` contiene acquisition_folder, se pre-selecciona esa carpeta
    y se omite el diálogo (pero la elección de con qué caso(s) trabajar
    se pregunta siempre, ver main()).

    Retorna la carpeta raíz (Path) o None si se cancela.
    """
    if session is not None and getattr(session, 'acquisition_folder', None):
        carpeta_raiz = session.acquisition_folder
        print_ok(f"Carpeta raíz del dispositivo (desde sesión): {carpeta_raiz}")
        return Path(carpeta_raiz)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        "Carpeta raíz del dispositivo",
        "Selecciona la carpeta RAÍZ del dispositivo\n"
        "(contiene los casos: Sin turbulencia, Transitorio, Con turbulencia)."
    )
    carpeta_raiz = filedialog.askdirectory(
        title="Selecciona la carpeta raíz del dispositivo")
    root.destroy()
    if not carpeta_raiz:
        messagebox.showerror("Cancelado", "No se seleccionó carpeta raíz.")
        return None

    return Path(utils_carpetas.normalizar_carpeta_raiz_dispositivo(carpeta_raiz))


# =============================================================================
# UTILIDADES
# =============================================================================

def buscar_imagen_referencia(subcarpeta: Path) -> Path | None:
    """
    Elige la imagen sobre la que el usuario dibujará el ROI. Prefiere la
    primera foto de la serie temporal (`_T01`) porque es la que mejor
    representa el encuadre del caso; si no existe (p. ej. la Etapa 1 se
    capturó solo en video), acepta cualquier otra imagen de la carpeta.

    Retorna None si no hay ninguna imagen — el llamador cae entonces a
    extraer el primer frame del video (`extraer_primer_frame_como_imagen`).
    """
    for f in sorted(subcarpeta.iterdir()):
        if f.suffix.lower() in EXTENSIONES_IMAGEN and "_T01" in f.name:
            return f
    for f in sorted(subcarpeta.iterdir()):
        if f.suffix.lower() in EXTENSIONES_IMAGEN:
            return f
    return None


def buscar_video(subcarpeta: Path) -> Path | None:
    """Retorna el primer archivo de video de la carpeta, o None. La
    adquisición genera un único video por caso, así que en el flujo normal
    no hay ambigüedad."""
    for f in subcarpeta.iterdir():
        if f.suffix.lower() in EXTENSIONES_VIDEO:
            return f
    return None


def extraer_primer_frame_como_imagen(video: Path) -> Path | None:
    """
    Extrae el primer frame de `video` y lo guarda como PNG temporal.
    Se usa como imagen de referencia para dibujar la ROI cuando la Etapa 1
    se capturó solo en video (sin fotos fijas). Retorna None si no se pudo
    leer el video.
    """
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    tmp = Path(tempfile.gettempdir()) / f"{video.stem}_frame0_ref.png"
    utils_imagenes.guardar_imagen(tmp, frame)
    return tmp


def buscar_reporte(carpeta_raiz: Path) -> Path | None:
    """Localiza `Reporte_Experimento.txt` en la carpeta de adquisición,
    ignorando la extensión. Se copia junto con los datos preprocesados para
    que la trazabilidad del experimento acompañe a ambas generaciones de
    datos."""
    for f in carpeta_raiz.iterdir():
        if f.stem == NOMBRE_REPORTE:
            return f
    return None


def buscar_subcarpeta_opcional(carpeta_padre: Path, nombre: str) -> Path | None:
    """Retorna el Path si la subcarpeta existe, None si no (sin lanzar error)."""
    candidato = carpeta_padre / nombre
    return candidato if candidato.exists() else None


def buscar_subcarpeta_obligatoria(carpeta_padre: Path, nombre: str) -> Path:
    """Retorna el Path si existe; lanza FileNotFoundError si no."""
    candidato = carpeta_padre / nombre
    if candidato.exists():
        return candidato

    disponibles = [f.name for f in carpeta_padre.iterdir() if f.is_dir()]
    lista = "\n    ".join(disponibles) if disponibles else "(ninguna)"
    raise FileNotFoundError(
        f"\n\n❌ Subcarpeta obligatoria no encontrada: '{nombre}'"
        f"\n   Ruta buscada: {candidato}"
        f"\n\n   Subcarpetas disponibles en '{carpeta_padre.name}':"
        f"\n    {lista}"
        f"\n\n   Ajusta SUBCARPETA_ESTABILIDAD al inicio del script."
    )


# =============================================================================
# DARK FRAME
# =============================================================================

def cargar_dark_frame(carpeta_adq: Path,
                      x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """
    Busca la imagen dentro de la subcarpeta darkframe ANIDADA dentro de
    Adquisicion (cada caso adquiere y usa su propio dark frame, nunca
    compartido con otro caso), la recorta con las mismas coordenadas
    que el resto de imágenes y la devuelve como float32. Retorna None
    si no se puede cargar.
    """
    sub_dark = carpeta_adq / utils_carpetas.NOMBRE_DARKFRAME
    if not sub_dark.exists():
        print_warn(f"Subcarpeta '{utils_carpetas.NOMBRE_DARKFRAME}' no encontrada. "
                   "Se procesará sin sustracción de dark frame.")
        return None

    img_dark_path = None
    for f in sorted(sub_dark.iterdir()):
        if f.suffix.lower() in EXTENSIONES_IMAGEN:
            img_dark_path = f
            break

    if img_dark_path is None:
        print_warn(f"No se encontró imagen en '{utils_carpetas.NOMBRE_DARKFRAME}'. "
                   "Se procesará sin sustracción de dark frame.")
        return None

    dark_bgr = utils_imagenes.leer_imagen(img_dark_path)
    if dark_bgr is None:
        print_warn(f"No se pudo leer el dark frame: {img_dark_path.name}")
        return None

    dark_recortado = dark_bgr[y1:y2, x1:x2].astype(np.float32)
    print(f"  🌑 Dark frame cargado y recortado: {img_dark_path.name}  "
          f"({dark_recortado.shape[1]}×{dark_recortado.shape[0]} px)")
    return dark_recortado


# =============================================================================
# VALIDACIÓN: ROI vs. ZONA DE OVERLAYS
# =============================================================================
#
# Parte1.py quema la barra de escala (esquina inferior izquierda) y el
# texto configurable + etiqueta "<caso> | <dispositivo>" (esquina inferior
# derecha, apilados) sobre el PNG/MP4 de Adquisicion -- ver
# Parte1.py::_add_scale_bar/_add_overlay_text/_add_caso_dispositivo_overlay.
# Esos mismos archivos son los que Parte3.py usa para el análisis
# cuantitativo (centroide, ancho de haz, energía encerrada). Si el usuario
# dibuja aquí un ROI que incluye alguna de esas esquinas, esos píxeles de
# texto quedarían dentro de los datos analizados sin ningún aviso.
#
# NOTA DE MANTENIMIENTO: las constantes de layout de abajo (márgenes,
# fuente, grosor) están duplicadas deliberadamente de Parte1.py, no
# importadas de ahí -- Parte1.py carga DLLs de hardware (Thorlabs) a nivel
# de módulo y Parte2.py debe poder ejecutarse sin ese hardware conectado.
# Si el layout de overlays de Parte1.py cambia alguna vez, hay que
# actualizar también estas funciones.

def _zona_barra_escala(alto: int, ancho: int, scale_cfg) -> tuple | None:
    """Rectángulo (x1,y1,x2,y2) ocupado por la barra de escala + su
    etiqueta (esquina inferior izquierda), o None si está desactivada."""
    if not getattr(scale_cfg, 'enabled', True):
        return None
    margin    = scale_cfg.margin
    bar_h     = scale_cfg.bar_height_px
    bar_um    = scale_cfg.bar_length_um
    px_per_um = getattr(scale_cfg, 'px_per_um', 0.0) or 0.0
    bar_px    = round(bar_um * px_per_um)
    label     = f"{bar_um} um"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   scale_cfg.font_scale, scale_cfg.font_thickness)
    ancho_zona = margin + max(bar_px, tw) + margin
    alto_zona  = margin + bar_h + 4 + th + 6   # +text_gap(4) +colchón(6)
    x1, y1 = 0, max(0, alto - alto_zona)
    x2, y2 = min(ancho, ancho_zona), alto
    return (x1, y1, x2, y2)


def _zona_textos_overlay(alto: int, ancho: int, overlay_cfg,
                         caso_label: str, device_name: str) -> tuple | None:
    """Rectángulo (x1,y1,x2,y2) ocupado por el texto configurable y/o la
    etiqueta "<caso> | <dispositivo>" (esquina inferior derecha, apilados),
    o None si no hay ningún texto que dibujar."""
    margin, font_scale, thickness, line_gap = 8, 1, 1, 6
    textos = []
    texto_usuario = (getattr(overlay_cfg, 'text', '') or '').strip()
    if texto_usuario:
        textos.append(texto_usuario)
    partes = [p for p in (caso_label, device_name) if p]
    if partes:
        textos.append("  |  ".join(partes))
    if not textos:
        return None
    max_w, alto_total = 0, 0
    for i, t in enumerate(textos):
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, tw)
        alto_total += th + (line_gap if i > 0 else 0)
    ancho_zona = margin + max_w + margin
    alto_zona  = margin + alto_total + 6   # +colchón
    x1, y1 = max(0, ancho - ancho_zona), max(0, alto - alto_zona)
    return (x1, y1, ancho, alto)


def _rects_se_superponen(a: tuple, b: tuple) -> bool:
    """Test de intersección entre dos rectángulos (x1,y1,x2,y2). Compara
    por separado la superposición en X y en Y: dos rectángulos se cortan
    solo si se superponen en AMBOS ejes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _advertir_si_roi_toca_overlays(rect: tuple, alto_orig: int, ancho_orig: int,
                                   session, caso_label: str) -> bool:
    """
    Verifica si `rect` (x1,y1,x2,y2, coordenadas de imagen original) se
    superpone con la zona donde Parte1.py dibuja overlays sobre las
    imágenes de Adquisicion. Si hay superposición, avisa y pregunta si se
    desea continuar de todas formas.

    Usa la calibración/config de `session` si está disponible (la de la
    corrida actual); si no (modo standalone o carpeta de otra sesión), usa
    los valores por defecto de config.py -- es una aproximación best-effort,
    no una lectura exacta de cómo se generó originalmente cada imagen.

    Retorna True si no hay superposición o el usuario decide continuar;
    False si el usuario prefiere volver a seleccionar el recorte.
    """
    scale_cfg   = session.scale     if session is not None else config.ScaleConfig()
    overlay_cfg = session.overlay   if session is not None else config.OverlayConfig()
    device_name = (session.experiment.device_name if session is not None
                   else config.ExperimentConfig().device_name)

    zonas = [z for z in (
        _zona_barra_escala(alto_orig, ancho_orig, scale_cfg),
        _zona_textos_overlay(alto_orig, ancho_orig, overlay_cfg, caso_label, device_name),
    ) if z is not None]

    if not any(_rects_se_superponen(rect, z) for z in zonas):
        return True

    print_warn("El recorte seleccionado se superpone con la zona donde Parte1.py "
               "dibuja overlays (barra de escala / texto / etiqueta) en las "
               "imágenes de Adquisición.")
    print_warn("Esos píxeles de texto quedarían dentro de los datos que Parte3.py "
               "usa para el análisis cuantitativo (centroide, ancho de haz, "
               "energía encerrada).")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    continuar = messagebox.askyesno(
        "El recorte incluye la zona de overlays",
        "El rectángulo que seleccionaste incluye la esquina donde se dibujan "
        "la barra de escala y/o el texto/etiqueta sobre las imágenes de "
        "Adquisición.\n\n"
        "Esos píxeles quedarían dentro de los datos que se usan para el "
        "análisis cuantitativo (centroide, ancho de haz, energía encerrada) "
        "en la Opción 3.\n\n"
        "¿Deseas continuar con este recorte de todas formas?\n\n"
        "(Elige 'No' para volver a seleccionar el recorte.)",
        parent=root)
    root.destroy()
    return continuar


# =============================================================================
# GUI DE SELECCIÓN DE RECORTE
# =============================================================================

class SelectorRecorte:
    """
    Abre una ventana con la imagen de referencia para que el usuario
    dibuje un rectángulo de recorte con el ratón.
    Devuelve (x1, y1, x2, y2) en coordenadas de la imagen original.

    Es la ÚNICA interfaz de selección de ROI del proyecto — tanto el
    preprocesado del haz óptico (Opción 2) como el de la cámara de
    turbulencia (Opción 5) la reutilizan, para no mantener dos interfaces
    distintas para la misma tarea.
    """

    def __init__(self, imagen: "Path | np.ndarray", rect_inicial: tuple | None = None):
        """
        `imagen`: una ruta (Path o str) a un archivo de imagen en disco, o
        directamente un array de NumPy ya cargado en memoria (grises o
        BGR) — por ejemplo el primer frame de un video leído con
        cv2.VideoCapture, sin necesidad de escribirlo a disco primero.

        `rect_inicial`: si se proporciona (x1,y1,x2,y2 en coordenadas de la
        imagen ORIGINAL), se dibuja como vista previa al abrir la ventana
        — útil para mostrar el ROI reutilizado de un caso anterior sobre
        la imagen de referencia del caso actual, antes de que el usuario
        decida confirmarlo tal cual o dibujar uno nuevo que lo reemplace.
        """
        self.rect = rect_inicial
        self._rect_inicial = rect_inicial

        if isinstance(imagen, np.ndarray):
            self.ruta_imagen = None
            img_bgr = imagen
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif img_bgr.dtype != np.uint8:
                img_bgr = cv2.normalize(img_bgr, None, 0, 255,
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR) if img_bgr.ndim == 2 else img_bgr
        else:
            self.ruta_imagen = imagen
            img_bgr = utils_imagenes.leer_imagen(imagen)
            if img_bgr is None:
                raise ValueError(f"No se pudo leer la imagen: {imagen}")

        self.img_original = img_bgr
        self.alto_orig, self.ancho_orig = img_bgr.shape[:2]

        max_w, max_h = 1200, 800
        escala_w = max_w / self.ancho_orig
        escala_h = max_h / self.alto_orig
        self.escala = min(escala_w, escala_h, 1.0)

        self.ancho_disp = int(self.ancho_orig * self.escala)
        self.alto_disp  = int(self.alto_orig  * self.escala)

        from PIL import Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (self.ancho_disp, self.alto_disp))
        self._pil_img = Image.fromarray(img_small)

        self._start_x = self._start_y = 0
        self._rect_id = None
        self._drawing = False

    def seleccionar(self) -> tuple[int, int, int, int] | None:
        """
        Abre la ventana Tkinter y bloquea hasta que el usuario confirma o
        cancela la selección del ROI.

        Detalle importante de coordenadas: la imagen se muestra ESCALADA
        para caber en 1200×800 px, pero el rectángulo devuelto se convierte
        de vuelta a coordenadas de la imagen ORIGINAL dividiendo entre
        `self.escala`. Todo el recorte posterior trabaja en resolución
        completa; el reescalado es solo para que la ventana quepa en
        pantalla. Confundir ambos sistemas produciría recortes
        desplazados o de tamaño incorrecto.

        Retorna (x1, y1, x2, y2) en píxeles de la imagen original, o None
        si el usuario cerró la ventana o canceló — en cuyo caso el
        llamador aborta el preprocesado de ese caso sin escribir nada.
        """
        from PIL import ImageTk
        root = tk.Tk()
        root.title("Selecciona la región de recorte — dibuja un rectángulo")

        self._tk_img = ImageTk.PhotoImage(self._pil_img)

        info_frame = tk.Frame(root, bg="#1e1e2e", pady=6)
        info_frame.pack(fill=tk.X)

        texto_instrucciones = ("Haz clic y arrastra para definir el recorte. Presiona CONFIRMAR cuando estés listo."
                               if self._rect_inicial is None else
                               "Recorte REUTILIZADO del caso anterior (mostrado abajo). Presiona CONFIRMAR "
                               "para usarlo tal cual, o haz clic y arrastra para dibujar uno nuevo.")
        tk.Label(
            info_frame,
            text=texto_instrucciones,
            bg="#1e1e2e", fg="#cdd6f4", font=("Courier", 11)
        ).pack(side=tk.LEFT, padx=12)

        if self._rect_inicial is not None:
            _ix1, _iy1, _ix2, _iy2 = self._rect_inicial
            texto_medidas_inicial = (f"Ancho: {_ix2-_ix1} px  |  Alto: {_iy2-_iy1} px  |  "
                                     f"Origen: ({_ix1}, {_iy1})  [reutilizado]")
        else:
            texto_medidas_inicial = "Ancho: — px  |  Alto: — px  |  Origen: (—, —)"
        self._lbl_medidas = tk.Label(
            info_frame,
            text=texto_medidas_inicial,
            bg="#1e1e2e", fg="#a6e3a1", font=("Courier", 11, "bold")
        )
        self._lbl_medidas.pack(side=tk.RIGHT, padx=12)

        canvas = tk.Canvas(root, width=self.ancho_disp, height=self.alto_disp,
                           cursor="crosshair", bg="#000")
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        if self._rect_inicial is not None:
            _dx1, _dy1 = _ix1 * self.escala, _iy1 * self.escala
            _dx2, _dy2 = _ix2 * self.escala, _iy2 * self.escala
            self._rect_id = canvas.create_rectangle(
                _dx1, _dy1, _dx2, _dy2, outline="#f9e2af", width=2, dash=(4, 2))

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)

        def on_confirmar():
            if self._rect_id is None:
                messagebox.showwarning("Sin selección",
                                       "Por favor dibuja un rectángulo antes de confirmar.")
                return
            root.destroy()

        def on_cancelar():
            self.rect = None
            root.destroy()

        crear_botones_confirmar_cancelar(
            btn_frame, "Confirmar recorte", on_confirmar, on_cancelar)

        # ── Manejadores de ratón ──────────────────────────────────────────
        # press → ancla la esquina inicial; drag → redibuja el rectángulo y
        # muestra sus dimensiones YA convertidas a píxeles de la imagen
        # original (dividiendo entre self.escala), para que el usuario vea
        # el tamaño real del recorte y no el de la vista reducida;
        # release → fija self.rect, también en coordenadas originales.

        def mouse_press(event):
            self._start_x, self._start_y = event.x, event.y
            self._drawing = True
            if self._rect_id:
                canvas.delete(self._rect_id)

        def mouse_drag(event):
            if not self._drawing:
                return
            if self._rect_id:
                canvas.delete(self._rect_id)
            self._rect_id = canvas.create_rectangle(
                self._start_x, self._start_y, event.x, event.y,
                outline="#89dceb", width=2, dash=(4, 2)
            )
            w_px = int(abs(event.x - self._start_x) / self.escala)
            h_px = int(abs(event.y - self._start_y) / self.escala)
            ox   = int(min(self._start_x, event.x) / self.escala)
            oy   = int(min(self._start_y, event.y) / self.escala)
            self._lbl_medidas.config(
                text=f"Ancho: {w_px} px  |  Alto: {h_px} px  |  Origen: ({ox}, {oy})"
            )
            self._cur_x, self._cur_y = event.x, event.y

        def mouse_release(event):
            self._drawing = False
            x1 = int(min(self._start_x, event.x) / self.escala)
            y1 = int(min(self._start_y, event.y) / self.escala)
            x2 = int(max(self._start_x, event.x) / self.escala)
            y2 = int(max(self._start_y, event.y) / self.escala)
            self.rect = (x1, y1, x2, y2)

        canvas.bind("<ButtonPress-1>",   mouse_press)
        canvas.bind("<B1-Motion>",       mouse_drag)
        canvas.bind("<ButtonRelease-1>", mouse_release)

        root.mainloop()
        return self.rect


# =============================================================================
# PROCESAMIENTO
# =============================================================================

def recortar_imagen(ruta_src: Path, ruta_dst: Path,
                    x1: int, y1: int, x2: int, y2: int,
                    dark_frame: np.ndarray | None = None):
    """
    Lee ruta_src, recorta y guarda en ruta_dst.
    ruta_src NUNCA es sobreescrita; siempre se escribe en ruta_dst.
    """
    # Seguridad: nunca sobreescribir el original
    if ruta_src.resolve() == ruta_dst.resolve():
        print(f"  ⛔ Origen y destino son el mismo archivo, se omite: {ruta_src.name}")
        return

    img = utils_imagenes.leer_imagen(ruta_src)
    if img is None:
        print_warn(f"No se pudo leer: {ruta_src.name}")
        return

    recorte = img[y1:y2, x1:x2].copy()   # .copy() garantiza array independiente

    if dark_frame is not None:
        if recorte.shape == dark_frame.shape:
            recorte = np.clip(
                recorte.astype(np.float32) - dark_frame, 0, 255
            ).astype(np.uint8)
        else:
            print_warn(f"Dimensiones incompatibles con dark frame en {ruta_src.name} "
                       f"({recorte.shape} vs {dark_frame.shape}). "
                       "Se guardará sin restar dark frame.")

    ruta_dst.parent.mkdir(parents=True, exist_ok=True)
    utils_imagenes.guardar_imagen(ruta_dst, recorte)
    print_ok(f"{ruta_src.name}")



def recortar_video(ruta_src: Path, ruta_dst: Path,
                   x1: int, y1: int, x2: int, y2: int,
                   dark_frame: np.ndarray | None = None):
    """
    Lee ruta_src (.mp4) frame a frame, recorta y guarda en ruta_dst.
    ruta_src NUNCA es modificada.

    Por decision de diseno, el preprocesado trabaja EXCLUSIVAMENTE con el
    .mp4 de la adquisicion (no con el .npz de datos crudos), para acelerar
    el computo. Los datos crudos siguen guardandose en la adquisicion por
    si se necesitan en el futuro, pero no se usan aqui.
    """
    if ruta_src.resolve() == ruta_dst.resolve():
        print(f"  ⛔ Origen y destino son el mismo archivo, se omite: {ruta_src.name}")
        return

    ancho = x2 - x1
    alto  = y2 - y1
    ruta_dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(ruta_src))
    if not cap.isOpened():
        print_warn(f"No se pudo abrir el video: {ruta_src.name}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(ruta_dst), fourcc, fps, (ancho, alto))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        recorte = frame[y1:y2, x1:x2].copy()
        if dark_frame is not None and recorte.shape == dark_frame.shape:
            recorte = np.clip(
                recorte.astype(np.float32) - dark_frame, 0, 255
            ).astype(np.uint8)
        out.write(recorte)
        frame_n += 1
        if frame_n % 50 == 0:
            pct = frame_n / total * 100 if total > 0 else 0
            print(f"    … frame {frame_n}/{total}  ({pct:.0f}%)")
    cap.release()
    out.release()
    print_ok(f"Video recortado: {ruta_dst.name}  ({frame_n} frames)")



def procesar_subcarpeta(subcarpeta_src: Path, subcarpeta_dst: Path,
                        x1: int, y1: int, x2: int, y2: int,
                        dark_frame: np.ndarray | None = None,
                        incluir_video: bool = False):
    """
    Itera sobre los archivos de subcarpeta_src y guarda los recortes
    en subcarpeta_dst. Los originales en subcarpeta_src no se tocan.
    """
    print(f"\n📂 {subcarpeta_src.name}")
    for f in sorted(subcarpeta_src.iterdir()):
        dst = subcarpeta_dst / f.name          # destino siempre en carpeta_sal
        if f.suffix.lower() in EXTENSIONES_IMAGEN:
            recortar_imagen(f, dst, x1, y1, x2, y2, dark_frame)
        elif incluir_video and f.suffix.lower() in EXTENSIONES_VIDEO:
            recortar_video(f, dst, x1, y1, x2, y2, dark_frame)


# =============================================================================
# MAIN
# =============================================================================

def _procesar_caso(carpeta_caso: Path, rect: tuple | None = None, session=None) -> tuple | None:
    """
    Preprocesa UN caso (Sin turbulencia / Transitorio / Con turbulencia):
    busca sus subcarpetas dentro de Adquisicion/, selecciona (o reutiliza)
    el rectángulo de recorte, resta su propio dark frame (nunca el de
    otro caso) y escribe en <carpeta_caso>/Preprocesado.

    Si `rect` ya viene dado (de un caso anterior en la misma corrida), se
    reutiliza sin abrir la interfaz de selección — el encuadre de cámara
    es el mismo entre casos de turbulencia del mismo dispositivo. Si se
    selecciona un rect nuevo, se valida que no se superponga con la zona
    de overlays antes de aceptarlo (ver _advertir_si_roi_toca_overlays).

    Retorna el rectángulo (x1, y1, x2, y2) usado (para reutilizarlo en el
    siguiente caso), o None si este caso se omitió (sin Adquisicion,
    sin estabilidad_temporal, o el usuario canceló la selección de ROI).
    """
    carpeta_adq = carpeta_caso / utils_carpetas.NOMBRE_ADQUISICION
    if not carpeta_adq.exists():
        print_warn(f"'{utils_carpetas.NOMBRE_ADQUISICION}' no encontrada en "
                   f"{carpeta_caso} — se omite este caso.")
        return None

    # ── 1. Subcarpetas: solo estabilidad es obligatoria ───────────────────
    try:
        sub_estabilidad = buscar_subcarpeta_obligatoria(carpeta_adq, SUBCARPETA_ESTABILIDAD)
    except FileNotFoundError as e:
        print_warn(f"Caso omitido — {e}")
        return None

    sub_haz       = buscar_subcarpeta_opcional(carpeta_adq, SUBCARPETA_HAZ)
    sub_haz_fibra = buscar_subcarpeta_opcional(carpeta_adq, SUBCARPETA_HAZ_FIBRA)

    if sub_haz is None:
        print_info(f"Subcarpeta '{SUBCARPETA_HAZ}' no encontrada — se omitirá.")
    if sub_haz_fibra is None:
        print_info(f"Subcarpeta '{SUBCARPETA_HAZ_FIBRA}' no encontrada — se omitirá.")

    # ── 2. Imagen de referencia ───────────────────────────────────────────
    img_ref = buscar_imagen_referencia(sub_estabilidad)
    if img_ref is None:
        # Etapa 1 pudo haberse capturado solo en video (sin fotos fijas):
        # usar el primer frame del video como referencia para la ROI.
        video_ref = buscar_video(sub_estabilidad)
        if video_ref is None:
            print_warn(f"No se encontró ninguna imagen ni video en "
                       f"'{sub_estabilidad}' — se omite este caso.")
            return None
        img_ref = extraer_primer_frame_como_imagen(video_ref)
        if img_ref is None:
            print_warn(f"No se encontró ninguna imagen en '{sub_estabilidad}' y no "
                       f"se pudo leer el primer frame de '{video_ref.name}' — "
                       f"se omite este caso.")
            return None
        print()
        print_info(f"No hay imagen fija en '{sub_estabilidad}'; se usará el "
                   f"primer frame de '{video_ref.name}' como referencia para la ROI.")
    print(f"\n📷 Imagen de referencia: {img_ref.name}")

    # ── 3. Selección de recorte. Se pide interactivamente en CADA caso --
    # si `rect` viene de un caso anterior en esta misma corrida, se
    # muestra como VISTA PREVIA sobre la imagen de referencia de este
    # caso (el encuadre pudo haber cambiado entre adquisiciones), y el
    # usuario decide si confirmarlo tal cual o dibujar uno nuevo. Nunca se
    # reutiliza en automático sin que el usuario lo vea primero. ──
    try:
        from PIL import Image, ImageTk  # noqa: F401
    except ImportError:
        print()
        print_warn("Pillow no está instalado. Instalando…")
        _resultado_pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "Pillow", "--break-system-packages", "-q"],
            capture_output=True, text=True)
        if _resultado_pip.returncode != 0:
            print_error("No se pudo instalar Pillow automáticamente:")
            print(f"   {_resultado_pip.stderr.strip()}")
            print_error("Instala Pillow manualmente ('pip install Pillow') y vuelve a ejecutar.")
            return None
        try:
            from PIL import Image, ImageTk  # noqa: F401
            print_ok("Pillow instalado correctamente.")
        except ImportError as _e:
            print_error(f"Pillow se instaló pero no se pudo importar: {_e}")
            print_error("Intenta reiniciar el script; si el error persiste, instala Pillow manualmente.")
            return None

    caso_label  = utils_carpetas.CASO_LABELS.get(carpeta_caso.name, carpeta_caso.name)
    rect_previo = rect  # None en el primer caso, o el rect confirmado en el caso anterior
    if rect_previo is None:
        print("\nAbriendo interfaz de selección de recorte…")
    else:
        print("\nMostrando vista previa del recorte reutilizado sobre la imagen de "
              "referencia de este caso — confirma o dibuja uno nuevo…")
    while True:
        selector = SelectorRecorte(img_ref, rect_inicial=rect_previo)
        rect = selector.seleccionar()
        if rect is None:
            print_error("Selección cancelada. El programa terminará sin procesar nada.")
            return None
        if _advertir_si_roi_toca_overlays(
                rect, selector.alto_orig, selector.ancho_orig, session, caso_label):
            break
        print_info("Repitiendo la selección del recorte…")

    x1, y1, x2, y2 = rect
    ancho_rec = x2 - x1
    alto_rec  = y2 - y1

    if ancho_rec <= 0 or alto_rec <= 0:
        print_error("El rectángulo seleccionado no tiene área válida.")
        return None

    print()
    print_ok("Recorte definido:")
    print(f"   Origen : ({x1}, {y1}) px")
    print(f"   Tamaño : {ancho_rec} × {alto_rec} px")

    # ── 4. Dark frame propio de este caso ─────────────────────────────────
    dark_frame = cargar_dark_frame(carpeta_adq, x1, y1, x2, y2)

    # ── 5. Crear carpeta de salida (siempre <carpeta_caso>/Preprocesado) ──
    carpeta_sal = Path(utils_carpetas.carpeta_salida_segura(
        str(carpeta_caso / utils_carpetas.NOMBRE_PREPROCESADO)))
    carpeta_sal.mkdir(parents=True, exist_ok=True)
    print(f"\nCarpeta de salida: {carpeta_sal}")

    # ── 6. Procesar subcarpetas ───────────────────────────────────────────
    if sub_haz is not None:
        procesar_subcarpeta(
            sub_haz,
            carpeta_sal / SUBCARPETA_HAZ,
            x1, y1, x2, y2,
            dark_frame=dark_frame
        )
    if sub_haz_fibra is not None:
        procesar_subcarpeta(
            sub_haz_fibra,
            carpeta_sal / SUBCARPETA_HAZ_FIBRA,
            x1, y1, x2, y2,
            dark_frame=dark_frame
        )

    procesar_subcarpeta(
        sub_estabilidad,
        carpeta_sal / SUBCARPETA_ESTABILIDAD,
        x1, y1, x2, y2,
        dark_frame=dark_frame,
        incluir_video=True
    )

    # ── 7. Copiar reporte y metadata (para que Parte3 pueda leer la ──────
    #      calibracion sin importar si el usuario elige Adquisicion o
    #      Preprocesado en la Opcion 3) ─────────────────────────────────
    reporte = buscar_reporte(carpeta_adq)
    if reporte:
        dst_reporte = carpeta_sal / reporte.name
        shutil.copy2(str(reporte), str(dst_reporte))
        print(f"\n📄 Reporte copiado: {reporte.name}")
    else:
        print()
        print_warn(f"No se encontró '{NOMBRE_REPORTE}' en {carpeta_adq}")

    metadata_json = carpeta_adq / "metadata_adquisicion.json"
    if metadata_json.exists():
        shutil.copy2(str(metadata_json), str(carpeta_sal / metadata_json.name))
        print(f"📄 Metadata copiada: {metadata_json.name}")

    # Registro de trazabilidad: si el dark frame realmente se restó o no
    # (antes solo quedaba un print_warn en consola, sin rastro persistente
    # si faltaba el dark frame de este caso).
    _meta_preprocesado = {
        "dark_frame_aplicado": dark_frame is not None,
        "roi_px": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }
    with open(carpeta_sal / "metadata_preprocesado.json", "w", encoding="utf-8") as _fh:
        json.dump(_meta_preprocesado, _fh, indent=2, ensure_ascii=False)
    if dark_frame is None:
        print_warn("metadata_preprocesado.json registra que este caso se procesó "
                   "SIN resta de dark frame.")

    print("\n✅ Procesamiento completado")
    print(f"   Archivos guardados en: {carpeta_sal}")
    print("   Los archivos originales NO fueron modificados.")

    return (x1, y1, x2, y2)


def main(session=None):
    """
    Punto de entrada del preprocesado.

    `session` (SessionConfig): si se pasa desde el orquestador, la carpeta
    raíz del DISPOSITIVO puede pre-seleccionarse (session.acquisition_folder).
    Siempre se pregunta explícitamente con cuáles de los casos EXISTENTES
    (Sin turbulencia/Transitorio/Con turbulencia — no todos existen
    necesariamente) se desea trabajar en esta corrida.

    El rectángulo de recorte se selecciona UNA sola vez (sobre el primer
    caso elegido) y se reutiliza para los demás — el encuadre de cámara
    es el mismo entre casos de turbulencia del mismo dispositivo.

    Retorna la ruta de la carpeta RAÍZ DEL DISPOSITIVO (str), o None si
    se cancela. Los recortes de cada caso se escriben en
    <raiz>/<caso>/Preprocesado.
    """
    # ── 1. Selección interactiva de la carpeta raíz del dispositivo ──────
    carpeta_raiz = seleccionar_carpeta_raiz(session=session)
    if carpeta_raiz is None:
        print_error("El programa terminará porque no se seleccionó la carpeta raíz.")
        return None

    # ── 2. Casos existentes → selección múltiple (siempre se pregunta) ───
    casos_disp = utils_carpetas.casos_existentes(str(carpeta_raiz))
    if not casos_disp:
        print_error(
            "No se encontró ningún caso (Sin turbulencia/Transitorio/Con "
            f"turbulencia) dentro de la carpeta raíz: {carpeta_raiz}")
        return None

    caso_sesion = getattr(session, 'caso_actual', None) if session is not None else None
    preseleccionados = [caso_sesion] if caso_sesion in casos_disp else None

    casos_elegidos = pedir_casos_multiples(casos_disp, preseleccionados=preseleccionados)
    if not casos_elegidos:
        print_error("No se seleccionó ningún caso. El programa terminará.")
        return None

    print(f"\nCarpeta raíz del dispositivo: {carpeta_raiz}")
    print("Casos elegidos: " + ", ".join(
        utils_carpetas.CASO_LABELS.get(c, c) for c in casos_elegidos))

    # ── 3. Procesar cada caso elegido, reutilizando el mismo rect ────────
    rect = None
    algun_caso_ok = False
    for caso in casos_elegidos:
        carpeta_caso = carpeta_raiz / caso
        print_banner(f"CASO: {utils_carpetas.CASO_LABELS.get(caso, caso)}")
        rect_resultante = _procesar_caso(carpeta_caso, rect=rect, session=session)
        if rect_resultante is not None:
            rect = rect_resultante
            algun_caso_ok = True

    if not algun_caso_ok:
        print_error("Ningún caso pudo procesarse.")
        return None

    return str(carpeta_raiz)   # ← retornar la carpeta RAIZ DEL DISPOSITIVO para el orquestador


if __name__ == "__main__":
    main()