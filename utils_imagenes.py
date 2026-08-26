# -*- coding: utf-8 -*-
"""
utils_imagenes.py — Lectura/escritura de imágenes segura con rutas Unicode.

`cv2.imread`/`cv2.imwrite` llaman internamente a `fopen()`, que en
Windows solo acepta la codificación de página de códigos activa del
sistema (nunca UTF-8). Cualquier ruta con tildes, eñes u otros
caracteres no-ASCII — algo habitual en este proyecto en español, ya sea
en el nombre del dispositivo, de la carpeta elegida por el usuario,
etc. — hace que `cv2.imread` retorne `None` silenciosamente o que
`cv2.imwrite` falle sin avisar.

Este módulo reemplaza ambas funciones por la combinación equivalente
`np.fromfile`/`cv2.imdecode` (lectura) y `cv2.imencode`/`ndarray.tofile`
(escritura), que sí son Unicode-safe en Windows, Linux y macOS por
igual — sin cambiar el formato de archivo ni el contenido.
"""

import os
import numpy as np
import cv2


def leer_imagen(ruta, flags: int = cv2.IMREAD_COLOR):
    """
    Equivalente Unicode-safe de `cv2.imread(ruta, flags)`.

    Retorna `None` si el archivo no existe o no se pudo decodificar
    (mismo comportamiento de "fallo silencioso" que `cv2.imread`, para
    no romper la lógica existente de chequeo `if img is None: ...`).
    """
    ruta = str(ruta)
    if not os.path.exists(ruta):
        return None
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        return None
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, flags)


def guardar_imagen(ruta, imagen: np.ndarray) -> bool:
    """
    Equivalente Unicode-safe de `cv2.imwrite(ruta, imagen)`.

    El formato de codificación se elige por la extensión de `ruta`
    (".png", ".jpg", ...), igual que hace `cv2.imwrite`.
    Retorna True si se guardó correctamente.
    """
    ruta = str(ruta)
    extension = os.path.splitext(ruta)[1] or ".png"
    ok, buffer = cv2.imencode(extension, imagen)
    if not ok:
        return False
    buffer.tofile(ruta)
    return True
