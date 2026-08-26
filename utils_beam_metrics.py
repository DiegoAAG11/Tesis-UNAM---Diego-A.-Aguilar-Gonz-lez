# -*- coding: utf-8 -*-
"""
utils_beam_metrics.py — Centroide y segundo momento espacial (D4σ) del
perfil de intensidad de un haz, con umbral de fondo ISO 11146.

Antes reimplementados por separado en `CamaraTurbulencia.py`
(`_centroide`/`_segundo_momento`) y en `MedicionWFibre.py`
(`_centroide`/`_segundo_momento`) -- ambos calculan exactamente la misma
cantidad física (W_fibre / W0 medido directamente por segundo momento a
partir de una foto del haz colimado), así que una divergencia entre las
dos copias produciría el mismo valor físico reportado de forma distinta
según qué script se haya usado para medirlo.

Opera sobre arrays 2D ya convertidos a escala de grises por el llamador
(la conversión de formato de frame/color es responsabilidad de cada
módulo, que conoce el formato real de sus propios datos de cámara).
"""

import numpy as np
import cv2


def umbralizar(img: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """
    Anula los píxeles por debajo de frac*I_max (umbral de fondo ISO 11146).
    frac=0.05 (5% del pico) es el valor por defecto usado en todo el
    proyecto para descartar fondo oscuro/ruido antes de calcular momentos.
    """
    i_max = img.max() if img.size else 0.0
    if i_max <= 0:
        return img
    out = img.copy()
    out[out < frac * i_max] = 0.0
    return out


def centroide(img: np.ndarray, umbral_frac: float = 0.0) -> tuple[float, float]:
    """
    Centroide ponderado por intensidad (cv2.moments).

    umbral_frac=0 (por defecto): usa la imagen completa sin umbral --
    apropiado para beam wander / seguimiento de posición frame a frame,
    donde se quiere la posición real del haz aunque el pico sea bajo.

    umbral_frac=0.05: aplica el umbral ISO 11146 antes de calcular el
    centroide -- apropiado como paso previo a `segundo_momento` para una
    medición de tamaño (W0/W_fibre), donde el fondo debe descartarse para
    no sesgar ni el centro ni sigma.
    """
    f = img.astype(np.float64)
    img_c = umbralizar(f, frac=umbral_frac) if umbral_frac > 0 else f
    m = cv2.moments(img_c)
    if m["m00"] == 0:
        return f.shape[1] / 2.0, f.shape[0] / 2.0
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def segundo_momento(img: np.ndarray, cx: float, cy: float,
                     umbral_frac: float = 0.05) -> tuple[float, float]:
    """
    Segundo momento espacial (D4σ real). Aplica el umbral ISO 11146 antes
    de integrar: solo se consideran píxeles por encima de umbral_frac*I_max,
    porque el momento de segundo orden pondera por r² y cualquier fondo/
    ruido residual lejos del centro (incluso pequeño) infla sigma_x/sigma_y
    de forma no lineal.
    """
    f = img.astype(np.float64)
    img_u = umbralizar(f, frac=umbral_frac)
    total = img_u.sum()
    if total == 0:
        return 0.0, 0.0
    H, W = img_u.shape
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    sx = np.sqrt(((XX - cx) ** 2 * img_u).sum() / total)
    sy = np.sqrt(((YY - cy) ** 2 * img_u).sum() / total)
    return float(sx), float(sy)
