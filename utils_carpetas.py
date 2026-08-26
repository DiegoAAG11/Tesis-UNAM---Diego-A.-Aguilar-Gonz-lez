# -*- coding: utf-8 -*-
"""
utils_carpetas.py — Utilidades compartidas de organización de carpetas.

Módulo sin dependencias internas del proyecto (solo `os`), para que
Parte1.py, Parte2.py, Parte3.py y CamaraTurbulencia.py puedan reutilizarlo
sin acoplarse entre sí. Centraliza la única regla de negocio que se repetía
con al menos cuatro convenciones ligeramente distintas en el proyecto:
NUNCA sobreescribir una carpeta de resultados existente.

Convención de carpetas de un experimento (haz óptico o cámara de
turbulencia), fijada a partir de esta refactorización:

    <raíz: YYYYMMDD_Caracterizacion...>/
        Adquisicion/       ← datos crudos
        Preprocesado/      ← ROI + resta de dark frame
        Analisis/          ← resultados (gráficas, videos, tablas)

Ninguna de las tres subcarpetas lleva prefijo de fecha (la fecha va
únicamente en el nombre de la carpeta raíz).
"""

import os
import glob
import time

NOMBRE_ADQUISICION = "Adquisicion"
NOMBRE_PREPROCESADO = "Preprocesado"
NOMBRE_ANALISIS = "Analisis"
NOMBRE_DARKFRAME = "darkframe"
# Subcarpeta de Analisis/ donde se exportan los datos numericos (CSV/TXT)
# por separado de las graficas/videos — mismo nombre que usa Parte3.py
# internamente (no se toca Parte3.py, solo se comparte el nombre).
NOMBRE_SUBCARPETA_DATOS_CRUDOS = "Datos_Crudos"

# Casos de turbulencia del pipeline del haz óptico (Opciones 1-3): cada
# dispositivo tiene UNA carpeta raíz que contiene estos 3 casos como
# subcarpetas, cada uno con su propio Adquisicion/Preprocesado/Analisis
# (darkframe/ anidada dentro de Adquisicion) — ver crear_carpetas_caso().
CASOS = ["SinTurbulencia", "Transitorio", "ConTurbulencia"]
CASO_LABELS = {
    "SinTurbulencia": "Sin turbulencia",
    "Transitorio":    "Transitorio",
    "ConTurbulencia": "Con turbulencia",
}


def carpeta_raiz_segura(base_dir: str, nombre_base: str) -> str:
    """
    Calcula una ruta de carpeta raíz que no exista todavía dentro de
    `base_dir`, para NUNCA sobreescribir una caracterización anterior.

    `nombre_base` ya debe incluir el prefijo de fecha
    (ej. "20260720_CaracterizacionHaz_MMI"). Si ese nombre ya existe,
    se agrega un sufijo numérico de dos dígitos: "_01", "_02", ...

    Retorna la ruta completa (no crea la carpeta).
    """
    ruta = os.path.join(base_dir, nombre_base)
    if not os.path.exists(ruta):
        return ruta
    cnt = 1
    nueva = os.path.join(base_dir, f"{nombre_base}_{cnt:02d}")
    while os.path.exists(nueva):
        cnt += 1
        nueva = os.path.join(base_dir, f"{nombre_base}_{cnt:02d}")
    return nueva


def carpeta_salida_segura(ruta_deseada: str) -> str:
    """
    Retorna una ruta de carpeta segura para escribir resultados.

    Si `ruta_deseada` no existe o está vacía, la retorna tal cual (reuso
    normal — por ejemplo, la carpeta `Preprocesado`/`Analisis` vacía que
    ya se creó junto con la carpeta raíz durante la adquisición).

    Si ya contiene archivos (de una corrida anterior), retorna una
    versión numerada (_02, _03, ...) para NUNCA sobreescribir resultados
    previos.
    """
    if not os.path.exists(ruta_deseada):
        return ruta_deseada
    try:
        tiene_contenido = any(True for _ in os.scandir(ruta_deseada))
    except Exception:
        tiene_contenido = True
    if not tiene_contenido:
        return ruta_deseada
    base = ruta_deseada
    cnt = 2
    nueva = f"{base}_{cnt:02d}"
    while os.path.exists(nueva):
        cnt += 1
        nueva = f"{base}_{cnt:02d}"
    print(f"  AVISO: '{os.path.basename(base)}' ya contiene resultados de un "
          f"ensayo anterior.")
    print(f"  Los nuevos resultados se guardaran en: "
          f"'{os.path.basename(nueva)}' para no sobreescribir nada.")
    return nueva


def crear_triple_raiz(carpeta_raiz: str) -> tuple:
    """
    Crea la estructura estándar de un experimento dentro de `carpeta_raiz`:
    Adquisicion/ (debe ser nueva — exist_ok=False, ya que carpeta_raiz
    recién fue desambiguada por carpeta_raiz_segura), Preprocesado/ y
    Analisis/ (exist_ok=True, se crean vacías junto con Adquisicion).

    Retorna (carpeta_adquisicion, carpeta_preprocesado, carpeta_analisis).
    """
    carpeta_adq = os.path.join(carpeta_raiz, NOMBRE_ADQUISICION)
    os.makedirs(carpeta_adq, exist_ok=False)
    carpeta_pre = os.path.join(carpeta_raiz, NOMBRE_PREPROCESADO)
    os.makedirs(carpeta_pre, exist_ok=True)
    carpeta_ana = os.path.join(carpeta_raiz, NOMBRE_ANALISIS)
    os.makedirs(carpeta_ana, exist_ok=True)
    return carpeta_adq, carpeta_pre, carpeta_ana


def normalizar_carpeta_raiz(carpeta_seleccionada: str) -> str:
    """
    Dado lo que el usuario seleccionó en un diálogo (que puede ser la
    carpeta raíz de la caracterización, o por error/costumbre la propia
    subcarpeta 'Adquisicion'), retorna la carpeta RAÍZ real.

    Si la selección es exactamente 'Adquisicion', 'Preprocesado' o
    'Analisis', se asume que el usuario navegó un nivel de más y se
    retorna su carpeta padre. En cualquier otro caso, se retorna la
    selección tal cual (se asume que ya es la raíz).
    """
    nombre = os.path.basename(os.path.normpath(carpeta_seleccionada))
    if nombre in (NOMBRE_ADQUISICION, NOMBRE_PREPROCESADO, NOMBRE_ANALISIS):
        return os.path.dirname(os.path.abspath(carpeta_seleccionada))
    return carpeta_seleccionada


def crear_carpetas_caso(carpeta_caso: str) -> tuple:
    """
    Crea la estructura de un CASO (Sin turbulencia / Transitorio / Con
    turbulencia) dentro de la carpeta raíz de un dispositivo:
    Adquisicion/ (debe ser nueva — exist_ok=False, ya que `carpeta_caso`
    recién fue desambiguada por carpeta_raiz_segura), con darkframe/
    ANIDADA dentro de Adquisicion (no como hermana), y Preprocesado/ y
    Analisis/ (exist_ok=True, se crean vacías junto con las anteriores).
    Cada caso adquiere y usa su propio dark frame — nunca se comparte
    entre casos.

    No es necesario crear `carpeta_caso` por separado: os.makedirs con
    exist_ok=False sobre la subcarpeta Adquisicion ya crea los niveles
    intermedios que falten.

    Retorna (carpeta_adquisicion, carpeta_preprocesado, carpeta_analisis,
    carpeta_darkframe).
    """
    carpeta_adq = os.path.join(carpeta_caso, NOMBRE_ADQUISICION)
    os.makedirs(carpeta_adq, exist_ok=False)
    carpeta_dark = os.path.join(carpeta_adq, NOMBRE_DARKFRAME)
    os.makedirs(carpeta_dark, exist_ok=False)
    carpeta_pre = os.path.join(carpeta_caso, NOMBRE_PREPROCESADO)
    os.makedirs(carpeta_pre, exist_ok=True)
    carpeta_ana = os.path.join(carpeta_caso, NOMBRE_ANALISIS)
    os.makedirs(carpeta_ana, exist_ok=True)
    return carpeta_adq, carpeta_pre, carpeta_ana, carpeta_dark


def casos_existentes(carpeta_raiz_dispositivo: str) -> list:
    """
    Retorna el subconjunto de CASOS cuya subcarpeta ya existe dentro de
    `carpeta_raiz_dispositivo`, en el mismo orden de CASOS. Usado por el
    diálogo de selección múltiple de Parte2.py/Parte3.py — "no todos los
    casos existen necesariamente".
    """
    return [c for c in CASOS
            if os.path.isdir(os.path.join(carpeta_raiz_dispositivo, c))]


def resolver_carpeta_mas_reciente(carpeta_base: str, prefijo: str) -> str | None:
    """
    Ubica la carpeta de resultados más reciente que empiece con `prefijo`
    dentro de `carpeta_base` (ej. "Analisis", "Analisis_02", "Analisis_03",
    generadas por carpeta_salida_segura() para nunca sobreescribir corridas
    anteriores). Retorna la de fecha de modificación más reciente, o None
    si no existe ninguna.
    """
    candidatas = [p for p in glob.glob(os.path.join(carpeta_base, f"{prefijo}*"))
                  if os.path.isdir(p)]
    if not candidatas:
        return None
    return max(candidatas, key=os.path.getmtime)


def normalizar_carpeta_raiz_dispositivo(carpeta_seleccionada: str) -> str:
    """
    Como normalizar_carpeta_raiz, pero para la jerarquía de tres niveles
    del pipeline del haz óptico: <raíz del dispositivo>/<caso>/
    {Adquisicion,Preprocesado,Analisis}, con darkframe ANIDADA dentro de
    Adquisicion. Sube los niveles que hagan falta según qué haya
    seleccionado el usuario: darkframe (3 niveles), Adquisicion/
    Preprocesado/Analisis (2 niveles), un caso directamente (1 nivel),
    o ya la raíz (0 niveles).
    """
    ruta = os.path.normpath(carpeta_seleccionada)
    nombre = os.path.basename(ruta)
    if nombre == NOMBRE_DARKFRAME:
        ruta = os.path.dirname(os.path.abspath(ruta))
        nombre = os.path.basename(ruta)
    if nombre in (NOMBRE_ADQUISICION, NOMBRE_PREPROCESADO, NOMBRE_ANALISIS):
        ruta = os.path.dirname(os.path.abspath(ruta))
        nombre = os.path.basename(ruta)
    if nombre in CASOS:
        ruta = os.path.dirname(os.path.abspath(ruta))
    return ruta


def asegurar_carpeta(carpeta: str, intentos: int = 5, espera_s: float = 0.3) -> None:
    """
    Verifica (y si hace falta, recrea) que la carpeta de salida exista
    justo antes de escribir un archivo en ella. Protege contra fallas
    transitorias del sistema de archivos -- por ejemplo, carpetas
    sincronizadas con OneDrive que a veces tardan un instante en
    reflejar una carpeta recién creada, causando un FileNotFoundError
    esporádico aunque os.makedirs() ya se haya ejecutado antes.
    """
    for _ in range(intentos):
        if os.path.isdir(carpeta):
            return
        try:
            os.makedirs(carpeta, exist_ok=True)
        except OSError:
            pass
        time.sleep(espera_s)
    os.makedirs(carpeta, exist_ok=True)  # último intento; si falla, se propaga
