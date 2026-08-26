# -*- coding: utf-8 -*-
"""
session_store.py — Persistencia de configuración entre sesiones.

Permite que las ventanas de configuración (Opciones 1, 3 y 4) ofrezcan
"Cargar última sesión" (sobrevive a cerrar y volver a abrir el programa) y
"Cargar preset" (configuraciones predefinidas, versionadas junto al código).

Estas funciones SOLO rellenan campos — nunca inician una adquisición ni un
análisis por sí solas; el usuario siempre debe revisar y presionar
"Confirmar y continuar".

Almacenamiento:
    .session_cache/ultima_<nombre>.json   — última sesión confirmada
    presets/<nombre>/<preset>.json        — presets predefinidos (versionados)
"""

import os
import json
import dataclasses

from console_ui import print_warn

_ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR   = os.path.join(_ROOT_DIR, ".session_cache")
_PRESETS_DIR = os.path.join(_ROOT_DIR, "presets")


def guardar_ultima_sesion(nombre: str, datos: dict) -> None:
    """Guarda `datos` (un dict serializable, típicamente
    dataclasses.asdict(session)) como la última sesión confirmada para
    el diálogo `nombre` ('setup', 'analysis' o 'turbulence')."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        ruta = os.path.join(_CACHE_DIR, f"ultima_{nombre}.json")
        with open(ruta, "w", encoding="utf-8") as fh:
            json.dump(datos, fh, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print_warn(f"No se pudo guardar la última sesión ({nombre}): {e}")


def cargar_ultima_sesion(nombre: str) -> dict | None:
    """Retorna el dict de la última sesión confirmada para `nombre`, o
    None si nunca se guardó una (incluye la primera vez que se usa el
    programa, y el caso de haberlo cerrado y vuelto a abrir)."""
    ruta = os.path.join(_CACHE_DIR, f"ultima_{nombre}.json")
    if not os.path.exists(ruta):
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print_warn(f"No se pudo leer la última sesión ({nombre}): {e}")
        return None


def listar_presets(nombre: str) -> list[str]:
    """Lista los nombres de preset disponibles (sin extensión) para
    `nombre`. El usuario puede agregar más archivos .json a
    presets/<nombre>/ sin tocar el código."""
    carpeta = os.path.join(_PRESETS_DIR, nombre)
    if not os.path.isdir(carpeta):
        return []
    return sorted(f[:-5] for f in os.listdir(carpeta) if f.endswith(".json"))


def cargar_preset(nombre: str, preset: str) -> dict | None:
    """Carga el dict de `presets/<nombre>/<preset>.json`, o None si no existe."""
    ruta = os.path.join(_PRESETS_DIR, nombre, f"{preset}.json")
    if not os.path.exists(ruta):
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print_warn(f"No se pudo leer el preset '{preset}' ({nombre}): {e}")
        return None



# Campos de SessionConfig que "cargar última sesión"/"cargar preset" NUNCA
# deben sobrescribir. um_per_px/scale.um_per_px/scale.px_per_um son la
# calibración de píxel de LA CORRIDA ACTUAL (ya fijada por
# step_pixel_calibration() antes de abrir estos diálogos) — sobrescribirla
# con un valor guardado de otra corrida/dispositivo corrompería en silencio
# todas las magnitudes en µm calculadas después, sin ningún error visible.
# acquisition_folder/cropped_folder/caso_actual se excluyen por la misma
# prudencia: son estado de ESTA corrida, nunca deberían venir de un preset
# ni de una sesión anterior.
_CAMPOS_EXCLUIDOS_NIVEL_SUPERIOR = frozenset({
    "um_per_px", "acquisition_folder", "cropped_folder", "caso_actual",
})
_CAMPOS_EXCLUIDOS_SCALE = frozenset({"um_per_px", "px_per_um"})


def _aplicar_dict_a_dataclass(obj, datos, excluir: frozenset = frozenset()) -> None:
    """
    Copia recursivamente las claves de `datos` sobre los campos de `obj`
    (una instancia de dataclass), ignorando claves desconocidas, los
    nombres de campo listados en `excluir`, y preservando cualquier campo
    que `datos` no mencione.
    """
    if not isinstance(datos, dict) or not dataclasses.is_dataclass(obj):
        return
    for campo in dataclasses.fields(obj):
        if campo.name not in datos or campo.name in excluir:
            continue
        valor_actual = getattr(obj, campo.name)
        valor_nuevo  = datos[campo.name]
        if dataclasses.is_dataclass(valor_actual) and isinstance(valor_nuevo, dict):
            sub_excluir = _CAMPOS_EXCLUIDOS_SCALE if campo.name == "scale" else frozenset()
            _aplicar_dict_a_dataclass(valor_actual, valor_nuevo, sub_excluir)
        else:
            setattr(obj, campo.name, valor_nuevo)


def aplicar_datos_a_session(session, datos: dict) -> None:
    """
    Punto de entrada público: aplica un dict guardado (última sesión o
    preset) sobre una instancia de SessionConfig, mutándola in-place.
    Solo rellena campos — no valida ni confirma nada. Nunca sobrescribe
    la calibración de píxel de la corrida actual ni las rutas de carpeta
    en curso (ver _CAMPOS_EXCLUIDOS_NIVEL_SUPERIOR).
    """
    _aplicar_dict_a_dataclass(session, datos, _CAMPOS_EXCLUIDOS_NIVEL_SUPERIOR)
