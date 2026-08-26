# -*- coding: utf-8 -*-
"""
console_ui.py — Formato unificado de mensajes de consola.

Antes de este módulo, cada archivo del proyecto imprimía encabezados y
mensajes de estado con su propio estilo (separadores "=", "-", "─", "═"
mezclados incluso dentro de un mismo archivo; símbolos ✓/✗/⚠ aplicados de
forma inconsistente). Este módulo centraliza esas convenciones para que
la consola se vea como un único programa desarrollado de forma consistente.

Uso típico:
    from console_ui import print_banner, print_seccion, print_ok, print_error

    print_banner("OPCION 1 — ADQUISICION DEL HAZ OPTICO")
    print_seccion("Configuración del experimento")
    print_ok("Adquisición completada.")
    print_error("No se seleccionó carpeta de origen.")
"""

ANCHO_BANNER = 68


def print_banner(texto: str, ancho: int = ANCHO_BANNER) -> None:
    """Encabezado principal (inicio de una opción del menú o de una fase)."""
    linea = "═" * ancho
    print("\n" + linea)
    print(f"  {texto}")
    print(linea)


def print_seccion(texto: str, ancho: int = ANCHO_BANNER) -> None:
    """Sub-encabezado de una sola línea (dentro de una opción ya iniciada)."""
    relleno = "─" * max(3, ancho - len(texto) - 5)
    print(f"\n── {texto} {relleno}")


def print_ok(texto: str) -> None:
    """Mensaje de éxito/confirmación."""
    print(f"  ✓ {texto}")


def print_error(texto: str) -> None:
    """Mensaje de error (operación no completada)."""
    print(f"  ✗ {texto}")


def print_warn(texto: str) -> None:
    """Advertencia (la operación continúa, pero el usuario debe saberlo)."""
    print(f"  ⚠  {texto}")


def print_info(texto: str) -> None:
    """Información neutra (no es éxito, error ni advertencia)."""
    print(f"  ℹ  {texto}")


def print_skip(texto: str) -> None:
    """Paso omitido intencionalmente."""
    print(f"  ⏭  {texto}")
