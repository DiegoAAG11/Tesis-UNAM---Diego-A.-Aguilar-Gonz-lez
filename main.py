# -*- coding: utf-8 -*-
"""
main.py — Inicializador principal del sistema de análisis de haz de fibra óptica.

Presenta un menú gráfico con nueve opciones de flujo:

  1. Adquisición del haz óptico
  2. Recorte y preprocesado del haz óptico
  3. Caracterización del haz óptico (análisis)
  4. Adquisición: cámara de turbulencia
  5. Preprocesado: cámara de turbulencia
  6. Análisis: cámara de turbulencia
  7. Medición de W_fibre
  8. Comparación de la resiliencia de haz
  9. Comparación de mediciones de cámara de turbulencia

Uso:
    python main.py              → muestra el menú gráfico (modo normal)
    python main.py --opcion 3   → salta a la opción 3 sin menú (modo dev)

Autor: Diego Aguilar
"""

import sys
import os
import argparse
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from config import SessionConfig
import utils_carpetas
from gui.main_menu         import run_main_menu
from gui.setup_dialog      import run_setup_dialog
from gui.analysis_dialog   import run_analysis_dialog
from gui.turbulence_dialog import run_turbulence_dialog
from modules.acquisition   import run_acquisition
from modules.cropping      import run_cropping
from modules.analysis      import run_analysis
from modules.wfibre        import run_wfibre
from console_ui import print_banner, print_ok, print_error, print_warn


# =============================================================================
# ARGPARSE (modo desarrollador)
# =============================================================================

def _parse_args():
    """
    Procesa los argumentos de línea de comandos. Solo existe `--opcion N`,
    un atajo de desarrollo para saltar directo a una opción sin pasar por
    el menú gráfico; se consume UNA sola vez y después el programa vuelve
    al comportamiento normal (menú en cada vuelta del bucle principal).
    """
    parser = argparse.ArgumentParser(
        description='Sistema de análisis de haz de fibra óptica',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Sin argumentos abre el menú gráfico.\n'
            'Con --opcion N salta directamente a esa opción (modo dev).\n\n'
            '  1 — Adquisición del haz óptico\n'
            '  2 — Recorte / preprocesado del haz óptico\n'
            '  3 — Caracterización del haz óptico (análisis)\n'
            '  4 — Adquisición: cámara de turbulencia\n'
            '  5 — Preprocesado: cámara de turbulencia\n'
            '  6 — Análisis: cámara de turbulencia\n'
            '  7 — Medición de W_fibre\n'
            '  8 — Comparación de la resiliencia de haz\n'
            '  9 — Comparación de mediciones de cámara de turbulencia'
        )
    )
    parser.add_argument(
        '--opcion',
        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        default=None,
        metavar='N',
        help='Saltar directamente a la opción N (1-9) sin mostrar el menú.'
    )
    return parser.parse_args()


# =============================================================================
# PASO 0: CALIBRACIÓN DEL TAMAÑO DE PÍXEL
# =============================================================================

def step_pixel_calibration(session: SessionConfig) -> bool:
    """
    Paso 0 de casi todas las opciones: establece la escala física µm/px
    de la sesión. Sin este valor el programa solo puede trabajar en
    píxeles, y ninguna magnitud (beam wander, ancho del haz, r₀, Cₙ²)
    tendría unidades físicas reales.

    Ofrece dos vías equivalentes:
      - Calibración óptica real (`CalibracionLongitudPixel.py`): se
        fotografía la punta de una fibra de diámetro conocido y se despeja
        µm/px del radio detectado. Es la vía correcta cada vez que cambia
        el montaje (distancia cámara-objeto, óptica, resolución).
      - Entrada manual: se reutiliza un valor ya conocido del mismo
        montaje, sin volver a fotografiar.

    Efecto secundario: propaga el valor a toda la sesión mediante
    `session.apply_um_per_px()` (actualiza también `session.scale`, que
    gobierna la barra de escala quemada en las imágenes).

    Retorna True si la sesión quedó calibrada; False si el usuario
    canceló o introdujo un valor inválido (el flujo llamante aborta).
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    do_cal = messagebox.askyesno(
        'Calibración de píxel',
        '¿Deseas realizar una calibración del tamaño de píxel\n'
        'con la punta de la fibra óptica?\n\n'
        '• SÍ  → Se pedirán los parámetros de calibración y luego\n'
        '        se abrirá la vista en vivo de la cámara.\n'
        '• NO  → Introduce el valor de µm/px manualmente.',
        parent=root)
    root.destroy()

    if do_cal:
        cal_path = os.path.join(_ROOT_DIR, 'CalibracionLongitudPixel.py')
        if not os.path.exists(cal_path):
            print_error(f'No se encontró CalibracionLongitudPixel.py en:\n  {cal_path}')
            return False
        try:
            from CalibracionLongitudPixel import calibrate_pixel_size
        except ImportError as e:
            print_error(f'Error al importar calibración: {e}')
            return False
        print('\n── CALIBRACIÓN DE TAMAÑO DE PÍXEL ──────────────────────────────')
        um_per_px = calibrate_pixel_size(session=session)
        if um_per_px is None:
            print_warn('Calibración cancelada.')
            return False
    else:
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)
        raw = simpledialog.askstring(
            'Tamaño de píxel',
            'Introduce el tamaño de píxel en µm/px:',
            initialvalue=str(session.um_per_px), parent=root2)
        root2.destroy()
        if not raw:
            print_error('No se introdujo el tamaño de píxel.')
            return False
        try:
            um_per_px = float(raw.replace(',', '.'))
        except ValueError:
            print_error(f'Valor inválido: {raw!r}')
            return False
        if um_per_px <= 0:
            print_error(f'El tamaño de píxel debe ser mayor que cero (se introdujo {um_per_px}).')
            return False
        session.apply_um_per_px(um_per_px)

    print_ok(f'Tamaño de píxel: {um_per_px:.5f} µm/px  ({1/um_per_px:.4f} px/µm)')
    print()
    return True


# =============================================================================
# PASO 1: ADQUISICIÓN
# =============================================================================

def step_acquisition(session: SessionConfig) -> bool:
    """
    Etapa de ADQUISICIÓN del haz óptico (Opción 1): abre el diálogo de
    configuración del experimento y, si el usuario confirma, ejecuta
    `Parte1.py` para capturar dark frame, estabilidad temporal y
    polarizaciones con el hardware real.

    Efecto secundario clave: deja en `session.acquisition_folder` la
    carpeta RAÍZ del dispositivo recién creada/reutilizada, y en
    `session.caso_actual` el caso adquirido. Ambos permiten encadenar
    directamente con las Opciones 2 y 3 sin volver a pedir la carpeta.

    Retorna False si el usuario cancela la configuración o si la
    adquisición no llegó a completarse.
    """
    print('\n── CONFIGURACIÓN DEL EXPERIMENTO ────────────────────────────────')
    if not run_setup_dialog(session):
        print_error('Configuración cancelada.')
        return False
    session.temporal.video_filename = session.filename_video(session.experiment.channel)
    print('\n── ADQUISICIÓN DE IMÁGENES ──────────────────────────────────────')
    folder = run_acquisition(session)
    if folder is None:
        print_error('La adquisición no se completó.')
        return False
    session.acquisition_folder = folder
    print()
    caso_label = utils_carpetas.CASO_LABELS.get(session.caso_actual, session.caso_actual)
    print_ok(f'Adquisición completada. Caso: {caso_label}. Carpeta raíz: {folder}')
    return True


# =============================================================================
# PASO 2: RECORTE
# =============================================================================

def step_cropping(session: SessionConfig) -> bool:
    """
    Etapa de PREPROCESADO del haz óptico (Opción 2): recorta las imágenes
    y el video a la región de interés (ROI) elegida por el usuario y les
    resta el dark frame propio de cada caso, dejando el resultado en
    `Preprocesado/`.

    Entrada: la carpeta raíz del dispositivo — reutilizada de la sesión si
    se viene encadenado desde la Opción 1, o pedida por diálogo si se
    ejecuta de forma independiente.

    No requiere calibración de píxel: esta etapa trabaja íntegramente en
    coordenadas de píxel, la escala física solo hace falta al analizar.

    Efecto secundario: deja la raíz en `session.cropped_folder`.
    """
    if not session.acquisition_folder:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo('Carpeta raíz',
                            'Selecciona la carpeta raíz del dispositivo\n'
                            '(contiene los casos: Sin turbulencia, Transitorio, '
                            'Con turbulencia).',
                            parent=root)
        folder = filedialog.askdirectory(title='Seleccionar carpeta raíz del dispositivo',
                                         parent=root)
        root.destroy()
        if not folder:
            print_error('No se seleccionó carpeta raíz.')
            return False
        session.acquisition_folder = utils_carpetas.normalizar_carpeta_raiz_dispositivo(folder)

    print('\n── RECORTE DE IMÁGENES ──────────────────────────────────────────')
    cropped = run_cropping(session)
    if cropped is None:
        print_error('El recorte no se completó.')
        return False
    session.cropped_folder = cropped
    print()
    print_ok(f'Recorte completado. Carpeta: {cropped}')
    return True


# =============================================================================
# PASO 3: ANÁLISIS
# =============================================================================

def step_analysis(session: SessionConfig) -> bool:
    """
    Etapa de ANÁLISIS del haz óptico (Opción 3): ejecuta `Parte3.py`, que
    extrae del video de estabilidad temporal las métricas centrales de la
    tesis (beam wander y su ajuste Rayleigh, índice de centelleo,
    correlación espacial, ancho del haz) y, si existen, analiza también
    las imágenes de polarización.

    Entrada: la carpeta raíz del dispositivo (reutilizada de la sesión o
    pedida por diálogo). Dentro de `Parte3.py` el usuario elige además
    qué casos analizar y si trabajar sobre `Adquisicion/` (crudo) o
    `Preprocesado/` (con ROI y dark frame ya aplicados).

    Salida en disco: `Analisis/` con figuras y videos, más `Datos_Crudos/`
    con el CSV/TXT de cada figura para poder rehacerla en otro software.

    Retorna False si se canceló la configuración o si ningún caso pudo
    analizarse.
    """
    print('\n── CONFIGURACIÓN DEL ANÁLISIS ───────────────────────────────────')
    if not run_analysis_dialog(session):
        print('Análisis cancelado.')
        return False

    if not (session.cropped_folder or session.acquisition_folder):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo('Carpeta raíz',
                            'Selecciona la carpeta raíz del dispositivo\n'
                            '(contiene los casos: Sin turbulencia, Transitorio, '
                            'Con turbulencia).\n\n'
                            'A continuación se te preguntará con cuáles casos y con '
                            'cuál de las dos generaciones de datos deseas trabajar.',
                            parent=root)
        folder = filedialog.askdirectory(
            title='Carpeta raíz del dispositivo', parent=root)
        root.destroy()
        if not folder:
            print_error('No se seleccionó carpeta raíz.')
            return False
        session.acquisition_folder = utils_carpetas.normalizar_carpeta_raiz_dispositivo(folder)

    print('\n── ANÁLISIS DE RESULTADOS ───────────────────────────────────────')
    if not run_analysis(session):
        print_error('El análisis no se completó (cancelado o ningún caso pudo analizarse).')
        return False
    print()
    print_ok('Análisis completado.')
    return True


# =============================================================================
# ETAPAS DE LA CARACTERIZACIÓN DE CÁMARA DE TURBULENCIA (Opciones 4-6)
# =============================================================================
# Las tres delegan en CamaraTurbulencia.py a través de sus wrappers de
# modules/. Los imports son perezosos (dentro de cada función) porque
# CamaraTurbulencia.py arrastra dependencias de hardware que no deben
# cargarse si el usuario nunca entra a estas opciones.

def step_turbulencia_adquisicion(session: SessionConfig) -> str | None:
    """
    Opción 4 — adquisición de la cámara de turbulencia: captura un dark
    frame único de sesión y luego graba un video por cada condición
    experimental (ΔT del gradiente térmico y velocidad de los
    ventiladores) que el usuario vaya introduciendo.

    El diálogo de configuración se abre aquí UNA sola vez, antes de
    entrar al bucle de casos: los parámetros ópticos (L, método de W₀,
    polarización fija) deben ser idénticos para todo el barrido, o los
    r₀/Cₙ² de casos distintos no serían comparables entre sí.

    Retorna la carpeta raíz de la caracterización creada, o None si se
    canceló.
    """
    print('\n── CONFIGURACION — CAMARA DE TURBULENCIA ────────────────────────')
    if not run_turbulence_dialog(session):   # abre el dialogo UNA sola vez
        print('Configuracion cancelada.')
        return None
    from modules.turbulencia_adquisicion import run_turbulencia_adquisicion
    return run_turbulencia_adquisicion(session)  # main_adquisicion NO reabre el dialogo


def step_turbulencia_preprocesado(session: SessionConfig) -> str | None:
    """
    Opción 5 — preprocesado de la cámara de turbulencia: aplica ROI y
    resta del dark frame a los videos crudos, generando `_proc.mp4` en
    `Preprocesado/`. Procesa frame a frame para no cargar videos enteros
    en RAM.

    No pasa por diálogo de configuración ni requiere calibración: toda la
    información necesaria (casos disponibles, dark frame) se lee de la
    carpeta de adquisición. Retorna la carpeta `Preprocesado/` o None.
    """
    from modules.turbulencia_adquisicion import run_turbulencia_preprocesado
    return run_turbulencia_preprocesado(session)


def step_turbulencia_analisis(session: SessionConfig) -> str | None:
    """
    Opción 6 — análisis de la cámara de turbulencia: sigue el centroide
    del haz a lo largo de cada video, calcula la varianza del beam wander
    (excluyendo el transitorio térmico inicial) y despeja de ahí r₀, Cₙ²
    y σ_R para cada condición ΔT/velocidad.

    Los parámetros ópticos NO se re-piden aquí: se leen de
    `metadata_adquisicion.json`, de modo que el análisis reproduce
    exactamente las condiciones de la adquisición original aunque se
    ejecute meses después. Retorna la carpeta `Analisis/` o None.
    """
    from modules.turbulencia_analisis import run_turbulencia_analisis
    return run_turbulencia_analisis(session)


# =============================================================================
# FLUJOS POR OPCIÓN
# =============================================================================
# Una función `flujo_N` por cada opción del menú (ver _FLUJOS al final).
# Todas comparten el mismo contrato: reciben la SessionConfig recién
# creada, imprimen su encabezado, encadenan los `step_*` que correspondan,
# y retornan True/False según el flujo haya completado o se haya
# cancelado — el bucle principal solo usa ese booleano para elegir el
# banner final.
#
# Qué opciones piden calibración de píxel y cuáles no NO es arbitrario:
# las que producen o interpretan magnitudes físicas en µm (1, 3, 4, 6, 7,
# 8, 9) la exigen; las de preprocesado (2 y 5) trabajan solo en píxeles y
# la omiten deliberadamente.

def flujo_1(session):
    """Opcion 1 — Solo adquisicion del haz optico."""
    print('\n' + '═'*68)
    print('  OPCION 1 — ADQUISICION DEL HAZ OPTICO')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    return step_acquisition(session)


def flujo_2(session):
    """Opcion 2 — Solo preprocesado (pide carpeta de adquisicion)."""
    print('\n' + '═'*68)
    print('  OPCION 2 — PREPROCESADO DEL HAZ OPTICO')
    print('  (pide carpeta de adquisicion, aplica ROI + dark frame)')
    print('═'*68)
    return step_cropping(session)


def flujo_3(session):
    """Opcion 3 — Solo analisis (pide carpeta de adquisicion o preprocesado)."""
    print('\n' + '═'*68)
    print('  OPCION 3 — ANALISIS DEL HAZ OPTICO')
    print('  (pide carpeta; usa el primer video encontrado en estabilidad_temporal)')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    return step_analysis(session)


def flujo_4(session):
    """Opcion 4 — Solo adquisicion de la camara de turbulencia."""
    print('\n' + '═'*68)
    print('  OPCION 4 — ADQUISICION: CAMARA DE TURBULENCIA')
    print('  (dark frame + grabacion de videos caso por caso)')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    carpeta = step_turbulencia_adquisicion(session)
    if carpeta:
        print(f'\n  Adquisicion completada: {carpeta}')
    return bool(carpeta)


def flujo_5(session):
    """Opcion 5 — Solo preprocesado de la camara de turbulencia."""
    print('\n' + '═'*68)
    print('  OPCION 5 — PREPROCESADO: CAMARA DE TURBULENCIA')
    print('  (pide carpeta raiz, aplica ROI + dark frame, guarda _proc.mp4)')
    print('═'*68)
    carpeta = step_turbulencia_preprocesado(session)
    if carpeta:
        print(f'\n  Preprocesado completado: {carpeta}')
    return bool(carpeta)


def flujo_6(session):
    """Opcion 6 — Solo analisis de la camara de turbulencia."""
    print('\n' + '═'*68)
    print('  OPCION 6 — ANALISIS: CAMARA DE TURBULENCIA')
    print('  (pide carpeta raiz/Adquisicion/Preprocesado, genera graficas)')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    carpeta = step_turbulencia_analisis(session)
    if carpeta:
        print(f'\n  Analisis completado: {carpeta}')
    return bool(carpeta)


def flujo_7(session):
    """
    Opcion 7 — Medicion de W_fibre (radio del campo modal) por segundo
    momento sobre una foto del haz colimado. Alimenta el método indirecto
    de W₀ de la Opción 6: el valor medido se propaga automáticamente a
    `session.turbulence.w_fibre_um`.
    """
    print('\n' + '═'*68)
    print('  OPCION 7 — MEDICION W_FIBRE')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    w = run_wfibre(session)
    if w:
        print(f'\n  W_fibre = {w:.4f} um')
    return bool(w)


def flujo_8(session):
    """
    Opcion 8 — Comparacion de la resiliencia de haz. Es la opción que
    responde la hipótesis central de la tesis: contrasta N haces (el
    gaussiano de control frente a los estructurados) bajo las MISMAS 3
    condiciones de turbulencia, para determinar cuál preserva mejor su
    perfil espacial y su estabilidad geométrica.
    """
    print('\n' + '═'*68)
    print('  OPCION 8 — COMPARACION DE LA RESILIENCIA DE HAZ')
    print('  (compara N haces bajo distintas condiciones de turbulencia)')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    from modules.comparacion_resiliencia import run_comparacion_resiliencia
    carpeta = run_comparacion_resiliencia(session)
    if carpeta:
        print(f'\n  Comparacion completada: {carpeta}')
    return bool(carpeta)


def flujo_9(session):
    """
    Opcion 9 — Comparacion de mediciones de camara de turbulencia:
    superpone r₀/Cₙ²/σ_R de varios barridos independientes para evaluar
    la REPRODUCIBILIDAD de la cámara de turbulencia (¿las mismas
    condiciones ΔT/velocidad dan los mismos parámetros en días
    distintos?), requisito para poder usarla como instrumento de
    referencia en la Opción 8.
    """
    print('\n' + '═'*68)
    print('  OPCION 9 — COMPARACION DE MEDICIONES DE CAMARA DE TURBULENCIA')
    print('  (compara Cn2 y r0 entre distintos ensayos de caracterizacion)')
    print('═'*68)
    if not step_pixel_calibration(session): return False
    from modules.comparacion_turbulencia import run_comparacion_turbulencia
    carpeta = run_comparacion_turbulencia(session)
    if carpeta:
        print(f'\n  Comparacion completada: {carpeta}')
    return bool(carpeta)


_FLUJOS = {'1': flujo_1, '2': flujo_2, '3': flujo_3, '4': flujo_4,
           '5': flujo_5, '6': flujo_6, '7': flujo_7, '8': flujo_8,
           '9': flujo_9}


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():
    """
    Bucle principal del programa: muestra el menú, ejecuta el flujo
    elegido y vuelve al menú, indefinidamente, hasta que el usuario
    cierra la ventana.

    El proceso Python permanece vivo durante toda la sesión de trabajo
    (los módulos se importan una sola vez), pero cada vuelta arranca con
    una `SessionConfig` nueva para que la configuración de una opción no
    contamine la siguiente — ver el comentario dentro del bucle.
    """
    args = _parse_args()

    print_banner('SISTEMA DE ANÁLISIS DE HAZ DE FIBRA ÓPTICA')

    while True:
        if args.opcion is not None:
            opcion      = args.opcion
            args.opcion = None          # solo se aplica una vez
        else:
            opcion = run_main_menu()    # muestra el menú gráfico

        if opcion is None:
            print('\nPrograma finalizado desde el menú principal.')
            break

        # SessionConfig() nuevo en cada vuelta: los parámetros de una opción
        # nunca se filtran a la siguiente. El proceso Python sigue vivo entre
        # opciones (los módulos no se reimportan), así que el estado a nivel
        # de módulo de cada script raíz (ej. Parte3.py::SESSION,
        # CamaraTurbulencia.py::SESSION) sigue siendo responsabilidad de cada
        # script — ver Parte3.py::_patch_analysis_functions para el caso ya
        # corregido de un patrón de este tipo.
        session = SessionConfig()
        completado = _FLUJOS[opcion](session)

        if completado:
            print_banner('FLUJO COMPLETADO — Regresando al menú principal...')
        else:
            print_banner('FLUJO CANCELADO — Regresando al menú principal...')
        print()


if __name__ == '__main__':
    main()
