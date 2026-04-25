"""T1.a — Lectura cruda de los 4 datasets locales.

Los archivos ya están en la carpeta del proyecto (no se descargan).
Este módulo solo expone lectores que devuelven DataFrames/GeoDataFrames
sin transformar; la limpieza vive en `src/cleaning.py`.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.config import RUTA_CCPP, RUTA_DISTRITOS, RUTA_EMERGENCIAS, RUTA_IPRESS
from src.utils import obtener_logger

log = obtener_logger(__name__)


def leer_ipress_raw() -> pd.DataFrame:
    """IPRESS MINSA — Excel (.xlsx)."""
    log.info(f"Leyendo IPRESS desde {RUTA_IPRESS.name}")
    return pd.read_excel(RUTA_IPRESS, engine="openpyxl")


def leer_emergencias_raw() -> pd.DataFrame:
    """SUSALUD producción de emergencias 2026 — Excel (.xlsx)."""
    log.info(f"Leyendo emergencias desde {RUTA_EMERGENCIAS.name}")
    return pd.read_excel(RUTA_EMERGENCIAS, engine="openpyxl")


def leer_ccpp_raw() -> gpd.GeoDataFrame:
    """Centros Poblados IGN escala 1:100 000 — Shapefile."""
    log.info(f"Leyendo CCPP desde {RUTA_CCPP.name}")
    return gpd.read_file(RUTA_CCPP)


def leer_distritos_raw() -> gpd.GeoDataFrame:
    """Distritos del Perú — Shapefile."""
    log.info(f"Leyendo distritos desde {RUTA_DISTRITOS.name}")
    return gpd.read_file(RUTA_DISTRITOS)


# --- Main: verificación de lectura ------------------------------------------

def main() -> None:
    log.info("=== data_loader: verificando lectura de archivos locales ===")
    for nombre, fn in [
        ("IPRESS",       leer_ipress_raw),
        ("Emergencias",  leer_emergencias_raw),
        ("CCPP",         leer_ccpp_raw),
        ("Distritos",    leer_distritos_raw),
    ]:
        df = fn()
        log.info(f"  {nombre}: {len(df):,} filas × {len(df.columns)} columnas")
    log.info("Lectura OK — corre `python -m src.cleaning` para limpiar.")


if __name__ == "__main__":
    main()
