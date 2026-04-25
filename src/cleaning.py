"""T1.b — Limpieza, normalización y persistencia de los 4 datasets.

Lee los archivos crudos vía `src.data_loader`, aplica la limpieza
documentada y persiste los resultados a `data/processed/` como parquet.

Ejecutar con: python -m src.cleaning
"""

from __future__ import annotations

import unicodedata
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import CRS_GEO, DATA_PROCESSED
from src.data_loader import (
    leer_ccpp_raw,
    leer_distritos_raw,
    leer_emergencias_raw,
    leer_ipress_raw,
)
from src.utils import estandarizar_ubigeo, obtener_logger, primera_existente

log = obtener_logger(__name__)

# Rango de coordenadas válido para Perú continental
LAT_PERU = (-18.5,  0.5)
LON_PERU = (-81.5, -68.0)


# --- Helper interno ---------------------------------------------------------

def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """ASCII-ify + UPPER_SNAKE_CASE en todos los nombres de columna."""
    def _limpiar(c: str) -> str:
        c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode()
        return re.sub(r"[\s\.]+", "_", c.strip().upper())
    df = df.copy()
    df.columns = [_limpiar(c) for c in df.columns]
    return df


# --- Limpiezas por dataset --------------------------------------------------

def limpiar_ipress(raw: pd.DataFrame) -> pd.DataFrame:
    """IPRESS MINSA — corrige columnas NORTE/ESTE invertidas, filtra y deduplicada."""
    df = _normalizar_columnas(raw)
    log.info(f"IPRESS raw: {len(df):,} filas × {len(df.columns)} cols")

    # En la fuente original NORTE contiene longitud y ESTE contiene latitud (bug conocido)
    df = df.rename(columns={
        "CODIGO_UNICO":               "codigo",
        "NOMBRE_DEL_ESTABLECIMIENTO": "nombre",
        "UBIGEO":                     "ubigeo",
        "INSTITUCION":                "institucion",
        "CLASIFICACION":              "clasificacion",
        "CATEGORIA":                  "categoria",
        "NORTE":                      "longitud",
        "ESTE":                       "latitud",
    })

    cols_keep = ["codigo", "nombre", "ubigeo", "institucion",
                 "clasificacion", "categoria", "latitud", "longitud"]
    df = df[[c for c in cols_keep if c in df.columns]].copy()

    df["latitud"]  = pd.to_numeric(df["latitud"],  errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df["ubigeo"]   = estandarizar_ubigeo(df["ubigeo"])
    df["codigo"]   = df["codigo"].astype("string").str.strip()

    antes = len(df)
    df = df.dropna(subset=["latitud", "longitud", "ubigeo"])
    df = df[df["latitud"].between(*LAT_PERU) & df["longitud"].between(*LON_PERU)]
    df = df.drop_duplicates(subset=["codigo"])
    log.info(f"IPRESS limpio: {len(df):,} establecimientos (eliminados: {antes - len(df):,})")
    return df.reset_index(drop=True)


def limpiar_emergencias(raw: pd.DataFrame) -> pd.DataFrame:
    """SUSALUD 2026 — coerción numérica y suma anual por (codigo, ubigeo)."""
    df = _normalizar_columnas(raw)
    log.info(f"Emergencias raw: {len(df):,} filas × {len(df.columns)} cols")

    df = df.rename(columns={
        "CO_IPRESS":            "codigo",
        "UBIGEO":               "ubigeo",
        "NRO_TOTAL_ATENCIONES": "atenciones",
        "NRO_TOTAL_ATENDIDOS":  "atendidos",
        "ANHO":                 "anho",
        "MES":                  "mes",
    })

    df["atenciones"] = pd.to_numeric(df["atenciones"], errors="coerce")
    df["atendidos"]  = pd.to_numeric(df["atendidos"],  errors="coerce")
    df["codigo"]     = df["codigo"].astype("string").str.strip()
    df["ubigeo"]     = estandarizar_ubigeo(df["ubigeo"])

    agregado = (
        df.assign(
            atenciones=df["atenciones"].fillna(0),
            atendidos =df["atendidos"].fillna(0),
        )
        .groupby(["codigo", "ubigeo"], dropna=False, as_index=False)[["atenciones", "atendidos"]]
        .sum()
    )
    log.info(
        f"Emergencias agregadas: {len(agregado):,} registros únicos · "
        f"total atenciones: {agregado['atenciones'].sum():,.0f}"
    )
    return agregado


def limpiar_ccpp(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centros Poblados IGN — deriva UBIGEO desde los primeros 6 dígitos del código."""
    gdf = raw.copy()
    geom_col = gdf.geometry.name
    log.info(f"CCPP raw: {len(gdf):,} centros poblados | CRS: {gdf.crs}")

    col_codigo = primera_existente(gdf, ["CODIGO", "CÓDIGO", "COD_INT", "CÓD_INT"])
    if col_codigo is None:
        raise RuntimeError(f"CCPP: no se encontró columna de código en {list(gdf.columns)}")

    col_nombre = primera_existente(gdf, ["NOM_POBLAD", "NOMBRE", "NOM_CP"])
    col_cat    = primera_existente(gdf, ["CAT_POBLAD", "CATEGORIA", "CAT_CP"])

    rename = {col_codigo: "codigo"}
    if col_nombre: rename[col_nombre] = "nombre"
    if col_cat:    rename[col_cat]    = "categoria"
    gdf = gdf.rename(columns=rename)

    gdf["codigo"] = gdf["codigo"].astype("string").str.strip()
    gdf["ubigeo"] = estandarizar_ubigeo(gdf["codigo"].str.slice(0, 6))
    gdf["peso"]   = 1   # peso uniforme — sin datos de población disponibles

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_GEO)
    gdf = gdf.to_crs(CRS_GEO)

    cols_keep = ["ubigeo", "codigo", "nombre", "categoria", "peso", geom_col]
    gdf = gdf[[c for c in cols_keep if c in gdf.columns]].copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    log.info(f"CCPP limpio: {len(gdf):,} centros poblados")
    return gdf.reset_index(drop=True)


def limpiar_distritos(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Distritos — UBIGEO desde IDDIST + reproyección a EPSG:4326."""
    gdf = raw.copy()
    geom_col = gdf.geometry.name

    gdf = gdf.rename(columns={
        "IDDIST":     "ubigeo",
        "DISTRITO":   "distrito",
        "PROVINCIA":  "provincia",
        "DEPARTAMEN": "departamento",
    })

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_GEO)
    gdf = gdf.to_crs(CRS_GEO)
    gdf["ubigeo"] = estandarizar_ubigeo(gdf["ubigeo"])

    cols_keep = ["ubigeo", "distrito", "provincia", "departamento", geom_col]
    gdf = gdf[[c for c in cols_keep if c in gdf.columns]].copy()
    log.info(f"Distritos listos: {len(gdf):,}")
    return gdf.reset_index(drop=True)


# --- Persistencia -----------------------------------------------------------

def _guardar_parquet(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta)
    log.info(f"guardado {ruta.name} ({len(df):,} filas)")


# --- Pipeline completo ------------------------------------------------------

def main() -> None:
    """T1 completo: lectura → limpieza → persistencia."""
    log.info("=== T1: INGESTA Y LIMPIEZA ===")

    ipress      = limpiar_ipress(leer_ipress_raw())
    emergencias = limpiar_emergencias(leer_emergencias_raw())
    ccpp        = limpiar_ccpp(leer_ccpp_raw())
    distritos   = limpiar_distritos(leer_distritos_raw())

    _guardar_parquet(ipress,      DATA_PROCESSED / "ipress_limpio.parquet")
    _guardar_parquet(emergencias, DATA_PROCESSED / "emergencias_anuales.parquet")
    _guardar_parquet(ccpp,        DATA_PROCESSED / "ccpp_limpio.parquet")
    _guardar_parquet(distritos,   DATA_PROCESSED / "distritos_limpio.parquet")

    log.info("=== T1 completo ===")


if __name__ == "__main__":
    main()
