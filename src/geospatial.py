"""T2 — Integración geoespacial.

Lee los parquet limpios de T1, construye GeoDataFrames, reproyecta a UTM 18S
para cálculos de distancia en metros, valida ubigeos con spatial join,
calcula la distancia de cada centro poblado al IPRESS de emergencia más
cercano usando KDTree, y agrega todas las métricas a nivel distrital.

Ejecutar con: python -m src.geospatial
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import CRS_GEO, CRS_UTM, DATA_PROCESSED
from src.utils import obtener_logger

log = obtener_logger(__name__)


# --- Carga ------------------------------------------------------------------

def cargar_procesados() -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ipress      = pd.read_parquet(DATA_PROCESSED / "ipress_limpio.parquet")
    emergencias = pd.read_parquet(DATA_PROCESSED / "emergencias_anuales.parquet")
    ccpp        = gpd.read_parquet(DATA_PROCESSED / "ccpp_limpio.parquet")
    distritos   = gpd.read_parquet(DATA_PROCESSED / "distritos_limpio.parquet")
    return ipress, emergencias, ccpp, distritos


# --- GeoDataFrame IPRESS ----------------------------------------------------

def ipress_a_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs=CRS_GEO,
    )


# --- Unir actividad de emergencias ------------------------------------------

def unir_actividad_emergencias(
    gdf_est: gpd.GeoDataFrame, df_em: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Une atenciones de emergencia a IPRESS. Flag es_emergencia = atenciones > 0.
    El join se hace por codigo sin ceros iniciales para evitar mismatches de padding."""
    est = gdf_est.copy()
    est["codigo_cmp"] = est["codigo"].astype("string").str.lstrip("0")

    em = df_em.copy()
    em["codigo_cmp"] = em["codigo"].astype("string").str.lstrip("0")
    em_suma = em.groupby("codigo_cmp", as_index=False)["atenciones"].sum()

    merged = est.merge(em_suma, on="codigo_cmp", how="left")
    merged["atenciones"]   = merged["atenciones"].fillna(0).astype(int)
    merged["es_emergencia"] = merged["atenciones"] > 0
    return merged.drop(columns=["codigo_cmp"])


# --- Validar UBIGEO via spatial join ----------------------------------------

def validar_ubigeo_sjoin(
    gdf_est: gpd.GeoDataFrame, gdf_dist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Asigna el UBIGEO del polígono distrital en que cae cada IPRESS.
    Cuando difiere del declarado, se usa el derivado de geometría."""
    derecha = gdf_dist[["ubigeo", "geometry"]].rename(columns={"ubigeo": "ubigeo_geom"})
    joined  = gpd.sjoin(gdf_est, derecha, how="left", predicate="within")
    joined  = joined.loc[~joined.index.duplicated(keep="first")]
    joined  = joined.drop(columns=["index_right"], errors="ignore")
    joined["ubigeo_final"] = joined["ubigeo_geom"].fillna(joined["ubigeo"])
    return joined


# --- Distancia CP → IPRESS de emergencia ------------------------------------

def distancia_emergencia_mas_cercana(
    gdf_ccpp: gpd.GeoDataFrame,
    gdf_est:  gpd.GeoDataFrame,
    solo_emergencia: bool = True,
) -> gpd.GeoDataFrame:
    """Distancia en km al IPRESS (de emergencia si solo_emergencia=True) más cercano.
    Usa UTM 18S para distancias euclídeas métricas precisas."""
    pool = gdf_est[gdf_est["es_emergencia"]] if solo_emergencia else gdf_est
    if len(pool) == 0:
        raise RuntimeError("Sin IPRESS en el pool para calcular distancias")

    pool_utm = pool.to_crs(CRS_UTM)
    ccpp_utm = gdf_ccpp.to_crs(CRS_UTM)

    coords_pool  = np.column_stack([pool_utm.geometry.x.values, pool_utm.geometry.y.values])
    coords_query = np.column_stack([ccpp_utm.geometry.x.values, ccpp_utm.geometry.y.values])

    arbol = cKDTree(coords_pool)
    distancias_m, idx = arbol.query(coords_query, k=1)

    resultado = gdf_ccpp.copy()
    resultado["dist_emergencia_km"]    = distancias_m / 1_000.0
    resultado["codigo_ipress_cercano"] = pool_utm["codigo"].values[idx]
    return resultado


# --- Agregación a nivel distrital -------------------------------------------

def agregar_por_distrito(
    gdf_est:      gpd.GeoDataFrame,
    ccpp_con_dist: gpd.GeoDataFrame,
    gdf_dist:     gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    col_ub = "ubigeo_final" if "ubigeo_final" in gdf_est.columns else "ubigeo"

    metricas_est = (
        gdf_est.groupby(col_ub, dropna=True)
        .agg(
            n_establecimientos=("codigo",        "count"),
            n_con_emergencia  =("es_emergencia", "sum"),
            total_atenciones  =("atenciones",    "sum"),
        )
        .reset_index()
        .rename(columns={col_ub: "ubigeo"})
    )

    metricas_ccpp = (
        ccpp_con_dist.groupby("ubigeo", dropna=True)
        .agg(
            n_ccpp          =("codigo",             "count"),
            dist_media_km   =("dist_emergencia_km", "mean"),
            dist_mediana_km =("dist_emergencia_km", "median"),
            dist_max_km     =("dist_emergencia_km", "max"),
        )
        .reset_index()
    )

    out = gdf_dist.merge(metricas_est,  on="ubigeo", how="left")
    out = out.merge(metricas_ccpp, on="ubigeo", how="left")

    for col in ["n_establecimientos", "n_con_emergencia", "total_atenciones", "n_ccpp"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    return out


# --- Main -------------------------------------------------------------------

def main() -> None:
    log.info("=== T2: INTEGRACIÓN GEOESPACIAL ===")

    ipress_df, emergencias_df, ccpp_gdf, distritos_gdf = cargar_procesados()
    log.info(
        f"cargados — ipress: {len(ipress_df):,} | emergencias: {len(emergencias_df):,} | "
        f"ccpp: {len(ccpp_gdf):,} | distritos: {len(distritos_gdf):,}"
    )

    gdf_est = ipress_a_geodataframe(ipress_df)
    gdf_est = unir_actividad_emergencias(gdf_est, emergencias_df)
    log.info(
        f"IPRESS con emergencia: {int(gdf_est['es_emergencia'].sum()):,} | "
        f"total atenciones: {int(gdf_est['atenciones'].sum()):,}"
    )

    gdf_est = validar_ubigeo_sjoin(gdf_est, distritos_gdf)
    n_reasig = int(
        (gdf_est["ubigeo_geom"].notna() & (gdf_est["ubigeo_geom"] != gdf_est["ubigeo"])).sum()
    )
    n_fuera = int(gdf_est["ubigeo_geom"].isna().sum())
    log.info(f"sjoin: {n_reasig:,} reasignados por geometría | {n_fuera:,} fuera de polígonos")

    ccpp_con_dist = distancia_emergencia_mas_cercana(ccpp_gdf, gdf_est, solo_emergencia=True)
    d = ccpp_con_dist["dist_emergencia_km"]
    log.info(
        f"distancias CP→emergencia: media={d.mean():.1f} km | "
        f"mediana={d.median():.1f} km | max={d.max():.1f} km"
    )

    dist_integrado = agregar_por_distrito(gdf_est, ccpp_con_dist, distritos_gdf)
    sin_ipress = int((dist_integrado["n_establecimientos"] == 0).sum())
    sin_em     = int((dist_integrado["n_con_emergencia"]   == 0).sum())
    log.info(f"distritos sin IPRESS: {sin_ipress:,} | sin IPRESS emergencia: {sin_em:,}")

    gdf_est.to_parquet(DATA_PROCESSED / "ipress_geo.parquet")
    ccpp_con_dist.to_parquet(DATA_PROCESSED / "ccpp_distancias.parquet")
    dist_integrado.to_parquet(DATA_PROCESSED / "distritos_integrado.parquet")
    log.info("=== T2 completo ===")


if __name__ == "__main__":
    main()
