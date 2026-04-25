"""T3 — Índice de Acceso a Emergencias (IAE) en escala [0, 1].

Construye dos especificaciones:

  * BASELINE   : oferta por km², umbral de acceso 25 km, pesos 35/30/35.
  * ALTERNATIVA: oferta por centro poblado, umbral estricto 10 km, pesos 40/25/35.

Cada componente se normaliza a [0, 1] con min-max (previa transformación log
para las dimensiones sesgadas) y se combina con los pesos definidos en config.
IAE = 1 → mejor atendido | IAE = 0 → peor atendido.

Ejecutar con: python -m src.metrics
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    CRS_UTM, DATA_PROCESSED, TABLAS,
    PESOS_BASE, PESOS_ALT,
    UMBRAL_BASE_KM, UMBRAL_ALT_KM,
)
from src.utils import obtener_logger

log = obtener_logger(__name__)


# --- Helpers ----------------------------------------------------------------

def _area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    return gdf.to_crs(CRS_UTM).geometry.area / 1e6


def _minmax_01(serie: pd.Series) -> pd.Series:
    """Normalización min-max a [0, 1]. Devuelve 0.0 si la serie es constante."""
    s = pd.Series(serie, dtype="float64")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(hi - lo) or (hi - lo) == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1).fillna(0.0)


def _share_acceso(
    ccpp: gpd.GeoDataFrame, umbral_km: float, nombre_col: str
) -> pd.DataFrame:
    """Proporción de CPs por distrito cuya distancia es ≤ umbral_km."""
    df = ccpp[["ubigeo", "dist_emergencia_km"]].copy()
    df["dentro"] = df["dist_emergencia_km"] <= umbral_km
    return (
        df.groupby("ubigeo", dropna=True, as_index=False)["dentro"]
        .mean()
        .rename(columns={"dentro": nombre_col})
    )


# --- Especificación Baseline ------------------------------------------------

def calcular_baseline(
    dist_int: gpd.GeoDataFrame, ccpp: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Baseline: oferta por km², acceso a 25 km, pesos 35/30/35."""
    df = dist_int.copy()
    df["area_km2"]         = _area_km2(df)
    area_segura            = df["area_km2"].replace(0, np.nan)
    df["estab_por_100km2"] = 100 * df["n_establecimientos"] / area_segura
    df["emerg_por_100km2"] = 100 * df["n_con_emergencia"]   / area_segura

    acc = _share_acceso(ccpp, UMBRAL_BASE_KM, "share_ccpp_25km")
    df  = df.merge(acc, on="ubigeo", how="left")

    # Componentes normalizados [0, 1]
    df["oferta_01"]    = _minmax_01(np.log1p(df["emerg_por_100km2"].fillna(0)))
    df["actividad_01"] = _minmax_01(np.log1p(df["total_atenciones"].fillna(0)))
    df["acceso_01"]    = df["share_ccpp_25km"].fillna(0).clip(0, 1)

    # IAE con pesos diferenciados
    df["iae_base"] = (
        PESOS_BASE["oferta"]    * df["oferta_01"] +
        PESOS_BASE["actividad"] * df["actividad_01"] +
        PESOS_BASE["acceso"]    * df["acceso_01"]
    )
    return df


# --- Especificación Alternativa ---------------------------------------------

def calcular_alternativa(
    dist_int: gpd.GeoDataFrame, ccpp: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Alternativa: oferta por CP, acceso a 10 km, pesos 40/25/35."""
    df = dist_int.copy()
    ccpp_seguro         = df["n_ccpp"].replace(0, np.nan)
    df["estab_por_ccpp"] = df["n_establecimientos"] / ccpp_seguro
    df["emerg_por_ccpp"] = df["n_con_emergencia"]   / ccpp_seguro

    acc = _share_acceso(ccpp, UMBRAL_ALT_KM, "share_ccpp_10km")
    df  = df.merge(acc, on="ubigeo", how="left")

    df["oferta_01_alt"]    = _minmax_01(np.log1p(df["emerg_por_ccpp"].fillna(0)))
    df["actividad_01_alt"] = _minmax_01(np.log1p(df["total_atenciones"].fillna(0)))
    df["acceso_01_alt"]    = df["share_ccpp_10km"].fillna(0).clip(0, 1)

    df["iae_alt"] = (
        PESOS_ALT["oferta"]    * df["oferta_01_alt"] +
        PESOS_ALT["actividad"] * df["actividad_01_alt"] +
        PESOS_ALT["acceso"]    * df["acceso_01_alt"]
    )
    return df


# --- Sensibilidad -----------------------------------------------------------

def tabla_sensibilidad(
    baseline: gpd.GeoDataFrame, alternativa: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, float]:
    cols_b = ["ubigeo", "departamento", "provincia", "distrito", "iae_base"]
    cols_a = ["ubigeo", "iae_alt"]
    unido  = baseline[cols_b].merge(alternativa[cols_a], on="ubigeo", how="inner")
    unido  = unido.dropna(subset=["iae_base", "iae_alt"])

    unido["rango_base"] = unido["iae_base"].rank(ascending=True, method="min").astype(int)
    unido["rango_alt"]  = unido["iae_alt"].rank( ascending=True, method="min").astype(int)
    unido["dif_rango"]  = unido["rango_alt"] - unido["rango_base"]
    unido = unido.sort_values("rango_base").reset_index(drop=True)

    rho, _ = spearmanr(unido["rango_base"], unido["rango_alt"])
    return unido, float(rho)


# --- Main -------------------------------------------------------------------

def main() -> None:
    log.info("=== T3: ÍNDICE DE ACCESO A EMERGENCIAS (IAE) ===")

    dist   = gpd.read_parquet(DATA_PROCESSED / "distritos_integrado.parquet")
    ccpp   = gpd.read_parquet(DATA_PROCESSED / "ccpp_distancias.parquet")

    base = calcular_baseline(dist, ccpp)
    alt  = calcular_alternativa(dist, ccpp)

    # Sanity check
    for col, df in [("iae_base", base), ("iae_alt", alt)]:
        v = df[col].dropna()
        assert v.between(0.0, 1.0).all(), f"{col} fuera de [0, 1]"
    log.info("Sanity check OK — ambos IAE en [0, 1]")

    sensibilidad, rho = tabla_sensibilidad(base, alt)
    log.info(f"Spearman(rango_base, rango_alt) = {rho:.4f}")

    # Guardar CSVs de métricas
    TABLAS.mkdir(parents=True, exist_ok=True)
    base[[c for c in base.columns if c != "geometry"]].to_csv(
        TABLAS / "metricas_base.csv", index=False
    )
    alt[[c for c in alt.columns if c != "geometry"]].to_csv(
        TABLAS / "metricas_alternativa.csv", index=False
    )
    sensibilidad.to_csv(TABLAS / "sensibilidad_rankings.csv", index=False)

    # Parquet consolidado
    overlap = [c for c in alt.columns if c in base.columns and c not in ("ubigeo", "geometry")]
    metricas = base.merge(
        alt.drop(columns=overlap + (["geometry"] if "geometry" in alt.columns else []), errors="ignore"),
        on="ubigeo", how="left",
    )
    metricas.to_parquet(DATA_PROCESSED / "metricas_distritales.parquet")
    log.info(f"metricas_distritales.parquet — {len(metricas):,} filas × {len(metricas.columns)} cols")

    # Resumen ejecutivo
    log.info("TOP 10 PEOR atendidos:")
    print(
        base.sort_values(["iae_base", "n_ccpp"], ascending=[True, False])
        .head(10)[["departamento", "distrito", "iae_base", "oferta_01", "actividad_01", "acceso_01"]]
        .to_string(index=False)
    )

    log.info("TOP 10 MEJOR atendidos:")
    print(
        base.nlargest(10, "iae_base")
        [["departamento", "distrito", "iae_base", "oferta_01", "actividad_01", "acceso_01"]]
        .to_string(index=False)
    )

    log.info("=== T3 completo ===")


if __name__ == "__main__":
    main()
