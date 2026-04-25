"""T4 + T5 — Visualizaciones estáticas (matplotlib/seaborn) y geoespaciales (Folium).

Estáticas (T4) — output/figuras/:
  01_distribucion_iae.png          — histograma doble baseline vs alternativa
  02_ranking_acceso_espacial.png   — top 25 peor acceso (Q2)
  03_descomposicion_iae.png        — stacked bars con 3 componentes (Q3)
  04_histograma_distancias.png     — KDE de distancias CP→emergencia
  05_violin_departamento.png       — violin por departamento (Q1 y Q3)
  06_sensibilidad_correlacion.png  — scatter + heatmap (Q4)

Geoespaciales (T5) — output/mapas/:
  mapa_iae_baseline.png
  mapa_iae_alternativa.png
  mapa_share_acceso.png
  mapa_bivariado_oferta_acceso.png
  mapa_interactivo.html            — Folium con IPRESS clusterizados

Ejecutar con: python -m src.visualization
"""

from __future__ import annotations

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster
from scipy.stats import spearmanr

from src.config import CRS_GEO, DATA_PROCESSED, FIGURAS, MAPAS
from src.utils import obtener_logger

log = obtener_logger(__name__)
sns.set_theme(style="ticks", context="notebook")


def _guardar(fig: plt.Figure, nombre: str, carpeta=FIGURAS) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"guardado {ruta.relative_to(carpeta.parent)}")


# =====================================================================
# T4 — VISUALIZACIONES ESTÁTICAS
# =====================================================================

def graficar_distribucion_iae(metricas: gpd.GeoDataFrame) -> None:
    """Histograma doble baseline vs alternativa + tabla de extremos. (Q1, Q4)"""
    vals_b = metricas["iae_base"].dropna()
    vals_a = metricas["iae_alt"].dropna()
    bins   = np.linspace(0, 1, 41)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals_b, bins=bins, color="#1a6b8a", alpha=0.6,
            label=f"Baseline · 25 km · oferta/km² (n={len(vals_b):,})", edgecolor="white")
    ax.hist(vals_a, bins=bins, color="#c0392b", alpha=0.5,
            label=f"Alternativa · 10 km · oferta/CP (n={len(vals_a):,})", edgecolor="white")
    ax.axvline(vals_b.median(), color="#1a6b8a", linestyle="--", linewidth=1.4,
               label=f"Mediana base = {vals_b.median():.3f}")
    ax.axvline(vals_a.median(), color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"Mediana alt  = {vals_a.median():.3f}")
    ax.set_xlabel("IAE [0 = peor atendido · 1 = mejor atendido]", fontsize=11)
    ax.set_ylabel("N° de distritos", fontsize=11)
    ax.set_title("Distribución del IAE: Baseline vs Alternativa\n"
                 "El umbral estricto (10 km) desplaza la masa hacia valores menores", fontsize=12)
    ax.legend(fontsize=9)
    sns.despine(fig)
    _guardar(fig, "01_distribucion_iae.png")


def graficar_ranking_acceso(metricas: gpd.GeoDataFrame, n: int = 25) -> None:
    """Top N distritos con peor acceso espacial. (Q2)"""
    df = metricas.dropna(subset=["dist_media_km"]).copy()
    df["etiqueta"] = (df["distrito"].str.title() + " (" +
                      df["departamento"].str[:3].str.title() + ")")
    peores = df.nlargest(n, "dist_media_km").sort_values("dist_media_km")

    norm    = mcolors.Normalize(0, 1)
    cmap_fn = cm.get_cmap("RdYlGn")
    colores = [cmap_fn(norm(v)) for v in peores["share_ccpp_25km"].fillna(0)]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(peores["etiqueta"], peores["dist_media_km"],
                   color=colores, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, peores["dist_media_km"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} km", va="center", fontsize=8)

    ax.axvline(25, color="navy",    linestyle="--", linewidth=1.2, label="Umbral baseline (25 km)")
    ax.axvline(10, color="#e67e22", linestyle=":",  linewidth=1.2, label="Umbral alternativo (10 km)")
    sm = cm.ScalarMappable(cmap=cmap_fn, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Share CPs ≤ 25 km", fontsize=9)
    ax.set_xlabel("Distancia media CP → IPRESS emergencia (km)", fontsize=11)
    ax.set_title(f"Distritos con peor acceso espacial — Top {n} (Q2)\n"
                 "Color: proporción de CPs dentro del umbral baseline", fontsize=12)
    ax.legend(fontsize=9)
    sns.despine(fig)
    _guardar(fig, "02_ranking_acceso_espacial.png")


def graficar_descomposicion_iae(metricas: gpd.GeoDataFrame,
                                 n_peores: int = 12, n_mejores: int = 12) -> None:
    """Stacked bars con las 3 componentes del IAE. (Q3)"""
    df = metricas.dropna(subset=["iae_base"]).copy()
    df["etiqueta"] = (df["distrito"].str.title() + " (" +
                      df["departamento"].str[:3].str.title() + ")")
    peores  = df.sort_values(["iae_base", "n_ccpp"], ascending=[True, False]).head(n_peores)
    mejores = df.nlargest(n_mejores, "iae_base")
    plot_df = pd.concat([peores, mejores]).sort_values("iae_base").reset_index(drop=True)

    p = dict(oferta=0.35, actividad=0.30, acceso=0.35)
    c_of = plot_df["oferta_01"]    * p["oferta"]
    c_ac = plot_df["actividad_01"] * p["actividad"]
    c_ac2= plot_df["acceso_01"]    * p["acceso"]

    fig, ax = plt.subplots(figsize=(11, 10))
    y = np.arange(len(plot_df))
    ax.barh(y, c_of,  color="#2980b9", label="Oferta (35%)",    edgecolor="white", lw=0.3)
    ax.barh(y, c_ac,  left=c_of,       color="#e67e22",
            label="Actividad (30%)", edgecolor="white", lw=0.3)
    ax.barh(y, c_ac2, left=c_of + c_ac, color="#27ae60",
            label="Acceso (35%)", edgecolor="white", lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["etiqueta"], fontsize=8.5)
    ax.axhline(n_peores - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("IAE = suma ponderada de componentes [0, 1]", fontsize=11)
    ax.set_title(f"Descomposición del IAE: {n_peores} peor + {n_mejores} mejor atendidos (Q3)\n"
                 "Línea punteada divide peores (abajo) de mejores (arriba)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9)
    sns.despine(fig)
    _guardar(fig, "03_descomposicion_iae.png")


def graficar_histograma_distancias(ccpp_dist: gpd.GeoDataFrame) -> None:
    """KDE + histograma de distancias CP → emergencia con umbrales. (Q1, Q2)"""
    d = ccpp_dist["dist_emergencia_km"].dropna()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(d, bins=70, color="#1a6b8a", alpha=0.65, stat="count",
                 kde=True, line_kws={"color": "#0d3a4e", "linewidth": 2}, ax=ax)
    ax.axvline(10, color="#e67e22", linestyle="--", linewidth=2,
               label=f"Umbral alt 10 km  ({(d<=10).mean()*100:.1f}% CPs dentro)")
    ax.axvline(25, color="#c0392b", linestyle="--", linewidth=2,
               label=f"Umbral base 25 km ({(d<=25).mean()*100:.1f}% CPs dentro)")
    ax.set_xlim(0, d.quantile(0.99))
    ax.set_xlabel("Distancia al IPRESS con emergencia más cercano (km)", fontsize=11)
    ax.set_ylabel("N° de centros poblados", fontsize=11)
    ax.set_title(f"Distribución de distancias CP → IPRESS de emergencia\n"
                 f"n={len(d):,} · mediana={d.median():.1f} km · media={d.mean():.1f} km",
                 fontsize=12)
    ax.legend(fontsize=10)
    sns.despine(fig)
    _guardar(fig, "04_histograma_distancias.png")


def graficar_violin_departamento(metricas: gpd.GeoDataFrame) -> None:
    """Violin plot del IAE por departamento. (Q1, Q3)"""
    df = metricas.dropna(subset=["iae_base", "departamento"]).copy()
    df["Departamento"] = df["departamento"].str.title()
    orden = df.groupby("Departamento")["iae_base"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.violinplot(data=df, y="Departamento", x="iae_base",
                   order=orden, inner="quartile", linewidth=0.8, color="#5b8db8", ax=ax)
    sns.stripplot(data=df, y="Departamento", x="iae_base",
                  order=orden, color="black", size=1.5, alpha=0.3, ax=ax)
    med_nac = df["iae_base"].median()
    ax.axvline(med_nac, color="red", linestyle="--", linewidth=1,
               label=f"Mediana nacional = {med_nac:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("IAE baseline [0 = sin acceso · 1 = acceso pleno]", fontsize=11)
    ax.set_title("Distribución del IAE por departamento (violin plot)\n"
                 "Ordenado de peor mediana (arriba) a mejor (abajo)", fontsize=12)
    ax.legend(fontsize=9)
    sns.despine(fig)
    _guardar(fig, "05_violin_departamento.png")


def graficar_sensibilidad(metricas: gpd.GeoDataFrame) -> None:
    """Scatter baseline vs alternativa + heatmap de correlación. (Q4)"""
    df = metricas.dropna(subset=["iae_base", "iae_alt"])
    rho, _ = spearmanr(df["iae_base"], df["iae_alt"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(df["iae_base"], df["iae_alt"], alpha=0.35, s=18,
               color="#2c3e50", edgecolors="white", linewidths=0.2)
    ax.plot([0, 1], [0, 1], color="#e74c3c", linestyle="--", linewidth=1.5,
            label="Acuerdo perfecto (y = x)")
    ax.text(0.05, 0.92, "Mejora con\numbral estricto", fontsize=8, color="#27ae60")
    ax.text(0.72, 0.08, "Empeora con\numbral estricto", fontsize=8, color="#c0392b")
    ax.set_xlabel("IAE Baseline (25 km · oferta/km²)", fontsize=11)
    ax.set_ylabel("IAE Alternativa (10 km · oferta/CP)", fontsize=11)
    ax.set_title(f"Sensibilidad metodológica\nSpearman ρ = {rho:.3f}", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    comp_cols   = ["oferta_01", "actividad_01", "acceso_01", "iae_base"]
    comp_labels = ["Oferta", "Actividad", "Acceso", "IAE base"]
    corr = metricas[comp_cols].corr(method="spearman")
    corr.index = comp_labels; corr.columns = comp_labels
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax2, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.7})
    ax2.set_title("Correlación Spearman\nentre componentes del índice", fontsize=12)

    fig.tight_layout()
    _guardar(fig, "06_sensibilidad_correlacion.png")


# =====================================================================
# T5 — SALIDAS GEOESPACIALES
# =====================================================================

def choropleth(
    metricas: gpd.GeoDataFrame,
    columna: str,
    titulo: str,
    nombre_archivo: str,
    cmap: str = "RdYlGn",
    etiqueta_leyenda: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 13))
    metricas.plot(
        column=columna, cmap=cmap,
        linewidth=0.08, edgecolor="#aaaaaa",
        legend=True,
        vmin=0 if any(k in columna for k in ("iae", "share")) else None,
        vmax=1 if any(k in columna for k in ("iae", "share")) else None,
        missing_kwds={"color": "#d3d3d3", "label": "sin dato"},
        legend_kwds={"label": etiqueta_leyenda or columna, "shrink": 0.45},
        ax=ax,
    )
    ax.set_title(titulo, fontsize=13, pad=10)
    ax.set_axis_off()
    _guardar(fig, nombre_archivo, MAPAS)


def mapa_bivariado(metricas: gpd.GeoDataFrame) -> None:
    """Mapa bivariado oferta × acceso (3×3 clases)."""
    df = metricas.copy()
    df["q_oferta"] = pd.qcut(df["oferta_01"].rank(method="first"),         q=3, labels=[0, 1, 2])
    df["q_acceso"] = pd.qcut(df["acceso_01"].fillna(0).rank(method="first"), q=3, labels=[0, 1, 2])
    df["clase_biv"] = df["q_oferta"].astype(int) * 3 + df["q_acceso"].astype(int)

    paleta = [
        "#f7f7f7", "#b8d4e3", "#3e90bf",
        "#f4a9b2", "#b99bb8", "#4b83b4",
        "#c0392b", "#9e4f7e", "#1b3a8e",
    ]
    cmap_biv = ListedColormap(paleta)

    fig, (ax_map, ax_leg) = plt.subplots(1, 2, figsize=(14, 12),
                                          gridspec_kw={"width_ratios": [4, 1]})
    df.plot(column="clase_biv", categorical=True, cmap=cmap_biv,
            linewidth=0.08, edgecolor="#aaaaaa", legend=False, ax=ax_map)
    ax_map.set_title("Mapa bivariado: Oferta de emergencia × Acceso desde CPs\n"
                     "Azul oscuro = alta oferta y acceso · Rojo = baja oferta y acceso",
                     fontsize=12, pad=10)
    ax_map.set_axis_off()

    ax_leg.set_xlim(-0.5, 2.5); ax_leg.set_ylim(-0.5, 2.5)
    for i in range(3):
        for j in range(3):
            ax_leg.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor=paleta[i * 3 + j], edgecolor="white"))
    ax_leg.set_xticks([0, 1, 2]); ax_leg.set_xticklabels(["bajo", "medio", "alto"], fontsize=8)
    ax_leg.set_yticks([0, 1, 2]); ax_leg.set_yticklabels(["bajo", "medio", "alto"], fontsize=8)
    ax_leg.set_xlabel("Acceso (share CPs ≤ 25 km)", fontsize=9)
    ax_leg.set_ylabel("Oferta (emerg./100 km², log-norm.)", fontsize=9)
    ax_leg.set_title("Leyenda 3×3"); ax_leg.set_aspect("equal")

    _guardar(fig, "mapa_bivariado_oferta_acceso.png", MAPAS)


def mapa_interactivo_folium(
    metricas: gpd.GeoDataFrame, gdf_ipress: gpd.GeoDataFrame
) -> None:
    centro = [-9.19, -75.01]
    m = folium.Map(location=centro, zoom_start=5,
                   tiles="CartoDB positron", control_scale=True)

    col = "iae_base"
    gdf = metricas.dropna(subset=[col]).to_crs(CRS_GEO).copy()

    escala = LinearColormap(
        ["#c0392b", "#f9d342", "#1e8449"],
        vmin=0.0, vmax=1.0,
        caption="IAE Baseline — Índice de Acceso a Emergencias [0 = peor · 1 = mejor]",
    )
    escala.add_to(m)

    def estilo(feature):
        v = feature["properties"].get(col)
        return {
            "fillColor":   escala(v) if v is not None else "#cccccc",
            "color":       "#555555",
            "weight":      0.3,
            "fillOpacity": 0.72,
        }

    campos  = [col, "distrito", "provincia", "departamento",
               "n_establecimientos", "n_con_emergencia", "share_ccpp_25km"]
    aliases = ["IAE baseline:", "Distrito:", "Provincia:", "Departamento:",
               "IPRESS total:", "IPRESS emergencia:", "Share CPs ≤ 25 km:"]

    folium.GeoJson(
        data=gdf[[*campos, "geometry"]].to_json(),
        name="Distritos — IAE baseline",
        style_function=estilo,
        tooltip=folium.GeoJsonTooltip(
            fields=campos, aliases=aliases, localize=True, sticky=False
        ),
    ).add_to(m)

    em = gdf_ipress[gdf_ipress["es_emergencia"]].to_crs(CRS_GEO)
    cluster = MarkerCluster(name=f"IPRESS con emergencia (n={len(em):,})").add_to(m)
    for _, row in em.iterrows():
        popup = folium.Popup(
            f"<b>{row.get('nombre', '')}</b><br>"
            f"Ubigeo: {row.get('ubigeo_final') or row.get('ubigeo', '')}<br>"
            f"Atenciones 2026: {int(row.get('atenciones', 0)):,}<br>"
            f"Institución: {row.get('institucion', '')}",
            max_width=300,
        )
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4, color="#c0392b",
            fill=True, fill_opacity=0.8, popup=popup,
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    MAPAS.mkdir(parents=True, exist_ok=True)
    ruta = MAPAS / "mapa_interactivo.html"
    m.save(str(ruta))
    log.info(f"guardado {ruta.name}")


# =====================================================================
# Main — T4 + T5 completo
# =====================================================================

def main() -> None:
    log.info("=== T4 + T5: VISUALIZACIÓN ===")
    metricas  = gpd.read_parquet(DATA_PROCESSED / "metricas_distritales.parquet")
    ccpp_dist = gpd.read_parquet(DATA_PROCESSED / "ccpp_distancias.parquet")
    ipress    = gpd.read_parquet(DATA_PROCESSED / "ipress_geo.parquet")

    # T4 — estáticas
    graficar_distribucion_iae(metricas)
    graficar_ranking_acceso(metricas, n=25)
    graficar_descomposicion_iae(metricas)
    graficar_histograma_distancias(ccpp_dist)
    graficar_violin_departamento(metricas)
    graficar_sensibilidad(metricas)

    # T5 — geoespaciales
    choropleth(metricas, "iae_base",
               "IAE Baseline (25 km · oferta/km² · pesos 35/30/35)",
               "mapa_iae_baseline.png", legend_label="IAE Baseline [0–1]")
    choropleth(metricas, "iae_alt",
               "IAE Alternativa (10 km · oferta/CP · pesos 40/25/35)",
               "mapa_iae_alternativa.png", cmap="RdYlGn", legend_label="IAE Alternativa [0–1]")
    choropleth(metricas, "share_ccpp_25km",
               "Proporción de CPs con emergencia a ≤ 25 km",
               "mapa_share_acceso.png", cmap="YlGnBu", legend_label="Share CPs ≤ 25 km")
    mapa_bivariado(metricas)
    mapa_interactivo_folium(metricas, ipress)

    log.info("=== T4 + T5 completo ===")


if __name__ == "__main__":
    main()
