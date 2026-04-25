"""
Aplicación Streamlit — Acceso Distrital a Servicios de Emergencia en Perú
Ejecutar: streamlit run app_streamlit.py
"""

from pathlib import Path
import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import spearmanr
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"C:/Users/Matias/OneDrive/Documentos/jupyter/homework2")
PROCESADOS = BASE_DIR / "processed"
FIGURAS    = BASE_DIR / "figuras"
MAPAS      = BASE_DIR / "mapas"

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Acceso Distrital a Emergencias — Perú",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="ticks", context="notebook")

# ── Carga de datos (cacheada) ─────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    metricas  = gpd.read_parquet(PROCESADOS / "metricas_distritales.parquet")
    ccpp_dist = gpd.read_parquet(PROCESADOS / "ccpp_distancias.parquet")
    ipress    = gpd.read_parquet(PROCESADOS / "ipress_geo.parquet")
    return metricas, ccpp_dist, ipress

metricas, ccpp_dist, ipress = cargar_datos()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Datos & Metodología",
    "📊 Análisis Estático",
    "🗺️ Resultados Geoespaciales",
    "🔍 Exploración Interactiva",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATOS & METODOLOGÍA
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🏥 Acceso Distrital a Servicios de Emergencia en Perú")
    st.markdown("---")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Problema y objetivo analítico")
        st.markdown("""
        Perú tiene una marcada heterogeneidad territorial en la distribución de servicios de salud.
        Este proyecto evalúa **qué tan accesibles son los servicios de emergencia** en los 1 873
        distritos del país, combinando tres dimensiones:

        1. **Oferta** — densidad de establecimientos de salud con actividad de emergencia.
        2. **Actividad** — volumen de atenciones de emergencia registradas en 2026.
        3. **Acceso espacial** — proximidad de los centros poblados a esos establecimientos.
        """)

        st.subheader("Fuentes de datos")
        tabla_fuentes = pd.DataFrame({
            "Dataset": ["IPRESS MINSA", "SUSALUD Emergencias 2026", "CCPP IGN 100K", "DISTRITOS"],
            "Formato": ["Excel (.xlsx)", "Excel (.xlsx)", "Shapefile (.shp)", "Shapefile (.shp)"],
            "Descripción": [
                "Establecimientos de salud registrados (MINSA)",
                "Producción asistencial de emergencia por IPRESS",
                "Centros Poblados escala 1:100 000 (IGN)",
                "Polígonos distritales del Perú",
            ],
            "Registros usados": [
                f"{len(ipress):,}",
                f"{len(ccpp_dist):,} (CPs únicos)",
                f"{len(ccpp_dist):,}",
                f"{len(metricas):,}",
            ]
        })
        st.dataframe(tabla_fuentes, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Resumen de limpieza")
        st.markdown("""
        **IPRESS:**
        - Columnas NORTE/ESTE invertidas en la fuente → corregidas al renombrar.
        - Se eliminan filas sin coordenadas válidas o fuera del rango continental del Perú.
        - Duplicados por código único eliminados (primera ocurrencia).

        **Emergencias:**
        - Valores no numéricos tratados como NaN y luego 0.
        - Agregación anual por establecimiento.

        **Centros Poblados:**
        - UBIGEO derivado de los primeros 6 dígitos del código IGN.
        - Peso uniforme = 1 (sin datos de población disponibles).

        **Distritos:**
        - Reproyectados a WGS84 (EPSG:4326) para compatibilidad con Folium.
        """)

    st.markdown("---")
    st.subheader("🔧 Construcción del Índice de Acceso a Emergencias (IAE)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Especificación Baseline (umbral 25 km)**

        | Componente | Medida | Peso |
        |---|---|---|
        | Oferta | log(emergencias / 100 km²), min-max | 35% |
        | Actividad | log(atenciones totales), min-max | 30% |
        | Acceso | share de CPs a ≤ 25 km del IPRESS más cercano | 35% |

        `IAE_base = 0.35·oferta + 0.30·actividad + 0.35·acceso`
        """)
    with col2:
        st.markdown("""
        **Especificación Alternativa (umbral 10 km)**

        | Componente | Medida | Peso |
        |---|---|---|
        | Oferta | log(emergencias / N° CPs), min-max | 40% |
        | Actividad | log(atenciones totales), min-max | 25% |
        | Acceso | share de CPs a ≤ 10 km del IPRESS más cercano | 35% |

        `IAE_alt = 0.40·oferta + 0.25·actividad + 0.35·acceso`
        """)

    st.markdown("---")
    st.subheader("⚠️ Limitaciones principales")
    st.markdown("""
    - Los **Centros Poblados IGN** no incluyen datos de población, por lo que se asume peso uniforme. Distritos con muchos CPs pequeños son tratados igual que los que tienen pocos CPs grandes.
    - La distancia calculada es **euclidea en UTM** (línea recta), no distancia por red vial. Subestima el tiempo real de acceso en zonas de sierra y selva.
    - El dataset de emergencias (ConsultaC1_2026_v5) puede tener sub-registro en establecimientos con baja capacidad de digitalización.
    - El umbral de 25 km / 10 km es una decisión metodológica; no existe un estándar normativo peruano oficial para este análisis.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISIS ESTÁTICO
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Análisis Estático")
    st.markdown("Cada figura responde una o más de las preguntas analíticas requeridas.")

    figuras_info = [
        ("01_distribucion_iae.png",
         "Distribución del IAE (Baseline vs Alternativa)",
         "Responde Q1 y Q4. El histograma doble muestra cómo la especificación alternativa "
         "(umbral más estricto de 10 km) desplaza la distribución hacia valores menores, "
         "revelando que muchos distritos que parecen aceptables con 25 km dejan de serlo con 10 km."),
        ("02_ranking_acceso_espacial.png",
         "Ranking: peor acceso espacial (Top 25)",
         "Responde Q2. Barras horizontales con color secundario (share de CPs dentro del umbral) "
         "condensan dos dimensiones. Se eligió horizontal porque los nombres de distrito son largos."),
        ("03_descomposicion_iae.png",
         "Descomposición del IAE: mejores y peores",
         "Responde Q3. El stacked bar revela qué componente falta en los peores distritos "
         "(casi siempre actividad y oferta) y qué equilibra a los mejores."),
        ("04_histograma_distancias.png",
         "Distancia CP → IPRESS de emergencia",
         "Responde Q1 y Q2. KDE + histograma muestra la distribución real; las líneas verticales "
         "contextualizan ambos umbrales y el porcentaje de CPs que quedan dentro de cada uno."),
        ("05_violin_departamento.png",
         "Violin plot por departamento",
         "Responde Q3. El violin muestra la distribución completa dentro de cada región, "
         "incluyendo bimodalidad (e.g. capitales bien servidas + distritos rurales sin acceso)."),
        ("06_sensibilidad_y_correlacion.png",
         "Sensibilidad metodológica + correlación de componentes",
         "Responde Q4. El scatter detecta distritos cuya evaluación cambia al modificar el umbral. "
         "El heatmap muestra que oferta y actividad están más correlacionadas entre sí que con acceso."),
    ]

    for fname, titulo, interpretacion in figuras_info:
        ruta = FIGURAS / fname
        st.subheader(titulo)
        if ruta.exists():
            st.image(str(ruta), use_container_width=True)
        else:
            st.warning(f"Figura no encontrada: {fname}. Ejecuta el notebook 04 primero.")
        with st.expander("📝 Interpretación"):
            st.markdown(interpretacion)
        st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTADOS GEOESPACIALES
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🗺️ Resultados Geoespaciales")

    # ── Mapas estáticos ──────────────────────────────────────────────────────
    st.subheader("Mapas estáticos — Comparación de especificaciones")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        ruta_b = MAPAS / "mapa_iae_baseline.png"
        if ruta_b.exists():
            st.image(str(ruta_b), caption="IAE Baseline (25 km · oferta/km²)", use_container_width=True)
        else:
            st.warning("Ejecuta el notebook 05 para generar los mapas.")

    with col_m2:
        ruta_a = MAPAS / "mapa_iae_alternativa.png"
        if ruta_a.exists():
            st.image(str(ruta_a), caption="IAE Alternativa (10 km · oferta/CP)", use_container_width=True)

    ruta_biv = MAPAS / "mapa_bivariado_oferta_acceso.png"
    if ruta_biv.exists():
        st.subheader("Mapa bivariado: Oferta × Acceso")
        st.image(str(ruta_biv), use_container_width=True)
        st.markdown(
            "El mapa bivariado identifica cuatro patrones clave: "
            "**(1)** alta oferta y alto acceso (azul oscuro — mejor situación), "
            "**(2)** baja oferta y bajo acceso (rojo — peor situación), "
            "**(3)** alta oferta pero bajo acceso (aislamiento geográfico), "
            "**(4)** bajo acceso a la oferta existente (dispersión poblacional)."
        )

    st.markdown("---")

    # ── Tabla comparativa distrital ──────────────────────────────────────────
    st.subheader("Comparación distrital: Baseline vs Alternativa")

    cols_tabla = [c for c in [
        "departamento", "provincia", "distrito",
        "iae_base", "iae_alt",
        "n_establecimientos", "n_con_emergencia",
        "total_atenciones", "dist_media_km",
    ] if c in metricas.columns]

    df_tabla = metricas[cols_tabla].copy()
    df_tabla["dif_iae"] = (df_tabla["iae_alt"] - df_tabla["iae_base"]).round(3)

    # Filtros de la barra lateral
    st.sidebar.header("Filtros (Tab 3 y 4)")
    deptos = sorted(metricas["departamento"].dropna().unique().tolist())
    deptos_sel = st.sidebar.multiselect("Departamento(s)", deptos, default=[])
    if deptos_sel:
        df_tabla = df_tabla[df_tabla["departamento"].isin(deptos_sel)]

    show_worst = st.sidebar.slider("N° distritos a mostrar", 10, 200, 50)

    col_ord = st.selectbox("Ordenar tabla por:", ["iae_base", "iae_alt", "dist_media_km", "dif_iae"])
    asc_ord = st.radio("Orden:", ["Ascendente (peores primero)", "Descendente (mejores primero)"])
    asc = "Ascendente" in asc_ord

    df_show = df_tabla.sort_values(col_ord, ascending=asc).head(show_worst)
    st.dataframe(df_show.round(3), use_container_width=True, hide_index=True)

    # Resumen por departamento
    st.subheader("Resumen por departamento")
    df_dep = (
        metricas.groupby("departamento", dropna=True)
        .agg(
            n_distritos=("ubigeo", "count"),
            iae_mediana=("iae_base", "median"),
            iae_p25    =("iae_base", lambda x: x.quantile(0.25)),
            iae_p75    =("iae_base", lambda x: x.quantile(0.75)),
            pct_sin_emerg=("n_con_emergencia", lambda x: (x == 0).mean() * 100),
        )
        .reset_index()
        .sort_values("iae_mediana")
    )
    st.dataframe(df_dep.round(3), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLORACIÓN INTERACTIVA
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🔍 Exploración Interactiva")

    st.subheader("Mapa Folium — IAE por distrito + IPRESS de emergencia")
    st.markdown(
        "Haz clic en cualquier distrito para ver sus métricas. "
        "Los marcadores rojos son IPRESS con atenciones de emergencia registradas."
    )

    ruta_html = MAPAS / "mapa_interactivo.html"
    if ruta_html.exists():
        with open(ruta_html, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=620, scrolling=False)
    else:
        st.info("El mapa interactivo aún no está generado. Ejecuta el notebook 05 primero.")

    st.markdown("---")

    # ── Comparación dinámica baseline vs alternativa ──────────────────────────
    st.subheader("Comparación dinámica: Baseline vs Alternativa")
    rho, _ = spearmanr(
        metricas["iae_base"].dropna(),
        metricas["iae_alt"].dropna()
    )

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.metric("Correlación de Spearman (ρ)", f"{rho:.4f}")
        st.markdown(
            "Un ρ cercano a 1 indica que **el ranking de distritos cambia poco** al "
            "pasar de la especificación baseline a la alternativa. "
            "Valores de ρ < 0.85 indicarían una sensibilidad metodológica alta."
        )
    with col_der:
        umbral_sel = st.slider("Umbral de distancia a evaluar (km)", 5, 60, 25, 5)
        share_dentro = (ccpp_dist["dist_emergencia_km"] <= umbral_sel).mean() * 100
        st.metric(
            f"% de CPs dentro de {umbral_sel} km",
            f"{share_dentro:.1f}%",
            delta=f"{share_dentro - (ccpp_dist['dist_emergencia_km'] <= 25).mean()*100:.1f}pp vs 25 km"
        )

    # ── Scatter interactivo (via matplotlib en Streamlit) ────────────────────
    st.subheader("Scatter interactivo: Baseline vs Alternativa")
    df_sc = metricas.dropna(subset=["iae_base", "iae_alt"])

    if deptos_sel:
        df_sc_plot = df_sc[df_sc["departamento"].isin(deptos_sel)]
    else:
        df_sc_plot = df_sc

    fig_sc, ax_sc = plt.subplots(figsize=(7, 6))
    ax_sc.scatter(df_sc["iae_base"], df_sc["iae_alt"],
                  alpha=0.2, s=14, color="#aaaaaa", label="Todos los distritos")
    if len(df_sc_plot) < len(df_sc):
        ax_sc.scatter(df_sc_plot["iae_base"], df_sc_plot["iae_alt"],
                      alpha=0.7, s=25, color="#c0392b", label="Selección actual")
    ax_sc.plot([0,1],[0,1], color="#2980b9", linestyle="--", linewidth=1.5, label="Acuerdo perfecto")
    ax_sc.set_xlabel("IAE Baseline", fontsize=11)
    ax_sc.set_ylabel("IAE Alternativa", fontsize=11)
    ax_sc.set_xlim(0,1); ax_sc.set_ylim(0,1); ax_sc.set_aspect("equal")
    ax_sc.legend(fontsize=9)
    ax_sc.set_title(f"Baseline vs Alternativa · ρ = {rho:.3f}", fontsize=12)
    sns.despine(fig_sc)
    st.pyplot(fig_sc, use_container_width=True)
    plt.close(fig_sc)

    st.markdown("---")
    st.subheader("Hallazgos principales")
    st.markdown("""
    1. **La mayoría de los distritos con IAE = 0** no tienen ningún IPRESS con actividad de
       emergencia registrada y sus centros poblados están a más de 25 km del más cercano.
       Se concentran en la sierra y selva alta.

    2. **El peor acceso espacial** (distancias medias > 80 km) se observa en distritos
       amazónicos y de la sierra sur, donde la red vial es escasa y la orografía dificulta el traslado.

    3. **Los mejor atendidos** son distritos urbanos capitales de provincia o departamento,
       donde la oferta de IPRESS es alta, la actividad está documentada y los centros poblados
       están concentrados cerca de los establecimientos.

    4. **La sensibilidad metodológica** (Q4) muestra que el ranking general es estable (ρ > 0.85),
       pero los distritos periurbanos intermedios son los más sensibles al cambio de umbral:
       pasan de "aceptable" a "deficiente" cuando se exige 10 km en lugar de 25 km.
    """)
