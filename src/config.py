"""Rutas, CRS y constantes del proyecto."""

from pathlib import Path

# --- Paths ---
ROOT           = Path(r"C:\Users\Matias\OneDrive\Documentos\jupyter\emergency_access_peru-main")
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT         = ROOT / "output"
FIGURAS        = OUTPUT / "figuras"
MAPAS          = OUTPUT / "mapas"
TABLAS         = OUTPUT / "tablas"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURAS, MAPAS, TABLAS):
    _p.mkdir(parents=True, exist_ok=True)

# --- CRS ---
CRS_GEO = "EPSG:4326"    # WGS84 geográfico — para Folium y datos fuente
CRS_UTM = "EPSG:32718"   # UTM zona 18S — métrico, para cálculos de distancia

# --- Archivos locales (ya descargados) ---
RUTA_IPRESS    = ROOT / "IPRESS.xlsx"
RUTA_EMERGENCIAS = ROOT / "ConsultaC1_2026_v5.xlsx"
RUTA_CCPP      = ROOT / "CCPP_IGN100K.shp"
RUTA_DISTRITOS = ROOT / "DISTRITOS.shp"

# --- Parámetros de análisis ---
UMBRAL_BASE_KM = 25.0   # umbral de acceso — especificación baseline
UMBRAL_ALT_KM  = 10.0   # umbral de acceso — especificación alternativa (más estricta)

# Pesos del IAE por especificación (deben sumar 1.0)
PESOS_BASE = dict(oferta=0.35, actividad=0.30, acceso=0.35)
PESOS_ALT  = dict(oferta=0.40, actividad=0.25, acceso=0.35)
