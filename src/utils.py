"""Utilidades compartidas: normalización de UBIGEO, logging, helpers varios."""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd


def obtener_logger(nombre: str) -> logging.Logger:
    """Logger con formato consistente; idempotente si ya fue configurado."""
    logger = logging.getLogger(nombre)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s — %(message)s", "%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_RE_NO_DIGITOS = re.compile(r"\D")


def estandarizar_ubigeo(serie: pd.Series, longitud: int = 6) -> pd.Series:
    """Deja solo dígitos, rellena con ceros a la izquierda y trunca a `longitud`."""
    limpia = serie.astype("string").str.replace(_RE_NO_DIGITOS, "", regex=True)
    limpia = limpia.where(limpia.str.len() > 0, other=pd.NA)
    return limpia.str.zfill(longitud).str.slice(0, longitud)


def primera_existente(df: pd.DataFrame, candidatos: Iterable[str]) -> str | None:
    """Devuelve el primer nombre de columna de `candidatos` que exista en `df`, o None."""
    for col in candidatos:
        if col in df.columns:
            return col
    return None
