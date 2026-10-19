"""
Очистка данных для задачи предсказания правильности решения задачи.

Модуль содержит функции для приведения попыток решения задач к виду,
пригодному для feature engineering и обучения моделей.
"""

import pandas as pd

# Служебные значения time_sec, которые не являются реальным временем решения:
# -5     — попытка не начата,
#  0     — мгновенный ответ (вероятно, клик),
#  86400 — ровно сутки (таймаут),
#  99999 — заглушка.
INVALID_TIME_SEC = (-5, 0, 86400, 99999)


def _parse_ts(series: pd.Series) -> pd.Series:
    """
    Приводит колонку с датой к datetime64.

    Поддерживает два формата, встречающихся в данных:
      - DD.MM.YYYY HH:MM:SS
      - YYYY-MM-DD HH:MM:SS

    Некорректные значения превращаются в NaT.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    dot_mask = series.str.match(r"^\d{2}\.\d{2}\.\d{4}", na=False)
    iso_mask = series.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    parsed[dot_mask] = pd.to_datetime(
        series[dot_mask], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    )
    parsed[iso_mask] = pd.to_datetime(
        series[iso_mask], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    return parsed


def clean_attempts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Чистит таблицу попыток решения задач.

    Шаги:
      1. Приводит `ts` к datetime (два поддерживаемых формата).
      2. Удаляет дубликаты по `attempt_id`.
      3. Заменяет служебные значения `time_sec` на пропуски.
    """
    df = df.copy()

    df["ts"] = _parse_ts(df["ts"])
    df = df.drop_duplicates(subset="attempt_id")
    df.loc[df["time_sec"].isin(INVALID_TIME_SEC), "time_sec"] = pd.NA

    return df