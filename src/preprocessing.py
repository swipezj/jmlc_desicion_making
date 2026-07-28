"""
Предобработка: сырые архивы приборов учёта -> категориальные входы сети.

ГРАНИЦА ОТВЕТСТВЕННОСТИ. Модуль ничего не знает про конкретные имена столбцов
в источнике: они задаются в `ColumnMap`. Причина в том, что набор параметров
зависит от модели прибора (тепловычислитель, термопреобразователь, датчик
давления отдают разное), а состав узла учёта различается от объекта к объекту.
Захардкоженная схема здесь означала бы, что модуль работает на одной выгрузке
и ломается на второй.

ЧТО СЧИТАЕТСЯ, А ЧТО НЕТ. Модуль отвечает за приведение к пяти
категориальным признакам, которые ждёт `network_spec.STATES`. Он НЕ заполняет
пропуски догадками: если параметра нет или он вне допустимого физического
диапазона, признак остаётся `None`. Сеть принимает частичное свидетельство,
поэтому «неизвестно» -- корректный вход, а подстановка «норма» вместо
пропуска создала бы ложную уверенность.

ПОРОГИ ВЗЯТЫ НЕ ИЗ ВОЗДУХА:
  * температура -- отклонение от температурного графика, допуск ±3 %
    (ПТЭ тепловых энергоустановок, приказ Минэнерго России от 24.03.2003
    № 115). Опорная точка графика ищется по СРЕДНЕСУТОЧНОЙ уличной
    температуре, а не по мгновенной;
  * поверка -- дата последней поверки плюс межповерочный интервал из
    паспорта узла;
  * срок эксплуатации -- дата ввода в эксплуатацию;
  * давление и расход -- диапазон из паспорта узла, если он задан; иначе
    робастная оценка по собственной истории объекта (медиана и MAD).

Запуск как CLI: python -m src.preprocessing --help
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# Физически возможные границы: значения за ними -- сбой связи или пересчёт
# единиц, а не состояние объекта. Такие точки помечаются как пропуски.
PLAUSIBLE = {
    "t_supply": (0.0, 200.0),      # °C
    "t_outdoor": (-70.0, 50.0),    # °C
    "pressure": (0.0, 25.0),       # кгс/см² либо бар -- см. ColumnMap.pressure_unit
    "volume": (0.0, 1e7),          # м³ за сутки
}

# Температурный график 95/70 -- пример по умолчанию. Реальный график берётся
# из договора теплоснабжения и задаётся вызывающей стороной.
DEFAULT_CURVE_OUTDOOR = (8.0, 0.0, -10.0, -20.0, -30.0)
DEFAULT_CURVE_SUPPLY = (44.0, 58.0, 72.0, 84.0, 95.0)


@dataclass(frozen=True)
class ColumnMap:
    """Соответствие столбцов выгрузки ролям в модели.

    Заполняется под конкретную выгрузку. `None` означает «параметр в источнике
    отсутствует» -- соответствующий признак будет не рассчитан, а не угадан.
    """
    object_id: str = "object_id"
    timestamp: str = "ts"
    t_supply: str | None = "t1"
    pressure: str | None = "p1"
    volume: str | None = "v1"
    t_outdoor: str | None = "t_out"
    pressure_unit: str = "кгс/см2"


@dataclass(frozen=True)
class HeatingCurve:
    """Температурный график: уличная температура -> температура подачи."""
    outdoor: tuple[float, ...] = DEFAULT_CURVE_OUTDOOR
    supply: tuple[float, ...] = DEFAULT_CURVE_SUPPLY
    tolerance_pct: float = 3.0

    def expected(self, t_outdoor: np.ndarray) -> np.ndarray:
        """Линейная интерполяция по графику. За краями -- ближайшее значение.

        `np.interp` требует возрастающего аргумента, а график задаётся от тепла
        к холоду, поэтому оси переворачиваются.
        """
        x = np.asarray(self.outdoor, dtype=float)[::-1]
        y = np.asarray(self.supply, dtype=float)[::-1]
        return np.interp(np.asarray(t_outdoor, dtype=float), x, y)


@dataclass(frozen=True)
class Passport:
    """Паспорт узла учёта.

    ВАЖНОЕ ОГРАНИЧЕНИЕ МОДЕЛИ. Узел учёта -- это комплект: вычислитель,
    термопреобразователи, преобразователь расхода, датчики давления. У каждого
    компонента собственный межповерочный интервал и собственный срок службы.
    В сети же по одному узлу `calibration` и `age`, поэтому здесь берётся
    ХУДШИЙ компонент комплекта: ближайшая истекающая поверка и самый старый
    элемент. Это осознанное упрощение; разделение по компонентам потребовало
    бы менять структуру сети.
    """
    object_id: str
    last_verification: pd.Timestamp | None = None
    verification_interval_months: int | None = None
    commissioned: pd.Timestamp | None = None
    pressure_range: tuple[float, float] | None = None
    volume_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class Config:
    columns: ColumnMap = field(default_factory=ColumnMap)
    curve: HeatingCurve = field(default_factory=HeatingCurve)
    # окно усреднения уличной температуры, ч (ПТЭ: 12-24)
    outdoor_window_hours: int = 24
    # во сколько раз суточный объём должен отличаться от медианы объекта,
    # чтобы считаться аномальным, когда паспортного диапазона нет
    flow_mad_k: float = 3.0
    # минимум суток истории, ниже которого робастная оценка не считается
    min_history_days: int = 14


# ---------------------------------------------------------------- служебное
def _clip_implausible(s: pd.Series, kind: str) -> pd.Series:
    lo, hi = PLAUSIBLE[kind]
    return s.where((s >= lo) & (s <= hi))


def _robust_flag(x: pd.Series, k: float, min_n: int) -> pd.Series:
    """Робастная классификация отклонений: 'norma' / 'high' / 'low' / None.

    Медиана и MAD вместо среднего и стандартного отклонения: выборка по
    объекту короткая, а выбросы -- это ровно то, что мы ищем, поэтому оценка
    масштаба не должна ими портиться.
    """
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    v = x.dropna()
    if len(v) < min_n:
        return out
    med = v.median()
    mad = (v - med).abs().median()
    if mad == 0:
        scale = v.std(ddof=0)
        if not scale or np.isnan(scale):
            out[x.notna()] = "norma"
            return out
    else:
        scale = 1.4826 * mad          # приведение MAD к σ нормального закона
    z = (x - med) / scale
    out[x.notna() & (z.abs() <= k)] = "norma"
    out[x.notna() & (z > k)] = "high"
    out[x.notna() & (z < -k)] = "low"
    return out


# ---------------------------------------------------------------- проверки
def validate(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Список проблем выгрузки. Пустой список -- можно считать признаки."""
    c, problems = cfg.columns, []
    for role, name in (("object_id", c.object_id), ("timestamp", c.timestamp)):
        if name not in df.columns:
            problems.append(f"нет обязательного столбца '{name}' (роль {role})")
    if problems:
        return problems

    ts = pd.to_datetime(df[c.timestamp], errors="coerce")
    if ts.isna().any():
        problems.append(f"'{c.timestamp}': {int(ts.isna().sum())} значений не разобрались как дата")

    for role in ("t_supply", "pressure", "volume", "t_outdoor"):
        name = getattr(c, role)
        if name is None:
            problems.append(f"параметр {role} не сопоставлен: признак не будет рассчитан")
            continue
        if name not in df.columns:
            problems.append(f"столбец '{name}' (роль {role}) отсутствует в выгрузке")
            continue
        num = pd.to_numeric(df[name], errors="coerce")
        n_bad = int(num.isna().sum())
        if n_bad:
            problems.append(f"'{name}': {n_bad} нечисловых или пустых значений "
                            f"({n_bad / len(df):.1%})")
        out_of_range = int((~num.between(*PLAUSIBLE[role])).sum() - num.isna().sum())
        if out_of_range > 0:
            problems.append(f"'{name}': {out_of_range} значений вне физического "
                            f"диапазона {PLAUSIBLE[role]} -- будут отброшены")
    return problems


# ---------------------------------------------------------------- агрегация
def to_daily(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Часовые или суточные архивы -> одна строка на объект и сутки.

    Уличная температура усредняется скользящим окном (ПТЭ требует 12-24 ч),
    остальные параметры -- средним за сутки, объём -- суммой.
    """
    c = cfg.columns
    d = df.copy()
    d[c.timestamp] = pd.to_datetime(d[c.timestamp], errors="coerce")
    d = d.dropna(subset=[c.timestamp, c.object_id])

    for role in ("t_supply", "pressure", "volume", "t_outdoor"):
        name = getattr(c, role)
        if name and name in d.columns:
            d[name] = _clip_implausible(pd.to_numeric(d[name], errors="coerce"), role)

    d["date"] = d[c.timestamp].dt.floor("D")
    agg: dict[str, str] = {}
    if c.t_supply and c.t_supply in d: agg[c.t_supply] = "mean"
    if c.pressure and c.pressure in d: agg[c.pressure] = "mean"
    if c.volume and c.volume in d: agg[c.volume] = "sum"
    if c.t_outdoor and c.t_outdoor in d: agg[c.t_outdoor] = "mean"
    if not agg:
        raise ValueError("ни один измеряемый параметр не сопоставлен -- нечего агрегировать")

    daily = d.groupby([c.object_id, "date"], as_index=False).agg(agg)

    if c.t_outdoor and c.t_outdoor in daily:
        win = max(1, round(cfg.outdoor_window_hours / 24))
        daily["t_outdoor_avg"] = (daily.groupby(c.object_id)[c.t_outdoor]
                                  .transform(lambda s: s.rolling(win, min_periods=1).mean()))
    return daily


# ------------------------------------------------------------ дискретизация
def discretize(daily: pd.DataFrame, passports: dict[str, Passport],
               cfg: Config) -> pd.DataFrame:
    """Суточные показатели -> пять категориальных входов сети.

    Возвращает `object_id`, `date` и колонки `pressure`, `temperature`,
    `flow`, `calibration`, `age` в терминах `network_spec.STATES`. Пропуск
    остаётся пропуском.
    """
    c = cfg.columns
    out = daily[[c.object_id, "date"]].copy()
    out.columns = ["object_id", "date"]

    # --- температура: отклонение от графика -------------------------------
    if c.t_supply and c.t_supply in daily and "t_outdoor_avg" in daily:
        expected = cfg.curve.expected(daily["t_outdoor_avg"].to_numpy())
        rel = np.abs(daily[c.t_supply].to_numpy() - expected) / np.where(expected == 0, np.nan, expected)
        temp = np.where(np.isnan(rel), None,
                        np.where(rel * 100 <= cfg.curve.tolerance_pct, "norma", "otklonenie"))
        out["temperature"] = temp
    else:
        out["temperature"] = None

    # --- давление ----------------------------------------------------------
    if c.pressure and c.pressure in daily:
        vals = daily[c.pressure]
        flags = pd.Series(pd.NA, index=daily.index, dtype="object")
        for oid, idx in daily.groupby(c.object_id).groups.items():
            rng = getattr(passports.get(oid), "pressure_range", None)
            if rng is not None:
                lo, hi = rng
                sub = vals.loc[idx]
                flags.loc[idx] = np.where(sub.isna(), pd.NA,
                                          np.where(sub.between(lo, hi), "norma", "otklonenie"))
            else:
                r = _robust_flag(vals.loc[idx], cfg.flow_mad_k, cfg.min_history_days)
                flags.loc[idx] = r.map({"norma": "norma", "high": "otklonenie",
                                        "low": "otklonenie"}).where(r.notna())
        out["pressure"] = flags.where(flags.notna(), None)
    else:
        out["pressure"] = None

    # --- расход -----------------------------------------------------------
    if c.volume and c.volume in daily:
        vals = daily[c.volume]
        flags = pd.Series(pd.NA, index=daily.index, dtype="object")
        for oid, idx in daily.groupby(c.object_id).groups.items():
            rng = getattr(passports.get(oid), "volume_range", None)
            sub = vals.loc[idx]
            if rng is not None:
                lo, hi = rng
                flags.loc[idx] = np.where(
                    sub.isna(), pd.NA,
                    np.where(sub < lo, "anomal_low",
                             np.where(sub > hi, "anomal_high", "norma")))
            else:
                r = _robust_flag(sub, cfg.flow_mad_k, cfg.min_history_days)
                flags.loc[idx] = r.map({"norma": "norma", "high": "anomal_high",
                                        "low": "anomal_low"}).where(r.notna())
        out["flow"] = flags.where(flags.notna(), None)
    else:
        out["flow"] = None

    # --- поверка и срок эксплуатации из паспорта ---------------------------
    out["calibration"] = [
        _calibration_state(passports.get(oid), d)
        for oid, d in zip(out["object_id"], out["date"])
    ]
    out["age"] = [
        _age_state(passports.get(oid), d)
        for oid, d in zip(out["object_id"], out["date"])
    ]
    return out


def _calibration_state(p: Passport | None, on_date: pd.Timestamp) -> str | None:
    if p is None or p.last_verification is None or p.verification_interval_months is None:
        return None
    due = p.last_verification + pd.DateOffset(months=p.verification_interval_months)
    days_left = (due - on_date).days
    if days_left < 0:
        return "istek"
    if days_left <= 90:            # квартал -- горизонт планирования выездов
        return "priblizhaetsya"
    return "v_norme"


def _age_state(p: Passport | None, on_date: pd.Timestamp) -> str | None:
    if p is None or p.commissioned is None:
        return None
    years = (on_date - p.commissioned).days / 365.25
    if years < 3:
        return "lt_3"
    if years <= 7:
        return "3_7"
    return "gt_7"


# ---------------------------------------------------------------- отчёт
def coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Полнота признаков: сколько строк пригодно для вывода.

    Полезнее общего числа строк: сеть принимает частичное свидетельство,
    поэтому важно, каких именно признаков не хватает.
    """
    cols = ["pressure", "temperature", "flow", "calibration", "age"]
    rows = [{"признак": c,
             "заполнено": int(features[c].notna().sum()),
             "доля": round(float(features[c].notna().mean()), 4)} for c in cols]
    rows.append({"признак": "все пять сразу",
                 "заполнено": int(features[cols].notna().all(axis=1).sum()),
                 "доля": round(float(features[cols].notna().all(axis=1).mean()), 4)})
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, passports: dict[str, Passport],
        cfg: Config | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Полный проход: проверки -> агрегация -> дискретизация."""
    cfg = cfg or Config()
    problems = validate(df, cfg)
    return discretize(to_daily(df, cfg), passports, cfg), problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="выгрузка архивов (CSV из Postgres или экспорта АРМ)")
    ap.add_argument("--passports", default=None,
                    help="CSV с паспортами: object_id, last_verification, "
                         "verification_interval_months, commissioned")
    ap.add_argument("--out", default=None, help="куда сохранить признаки")
    a = ap.parse_args()

    raw = pd.read_csv(a.csv)
    pas: dict[str, Passport] = {}
    if a.passports:
        pdf = pd.read_csv(a.passports)
        for _, r in pdf.iterrows():
            pas[str(r["object_id"])] = Passport(
                object_id=str(r["object_id"]),
                last_verification=pd.to_datetime(r.get("last_verification"), errors="coerce"),
                verification_interval_months=(int(r["verification_interval_months"])
                                              if pd.notna(r.get("verification_interval_months"))
                                              else None),
                commissioned=pd.to_datetime(r.get("commissioned"), errors="coerce"))

    feats, problems = run(raw, pas)
    if problems:
        print("ЗАМЕЧАНИЯ ПО ВЫГРУЗКЕ:")
        for p in problems:
            print(f"  - {p}")
    print(f"\nстрок на входе: {len(raw)}, объект-суток на выходе: {len(feats)}")
    print("\nПОЛНОТА ПРИЗНАКОВ")
    print(coverage(feats).to_string(index=False))
    if a.out:
        feats.to_csv(a.out, index=False)
        print(f"\nсохранено -> {a.out}")
