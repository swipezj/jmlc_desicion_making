"""
Пересчёт PRIOR по выгрузке из АИС/ЛЭРС.

ЗАЧЕМ. `network_spec.PRIOR` -- это маргинальные частоты входных узлов по
парку. Сейчас там допущение для симуляции, а не измерение (см. README,
«Происхождение чисел»). Как только появится анонимизированная выгрузка,
числа заменяются измеренными, и скрипт готовит их вместе с проверками,
без которых замена опасна.

ЧТО ЗАМЕНА МЕНЯЕТ, А ЧТО НЕТ. Апостериорные в сценариях Табл. 3.6 не
изменятся вообще: там наблюдаются все пять входов, и априорные сокращаются
(проверено численно -- сдвиг порядка 1e-15). Изменятся только величины,
усредняющие по парку: базовая частота P(ТО = срочное), ценность информации,
все таблицы решающего слоя и синтетические выборки. Именно поэтому в
`src/decision.py` стоит оговорка о базовой частоте: сейчас модель завышает
долю тяжёлых объектов относительно пилота.

Запуск:
    python -m scripts.fit_priors выгрузка.csv
    python -m scripts.fit_priors выгрузка.csv --object-col nomer_uzla
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from src.network_spec import PRIOR, ROOTS, RUSSIAN_NAME, STATES

MIN_PER_CELL = 30       # ниже этого частота категории считается ненадёжной


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Уилсона: корректен при малых k, в отличие от нормального."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def audit(df: pd.DataFrame) -> list[str]:
    """Проверки, которые надо пройти ДО того, как числа попадут в модель."""
    problems = []
    for v in ROOTS:
        if v not in df.columns:
            problems.append(f"нет колонки '{v}' ({RUSSIAN_NAME[v]})")
            continue
        col = df[v]
        n_na = int(col.isna().sum())
        if n_na:
            problems.append(f"'{v}': {n_na} пропусков ({n_na / len(df):.1%}) -- "
                            "выясните, случайные ли они: если прибор не опрашивается "
                            "именно потому, что сломан, отбрасывать строки нельзя")
        unknown = sorted(set(col.dropna().unique()) - set(STATES[v]))
        if unknown:
            problems.append(f"'{v}': неизвестные категории {unknown}; "
                            f"ожидались {STATES[v]}")
    return problems


def estimate(df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    rows = []
    for v in ROOTS:
        col = df[v].dropna()
        n = len(col)
        counts = np.array([int((col == s).sum()) for s in STATES[v]], dtype=float)
        probs = (counts + alpha) / (counts + alpha).sum()
        for s, k, p, old in zip(STATES[v], counts, probs, PRIOR[v]):
            lo, hi = wilson(int(k), n)
            rows.append({
                "узел": v, "состояние": s, "наблюдений": int(k),
                "частота": round(float(p), 4),
                "ДИ_нижн": round(lo, 4), "ДИ_верх": round(hi, 4),
                "было_в_модели": old,
                "мало_данных": int(k) < MIN_PER_CELL,
            })
    return pd.DataFrame(rows)


def render(est: pd.DataFrame) -> str:
    out = ["PRIOR: dict[str, list[float]] = {"]
    for v in ROOTS:
        vals = est.loc[est["узел"] == v, "частота"].tolist()
        vals = [round(x / sum(vals), 4) for x in vals]
        out.append(f'    "{v}":'.ljust(20) + f" {vals},")
    out.append("}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="выгрузка с колонками pressure, temperature, flow, calibration, age")
    ap.add_argument("--object-col", default=None,
                    help="колонка с идентификатором узла учёта; если задана, частоты "
                         "считаются и по строкам, и по объектам")
    ap.add_argument("--alpha", type=float, default=1.0, help="сглаживание Лапласа")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    print(f"строк: {len(df)}")

    problems = audit(df)
    if problems:
        print("\nПРОБЛЕМЫ (исправьте до замены PRIOR):")
        for p in problems:
            print(f"  - {p}")
        if any("нет колонки" in p or "неизвестные категории" in p for p in problems):
            return 1

    est = estimate(df, a.alpha)
    print("\n" + est.to_string(index=False))

    thin = est[est["мало_данных"]]
    if len(thin):
        print(f"\nВНИМАНИЕ: {len(thin)} категорий набрали меньше {MIN_PER_CELL} наблюдений.")
        print("Их частоты держатся на сглаживании, а не на данных -- ширина ДИ показывает,")
        print("насколько. Такие состояния лучше оставить экспертными, чем подменить шумом.")

    if a.object_col and a.object_col in df.columns:
        # Мода по объекту здесь не годится: давление, температура и расход
        # меняются во времени, и мода схлопнула бы их к преобладающему
        # состоянию, занизив долю отклонений. Корректный способ получить
        # маргинальное распределение по ПАРКУ из несбалансированной панели --
        # взвесить строки обратно числу наблюдений по объекту, чтобы каждый
        # узел учёта вносил одинаковый вклад.
        w = 1.0 / df.groupby(a.object_col)[a.object_col].transform("size")
        n_obj = df[a.object_col].nunique()
        print(f"\nТо же, взвешенно по объектам ({n_obj} узлов учёта, "
              "каждый вносит одинаковый вклад):")
        rows = []
        for v in ROOTS:
            m = df[v].notna()
            tot = float(w[m].sum())
            vals = [float(w[m & (df[v] == s)].sum()) for s in STATES[v]]
            probs = [(x + a.alpha) / (tot + a.alpha * len(vals)) for x in vals]
            for s, p, old in zip(STATES[v], probs, PRIOR[v]):
                rows.append({"узел": v, "состояние": s,
                             "частота_по_объектам": round(p, 4),
                             "было_в_модели": old})
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nЕсли эта оценка заметно расходится с построчной -- панель")
        print("несбалансирована (часть узлов опрашивается чаще). Для PRIOR как")
        print("характеристики ПАРКА берите взвешенную, для симуляции потока")
        print("наблюдений -- построчную.")

    print("\n" + "=" * 70)
    print("Готовый блок для src/network_spec.py:")
    print("=" * 70)
    print(render(est))
    print("""
После замены обязательно перепрогнать: src.decision (базовая частота и все
таблицы стратегий), src.sensitivity (ценность информации), src.baselines и
src.structure_learning (они порождают данные из PRIOR). Сценарии Табл. 3.6
и scripts/verify_reference не изменятся -- там наблюдаются все входы.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
