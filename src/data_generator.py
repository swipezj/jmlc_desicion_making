"""
Генератор СИНТЕТИЧЕСКИХ данных методом ancestral sampling по сети из
network_spec.py.

Зачем: реальные данные пилота ООО «Спецмастер» (32 объекта, апрель-май
2026) -- собственность конкретной компании и не публикуются в открытом
репозитории. Этот генератор создаёт данные, статистически согласованные
с теми же CPT, что описаны в дипломе, -- на них можно безопасно
демонстрировать обучение параметров/структуры и сравнение с ML-моделями.

ВАЖНО: любые метрики, посчитанные на этих данных, -- ИЛЛЮСТРАТИВНЫЕ.
Реальные результаты пилота (-75% ложных выездов и т.д.) взяты из диплома
и никогда не подменяются числами, посчитанными здесь.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.network_spec import STATES, PRIOR, PARENTS, CPT_TABLES, ROOTS


def _sample(states: list[str], probs: list[float], rng: np.random.Generator) -> str:
    return states[rng.choice(len(states), p=probs)]


def _topological_order() -> list[str]:
    order, pending = list(ROOTS), [v for v in STATES if v not in ROOTS]
    while pending:
        ready = [v for v in pending if all(p in order for p in PARENTS[v])]
        if not ready:
            raise ValueError(f"В графе цикл или висячий родитель: {pending}")
        order += ready
        pending = [v for v in pending if v not in order]
    return order


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    """Сгенерировать n синтетических наблюдений (ancestral sampling).

    Порядок узлов и наборы родителей берутся из network_spec, а не
    прописаны здесь: после пересмотра структуры захардкоженный список
    вызовов пришлось бы править в третьем месте подряд.
    """
    rng = np.random.default_rng(seed)
    order = _topological_order()
    rows = []
    for _ in range(n):
        a: dict[str, str] = {}
        for v in order:
            probs = (PRIOR[v] if v in ROOTS
                     else CPT_TABLES[v][tuple(a[p] for p in PARENTS[v])])
            a[v] = _sample(STATES[v], probs, rng)
        rows.append(a)
    return pd.DataFrame(rows, columns=list(STATES.keys()))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5000, help="число наблюдений")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/synthetic/synthetic_observations.csv")
    args = p.parse_args()

    df = generate(args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Сгенерировано {len(df)} синтетических наблюдений -> {args.out}")
    print(df.head())
