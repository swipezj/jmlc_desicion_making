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

from src.network_spec import (
    STATES, PRIOR, CPT_DEVICE, CPT_PIPE, CPT_RELIABILITY, CPT_FAILURE,
    CPT_MAINT, cpt_anomaly, ROOTS,
)


def _sample(states: list[str], probs: list[float], rng: np.random.Generator) -> str:
    return states[rng.choice(len(states), p=probs)]


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    """Сгенерировать n синтетических наблюдений (ancestral sampling)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        a = {v: _sample(STATES[v], PRIOR[v], rng) for v in ROOTS}
        a["device_cond"] = _sample(STATES["device_cond"],
                                    CPT_DEVICE[(a["age"], a["calibration"])], rng)
        a["pipe_cond"] = _sample(STATES["pipe_cond"],
                                  CPT_PIPE[(a["pressure"], a["flow"])], rng)
        a["reliability"] = _sample(STATES["reliability"],
                                    CPT_RELIABILITY[(a["device_cond"], a["flow"])], rng)
        a["failure_prob"] = _sample(STATES["failure_prob"],
                                     CPT_FAILURE[(a["device_cond"], a["pipe_cond"])], rng)
        a["maintenance"] = _sample(STATES["maintenance"],
                                    CPT_MAINT[(a["pipe_cond"], a["failure_prob"])], rng)
        a["anomaly"] = _sample(STATES["anomaly"],
                                cpt_anomaly(a["pipe_cond"], a["reliability"], a["flow"]), rng)
        rows.append(a)
    cols = list(STATES.keys())
    return pd.DataFrame(rows, columns=cols)


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
