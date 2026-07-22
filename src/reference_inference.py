"""
Точный вывод методом полного перебора (brute-force enumeration) --
референсная реализация без зависимости от pgmpy.

Зачем она нужна, если есть pgmpy (src/network.py)? Две независимые
реализации одной и той же сети -- это способ проверить, что структура и
CPT в network_spec.py заданы непротиворечиво, не полагаясь на то, что
сторонняя библиотека "точно" всё сделает правильно. Плюс этот модуль
работает без единой внешней зависимости (только numpy), поэтому им можно
пользоваться даже там, где pgmpy не установлен.

Для сети такого масштаба (11 узлов, 2-3 состояния каждый, полное
пространство состояний ~79 тыс. комбинаций) полный перебор мгновенен --
дерево сочленений тут не требуется по соображениям производительности,
но именно его использует HUGIN в оригинальной работе (см. §3.1 диплома).
"""
from __future__ import annotations
import itertools
import numpy as np

from src.network_spec import (
    STATES, PRIOR, CPT_DEVICE, CPT_PIPE, CPT_RELIABILITY, CPT_FAILURE,
    CPT_MAINT, cpt_anomaly, ROOTS,
)

VARS = list(STATES.keys())


def _idx(var: str, label: str) -> int:
    return STATES[var].index(label)


def joint_prob(a: dict[str, str]) -> float:
    """P(полное присвоение всем 11 переменным) по факторизации сети."""
    p = 1.0
    for v in ROOTS:
        p *= PRIOR[v][_idx(v, a[v])]
    p *= CPT_DEVICE[(a["age"], a["calibration"])][_idx("device_cond", a["device_cond"])]
    p *= CPT_PIPE[(a["pressure"], a["flow"])][_idx("pipe_cond", a["pipe_cond"])]
    p *= CPT_RELIABILITY[(a["device_cond"], a["flow"])][_idx("reliability", a["reliability"])]
    p *= CPT_FAILURE[(a["device_cond"], a["pipe_cond"])][_idx("failure_prob", a["failure_prob"])]
    p *= CPT_MAINT[(a["pipe_cond"], a["failure_prob"])][_idx("maintenance", a["maintenance"])]
    p *= cpt_anomaly(a["pipe_cond"], a["reliability"], a["flow"])[_idx("anomaly", a["anomaly"])]
    return p


def posterior(evidence: dict[str, str], targets: list[str] | None = None) -> dict[str, np.ndarray]:
    """Точный апостериорный вывод P(targets | evidence) перебором.

    evidence: словарь {переменная: состояние}, любое подмножество из 11 узлов.
    targets: какие узлы вернуть (по умолчанию -- все, кроме тех, что в evidence).
    """
    if targets is None:
        targets = [v for v in VARS if v not in evidence]
    free_vars = [v for v in VARS if v not in evidence]
    totals = {t: np.zeros(len(STATES[t])) for t in targets}
    z = 0.0
    for combo in itertools.product(*[STATES[v] for v in free_vars]):
        a = dict(evidence)
        a.update(dict(zip(free_vars, combo)))
        p = joint_prob(a)
        z += p
        for t in targets:
            totals[t][_idx(t, a[t])] += p
    if z == 0:
        raise ValueError("Свидетельство имеет нулевую вероятность при заданных CPT")
    return {t: totals[t] / z for t in targets}


def explain(evidence: dict[str, str], targets: list[str]) -> str:
    """Человекочитаемая распечатка апостериорных распределений."""
    post = posterior(evidence, targets)
    lines = [f"Свидетельство: {evidence}"]
    for t in targets:
        dist = ", ".join(f"{s}={p:.2f}" for s, p in zip(STATES[t], post[t]))
        lines.append(f"  {t}: {dist}")
    return "\n".join(lines)


if __name__ == "__main__":
    targets = ["failure_prob", "maintenance", "anomaly"]
    print("Априори (без свидетельств):")
    print(explain({}, targets))
