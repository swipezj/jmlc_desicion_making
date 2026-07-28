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
    STATES, PRIOR, ROOTS,
    CPT_DEVICE_BASE, CPT_PIPE_BASE, CPT_MAINT_BASE, CPT_FAILURE,
    MULT_TEMP_DEVICE, MULT_PRESSURE_DEVICE, MULT_TEMP_PIPE,
    MULT_DEVICE_MAINT, MULT_RELIABILITY_MAINT,
    SEVERITY_RELIABILITY, RELIABILITY_ALPHA, RELIABILITY_GOOD, RELIABILITY_BAD,
    FLOW_WEIGHT, DELTA_FLOW_ANOMALY,
)

# ЭТОТ МОДУЛЬ НАМЕРЕННО НЕ ИМПОРТИРУЕТ ГОТОВЫЕ CPT_DEVICE / CPT_PIPE /
# CPT_RELIABILITY / CPT_MAINT / CPT_ANOMALY. Он берёт только базовые таблицы
# и коэффициенты и собирает строки заново -- иначе это была бы проверка
# движка вывода при общем источнике CPT, а не проверка самой сборки CPT.
# Тест test_engine_matches_reference_oracle сверяет обе ветки на всех 108
# комбинациях входных свидетельств.

VARS = list(STATES.keys())


def _idx(var: str, label: str) -> int:
    return STATES[var].index(label)


def _norm(v: np.ndarray) -> np.ndarray:
    return v / v.sum()


def _row_device(a: dict[str, str]) -> np.ndarray:
    v = np.array(CPT_DEVICE_BASE[(a["age"], a["calibration"])], dtype=float)
    if a["temperature"] == "otklonenie":
        v = _norm(v * np.array(MULT_TEMP_DEVICE))
    if a["pressure"] == "otklonenie":
        v = _norm(v * np.array(MULT_PRESSURE_DEVICE))
    return _norm(v)


def _row_pipe(a: dict[str, str]) -> np.ndarray:
    v = np.array(CPT_PIPE_BASE[(a["pressure"], a["flow"])], dtype=float)
    if a["temperature"] == "otklonenie":
        v = _norm(v * np.array(MULT_TEMP_PIPE))
    return _norm(v)


def _row_reliability(a: dict[str, str]) -> np.ndarray:
    s = [SEVERITY_RELIABILITY["flow"][a["flow"]],
         SEVERITY_RELIABILITY["calibration"][a["calibration"]],
         SEVERITY_RELIABILITY["age"][a["age"]]]
    sev = min(1.0, max(s) + RELIABILITY_ALPHA * (sum(s) - max(s)))
    return _norm(RELIABILITY_GOOD + (RELIABILITY_BAD - RELIABILITY_GOOD) * sev)


def _row_maintenance(a: dict[str, str]) -> np.ndarray:
    v = np.array(CPT_MAINT_BASE[(a["pipe_cond"], a["failure_prob"])], dtype=float)
    v = _norm(v * np.array(MULT_DEVICE_MAINT[a["device_cond"]]))
    v = _norm(v * np.array(MULT_RELIABILITY_MAINT[a["reliability"]]))
    return _norm(v)


def _row_anomaly(a: dict[str, str]) -> np.ndarray:
    v = {"норма":                  np.array([0.90, 0.07, 0.03]),
         "нарушена_герметичность": np.array([0.10, 0.85, 0.05]),
         "засор":                  np.array([0.75, 0.05, 0.20])}[a["pipe_cond"]].copy()
    v = v + FLOW_WEIGHT * np.array(DELTA_FLOW_ANOMALY[a["flow"]])
    if a["reliability"] == "недостоверная":
        v = v + np.array([0.35, -0.25, -0.10])
    elif a["reliability"] == "сомнительная":
        v = v + np.array([0.12, -0.08, -0.04])
    return _norm(np.clip(v, 0.01, None))


def joint_prob(a: dict[str, str]) -> float:
    """P(полное присвоение всем 11 переменным) по факторизации сети."""
    p = 1.0
    for v in ROOTS:
        p *= PRIOR[v][_idx(v, a[v])]
    p *= _row_device(a)[_idx("device_cond", a["device_cond"])]
    p *= _row_pipe(a)[_idx("pipe_cond", a["pipe_cond"])]
    p *= _row_reliability(a)[_idx("reliability", a["reliability"])]
    p *= CPT_FAILURE[(a["device_cond"], a["pipe_cond"], a["flow"])][
        _idx("failure_prob", a["failure_prob"])]
    p *= _row_maintenance(a)[_idx("maintenance", a["maintenance"])]
    p *= _row_anomaly(a)[_idx("anomaly", a["anomaly"])]
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
