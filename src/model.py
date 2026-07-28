"""
Обобщённый движок дискретной байесовской сети (numpy, без pgmpy).

Зачем ещё один модуль вывода, если есть `reference_inference.py`?
`reference_inference` намеренно захардкожен под конкретную сеть из диплома:
он читает именно те шесть CPT и служит независимым оракулом. Но чтобы
СРАВНИВАТЬ сети (экспертная vs обученная на данных vs искажённая), нужен
объект, у которого CPT -- это данные, а не код. Отсюда `BayesNet`.

Инвариант, который проверяется в tests/test_model.py:
    BayesNet.from_spec().posterior(e) == reference_inference.posterior(e)
для всех 108 комбинаций свидетельств на входных узлах. Пока он держится,
третий движок не размывает "единый источник правды", а усиливает проверку.

CPT хранится тензором формы (|pa_1|, ..., |pa_k|, |X|): последняя ось --
распределение узла, предыдущие -- конфигурация родителей.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.network_spec import STATES, PARENTS, ROOTS, PRIOR, CPT_TABLES


@dataclass
class BayesNet:
    """Дискретная БС с табличными CPT. Узлы хранятся в топологическом порядке."""

    states: dict[str, list[str]]
    parents: dict[str, tuple[str, ...]]
    cpt: dict[str, np.ndarray]
    order: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- служебное
    def __post_init__(self) -> None:
        if not self.order:
            self.order = _topological_order(self.states, self.parents)
        self._index = {v: {s: i for i, s in enumerate(ss)} for v, ss in self.states.items()}
        self.validate()

    def validate(self) -> None:
        """Формы тензоров и нормировка -- ловим опечатки в CPT сразу, а не в выводе."""
        for v in self.states:
            pa = self.parents.get(v, ())
            expected = tuple(len(self.states[p]) for p in pa) + (len(self.states[v]),)
            got = self.cpt[v].shape
            if got != expected:
                raise ValueError(f"CPT[{v}]: форма {got}, ожидалась {expected}")
            sums = self.cpt[v].sum(axis=-1)
            if not np.allclose(sums, 1.0, atol=1e-8):
                worst = float(np.abs(sums - 1.0).max())
                raise ValueError(f"CPT[{v}] не нормирована, макс. отклонение {worst:.2e}")
            if (self.cpt[v] < -1e-12).any():
                raise ValueError(f"CPT[{v}] содержит отрицательные вероятности")

    def copy(self) -> "BayesNet":
        return BayesNet(self.states, self.parents,
                        {v: t.copy() for v, t in self.cpt.items()}, list(self.order))

    def row(self, v: str, assignment: dict[str, str]) -> np.ndarray:
        """Строка CPT узла v при конфигурации его родителей из assignment."""
        pa = self.parents.get(v, ())
        if not pa:
            return self.cpt[v]
        return self.cpt[v][tuple(self._index[p][assignment[p]] for p in pa)]

    # ------------------------------------------------------------------- вывод
    def joint(self, assignment: dict[str, str]) -> float:
        p = 1.0
        for v in self.order:
            p *= self.row(v, assignment)[self._index[v][assignment[v]]]
        return p

    def posterior(self, evidence: dict[str, str],
                  targets: list[str] | None = None) -> dict[str, np.ndarray]:
        """Точный вывод P(targets | evidence) полным перебором свободных узлов."""
        unknown = set(evidence) - set(self.states)
        if unknown:
            raise KeyError(f"Неизвестные узлы в свидетельстве: {sorted(unknown)}")
        for v, s in evidence.items():
            if s not in self._index[v]:
                raise KeyError(f"У узла {v} нет состояния {s!r}")
        if targets is None:
            targets = [v for v in self.order if v not in evidence]
        free = [v for v in self.order if v not in evidence]
        totals = {t: np.zeros(len(self.states[t])) for t in targets}
        z = 0.0
        for combo in itertools.product(*[self.states[v] for v in free]):
            a = dict(evidence)
            a.update(zip(free, combo))
            p = self.joint(a)
            if p == 0.0:
                continue
            z += p
            for t in targets:
                totals[t][self._index[t][a[t]]] += p
        if z == 0:
            raise ValueError("Свидетельство имеет нулевую вероятность при заданных CPT")
        return {t: totals[t] / z for t in targets}

    # ---------------------------------------------------------------- генерация
    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Ancestral sampling. Векторизован по строкам: n может быть большим."""
        cols: dict[str, np.ndarray] = {}
        for v in self.order:
            pa = self.parents.get(v, ())
            probs = (np.repeat(self.cpt[v][None, :], n, axis=0) if not pa
                     else self.cpt[v][tuple(cols[p] for p in pa)])
            cum = probs.cumsum(axis=1)
            u = rng.random(n)[:, None]
            cols[v] = (u > cum).sum(axis=1).clip(0, len(self.states[v]) - 1)
        return pd.DataFrame({v: np.asarray(self.states[v], dtype=object)[cols[v]]
                             for v in self.states})

    # ----------------------------------------------------------------- обучение
    def fit(self, df: pd.DataFrame, alpha: float = 1.0) -> "BayesNet":
        """Оценка CPT по данным. alpha=0 -- MLE, alpha>0 -- сглаживание Дирихле.

        Возвращает НОВУЮ сеть с той же структурой: исходная не мутируется,
        чтобы экспертные CPT нельзя было затереть по неосторожности.
        """
        codes = {v: df[v].map(self._index[v]).to_numpy() for v in self.states}
        new: dict[str, np.ndarray] = {}
        for v in self.states:
            pa = self.parents.get(v, ())
            shape = tuple(len(self.states[p]) for p in pa) + (len(self.states[v]),)
            counts = np.zeros(shape) + alpha
            np.add.at(counts, tuple(codes[p] for p in pa) + (codes[v],), 1.0)
            total = counts.sum(axis=-1, keepdims=True)
            # конфигурация родителей, ни разу не встретившаяся при alpha=0,
            # даёт 0/0 -- заменяем равномерным, чтобы сеть осталась корректной
            uniform = np.full(shape, 1.0 / len(self.states[v]))
            new[v] = np.where(total > 0, counts / np.where(total > 0, total, 1.0), uniform)
        return BayesNet(self.states, self.parents, new, list(self.order))

    def perturb(self, rng: np.random.Generator, strength: float) -> "BayesNet":
        """Случайное искажение CPT: строка ~ Dirichlet(row * kappa).

        strength in [0, 1): 0 -- без искажений, ближе к 1 -- сильнее. Нужно,
        чтобы получить генеративный процесс, в котором экспертная сеть УЖЕ НЕ
        оптимальна (см. src/baselines.py, режим misspecified).
        """
        if not 0 <= strength < 1:
            raise ValueError("strength должен быть в [0, 1)")
        if strength == 0:
            return self.copy()
        # Меньше kappa -- шире разброс Дирихле. Шкала подобрана так, чтобы
        # strength=0.5 давал заметное, но не разрушительное искажение.
        kappa = (1.0 - strength) / strength * 10.0
        new = {}
        for v, table in self.cpt.items():
            flat = table.reshape(-1, table.shape[-1])
            drawn = np.stack([rng.dirichlet(np.maximum(r * kappa, 1e-3)) for r in flat])
            new[v] = drawn.reshape(table.shape)
        return BayesNet(self.states, self.parents, new, list(self.order))

    # ------------------------------------------------------------------ метрики
    def log_likelihood(self, df: pd.DataFrame) -> float:
        codes = {v: df[v].map(self._index[v]).to_numpy() for v in self.states}
        total = 0.0
        for v in self.states:
            pa = self.parents.get(v, ())
            probs = (self.cpt[v][codes[v]] if not pa
                     else self.cpt[v][tuple(codes[p] for p in pa) + (codes[v],)])
            total += float(np.log(np.maximum(probs, 1e-300)).sum())
        return total

    @property
    def n_free_params(self) -> int:
        return sum(int(np.prod([len(self.states[p]) for p in self.parents.get(v, ())] or [1]))
                   * (len(self.states[v]) - 1) for v in self.states)

    def bic(self, df: pd.DataFrame) -> float:
        return self.log_likelihood(df) - 0.5 * np.log(len(df)) * self.n_free_params

    @classmethod
    def from_spec(cls) -> "BayesNet":
        """Экспертная сеть из network_spec.py (см. README, 'Происхождение чисел')."""
        return cls(STATES, PARENTS, _spec_cpts())


# ---------------------------------------------------------------------------
def _topological_order(states: dict[str, list[str]],
                       parents: dict[str, tuple[str, ...]]) -> list[str]:
    order: list[str] = []
    pending = list(states)
    while pending:
        ready = [v for v in pending if all(p in order for p in parents.get(v, ()))]
        if not ready:
            raise ValueError(f"В графе цикл или висячий родитель: {pending}")
        order += ready
        pending = [v for v in pending if v not in order]
    return order


def _table(node: str) -> np.ndarray:
    """Словарную CPT из network_spec -> тензор (|pa_1|, ..., |pa_k|, |X|)."""
    pa = PARENTS[node]
    shape = tuple(len(STATES[p]) for p in pa) + (len(STATES[node]),)
    out = np.zeros(shape)
    for combo, probs in CPT_TABLES[node].items():
        out[tuple(STATES[p].index(s) for p, s in zip(pa, combo))] = probs
    return out


def _spec_cpts() -> dict[str, np.ndarray]:
    """Перенос словарных CPT из network_spec.py в тензорную форму.

    Ключи CPT_TABLES[node] уже упорядочены как PARENTS[node], поэтому
    отдельная лямбда на каждый узел больше не нужна: добавление родителя
    правится в одном месте -- в network_spec.py.
    """
    cpts = {v: np.asarray(PRIOR[v], dtype=float) for v in ROOTS}
    for node in CPT_TABLES:
        cpts[node] = _table(node)
    return cpts


if __name__ == "__main__":
    net = BayesNet.from_spec()
    print(f"Узлов: {len(net.states)}, свободных параметров: {net.n_free_params}")
    print("Топологический порядок:", " -> ".join(net.order))
    post = net.posterior({}, targets=["failure_prob", "maintenance", "anomaly"])
    for t, p in post.items():
        print(f"  априори {t:12s}", ", ".join(f"{s}={x:.3f}" for s, x in zip(STATES[t], p)))
