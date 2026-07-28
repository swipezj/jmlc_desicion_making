"""
Обучение структуры: восстановит ли алгоритм граф из §3.2 по одним данным?

Реализовано на numpy, без pgmpy. Причина не идеологическая: pgmpy сейчас
активно переименовывает API (`StructureScore` -> `structure_score`,
`MaximumLikelihoodEstimator` -> `DiscreteMLE` и т.д.), и завязывать на него
эксперимент, который должен воспроизводиться через год, не хочется.
Жадный поиск по BIC -- сотня строк, и она полностью под контролем.

BIC для дискретной БС раскладывается по узлам:
    BIC = Σ_i [ Σ_j Σ_k N_ijk · log(N_ijk / N_ij) ] - (log N / 2) · Σ_i q_i(r_i - 1)
где q_i -- число конфигураций родителей узла i, r_i -- число его состояний.
Разложимость и даёт эффективность: при добавлении дуги пересчитывается
слагаемое одного узла, а не весь граф.

ЧТО ЭТОТ ЭКСПЕРИМЕНТ ПОКАЗЫВАЕТ И ЧТО НЕТ. Данные синтетические, порождены
экспертной сетью, поэтому истинная структура известна -- в этом весь смысл:
можно измерить, СКОЛЬКО ДАННЫХ нужно, чтобы алгоритм её нашёл. Это верхняя
оценка удачи: на реальных данных со скрытыми переменными и шумом будет
хуже. Кроме того, по наблюдательным данным структура восстановима лишь с
точностью до класса Маркова-эквивалентности: направление дуги определяется
только там, где есть v-структура. Поэтому ниже отдельно считаются метрики
по скелету (неориентированному) и по направленным дугам.

Запуск: python -m src.structure_learning
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.model import BayesNet
from src.network_spec import EDGES, PARENTS, STATES

# Должно быть НЕ МЕНЬШЕ максимума по экспертному графу, иначе истинная
# структура недостижима для поиска и полнота по скелету упирается в
# искусственный потолок. Сейчас максимум 4 (device_cond и maintenance).
MAX_PARENTS = max(4, max(len(v) for v in PARENTS.values()))


class BICScorer:
    """Разложимый по узлам BIC с кэшем локальных оценок."""

    def __init__(self, df: pd.DataFrame, states: dict[str, list[str]] = STATES):
        self.states = states
        self.n = len(df)
        self.codes = {v: df[v].map({s: i for i, s in enumerate(states[v])}).to_numpy()
                      for v in states}
        self._local = lru_cache(maxsize=None)(self._local_uncached)

    def _local_uncached(self, node: str, parents: tuple[str, ...]) -> float:
        r = len(self.states[node])
        shape = tuple(len(self.states[p]) for p in parents) + (r,)
        counts = np.zeros(shape)
        np.add.at(counts, tuple(self.codes[p] for p in parents) + (self.codes[node],), 1.0)
        n_ij = counts.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.where(counts > 0, counts * np.log(np.divide(counts, n_ij,
                          out=np.ones_like(counts), where=n_ij > 0)), 0.0).sum()
        q = int(np.prod([len(self.states[p]) for p in parents])) if parents else 1
        return float(ll) - 0.5 * np.log(self.n) * q * (r - 1)

    def local(self, node: str, parents) -> float:
        return self._local(node, tuple(sorted(parents)))

    def total(self, parents: dict[str, tuple[str, ...]]) -> float:
        return sum(self.local(v, parents.get(v, ())) for v in self.states)


def _creates_cycle(parents: dict[str, set], child: str, new_parent: str) -> bool:
    """Появится ли цикл, если добавить дугу new_parent -> child."""
    stack, seen = [new_parent], set()
    while stack:
        v = stack.pop()
        if v == child:
            return True
        if v in seen:
            continue
        seen.add(v)
        stack.extend(parents.get(v, ()))
    return False


def hill_climb(df: pd.DataFrame, max_parents: int = MAX_PARENTS,
               start: dict[str, tuple[str, ...]] | None = None,
               max_iter: int = 500) -> tuple[dict[str, tuple[str, ...]], dict]:
    """Жадный поиск: add / remove / reverse, пока BIC растёт."""
    scorer = BICScorer(df)
    nodes = list(STATES)
    parents: dict[str, set] = {v: set(start.get(v, ())) if start else set() for v in nodes}
    history = []

    for step in range(max_iter):
        best = (1e-8, None)  # (прирост BIC, операция)
        for child in nodes:
            cur_child = scorer.local(child, parents[child])
            for other in nodes:
                if other == child:
                    continue
                if other in parents[child]:
                    # remove
                    d = scorer.local(child, parents[child] - {other}) - cur_child
                    if d > best[0]:
                        best = (d, ("remove", other, child))
                    # reverse: снять other->child, поставить child->other
                    if not _creates_cycle({k: v - {other} if k == child else v
                                           for k, v in parents.items()}, other, child) \
                            and len(parents[other]) < max_parents:
                        d = (scorer.local(child, parents[child] - {other}) - cur_child
                             + scorer.local(other, parents[other] | {child})
                             - scorer.local(other, parents[other]))
                        if d > best[0]:
                            best = (d, ("reverse", other, child))
                elif len(parents[child]) < max_parents and \
                        not _creates_cycle(parents, child, other):
                    d = scorer.local(child, parents[child] | {other}) - cur_child
                    if d > best[0]:
                        best = (d, ("add", other, child))
        if best[1] is None:
            break
        op, a, b = best[1]
        if op == "add":
            parents[b].add(a)
        elif op == "remove":
            parents[b].discard(a)
        else:
            parents[b].discard(a)
            parents[a].add(b)
        history.append({"шаг": step + 1, "операция": op, "дуга": f"{a}->{b}",
                        "ΔBIC": round(best[0], 2)})

    final = {v: tuple(sorted(p)) for v, p in parents.items() if p}
    return final, {"bic": scorer.total(final), "history": history, "scorer": scorer}


# ---------------------------------------------------------------------------
def _edge_set(parents: dict[str, tuple[str, ...]]) -> set[tuple[str, str]]:
    return {(p, c) for c, ps in parents.items() for p in ps}


def compare_to_expert(learned: dict[str, tuple[str, ...]]) -> dict:
    """Метрики по направленным дугам и отдельно по скелету."""
    exp, got = set(EDGES), _edge_set(learned)
    skel_e = {frozenset(e) for e in exp}
    skel_g = {frozenset(e) for e in got}

    def prf(true: set, pred: set) -> tuple[float, float, float]:
        tp = len(true & pred)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(true) if true else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    dp, dr, df1 = prf(exp, got)
    sp, sr, sf1 = prf(skel_e, skel_g)
    return {
        "дуг_найдено": len(got),
        "скелет_F1": round(sf1, 3), "скелет_полнота": round(sr, 3),
        "скелет_точность": round(sp, 3),
        "направл_F1": round(df1, 3),
        "пропущено": sorted(f"{a}->{b}" for a, b in exp - got
                            if frozenset((a, b)) not in skel_g),
        "перевёрнуто": sorted(f"{a}->{b}" for a, b in exp - got
                              if frozenset((a, b)) in skel_g),
        "лишние": sorted(f"{a}->{b}" for a, b in got
                         if frozenset((a, b)) not in skel_e),
    }


def sample_size_experiment(sizes=(200, 1000, 5000, 20000), seed: int = 42,
                           net: BayesNet | None = None) -> pd.DataFrame:
    net = net or BayesNet.from_spec()
    rng = np.random.default_rng(seed)
    big = net.sample(max(sizes), rng)
    rows = []
    for n in sizes:
        df = big.iloc[:n]
        learned, info = hill_climb(df)
        cmp = compare_to_expert(learned)
        expert_bic = BICScorer(df).total({c: tuple(sorted(ps)) for c, ps in
                                          _parents_from_edges().items()})
        rows.append({
            "n": n, **{k: cmp[k] for k in ["дуг_найдено", "скелет_F1", "скелет_полнота",
                                           "скелет_точность", "направл_F1"]},
            "BIC_обучен": round(info["bic"], 1), "BIC_эксперт": round(expert_bic, 1),
        })
    return pd.DataFrame(rows)


def _parents_from_edges() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {v: [] for v in STATES}
    for a, b in EDGES:
        out[b].append(a)
    return {v: tuple(sorted(p)) for v, p in out.items() if p}


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    net = BayesNet.from_spec()

    print("=" * 78)
    print("ВОССТАНОВЛЕНИЕ СТРУКТУРЫ ПО ДАННЫМ (жадный поиск по BIC)")
    print("=" * 78)
    print(sample_size_experiment(net=net).to_string(index=False))

    print("\nПодробно при n = 5000:")
    df = net.sample(5000, np.random.default_rng(42))
    learned, info = hill_climb(df)
    cmp = compare_to_expert(learned)
    for k in ["пропущено", "перевёрнуто", "лишние"]:
        print(f"  {k:12s}: {', '.join(cmp[k]) if cmp[k] else '--'}")

    print("""
КАК ЭТО ЧИТАТЬ. «Перевёрнутые» дуги -- не ошибка алгоритма: направление,
не входящее ни в одну v-структуру, по наблюдательным данным неразличимо,
такие графы задают в точности одно и то же распределение. Поэтому основная
метрика здесь -- F1 по скелету, а направленная F1 приводится справочно.
Если BIC обученной сети выше экспертной, это тоже не «алгоритм умнее
эксперта»: BIC вознаграждает подгонку под конкретную выборку, тогда как
экспертный граф несёт причинную интерпретацию, ради которой сеть и
строилась, -- она нужна для решающего слоя (src/decision.py), а
статистически эквивалентный граф с перевёрнутыми дугами её теряет.""")

    print("\nПроверка узла temperature (до пересмотра структуры он был изолирован):")
    t_edges = [e for e in _edge_set(learned) if "temperature" in e]
    if t_edges:
        print(f"  дуг с temperature найдено: {len(t_edges)}"
              f" ({', '.join(f'{a}->{b}' for a, b in t_edges)})")
        print("  Направление здесь читать не нужно -- см. оговорку о марковской"
              " эквивалентности выше; значимо то, что дуги вообще есть.")
    else:
        print("  дуг с temperature не найдено. Это НЕ подтверждение изоляции узла:"
              "\n  восстановленные экспертом эффекты температуры слабые"
              " (см. src/sensitivity.py),\n  и жадный поиск по BIC на такой"
              " выборке их может просто не увидеть.")
