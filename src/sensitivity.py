"""
Анализ чувствительности и ценность информации (VOI).

Три сценария Табл. 3.6 -- слишком тонкая проверка для сети с 146 свободными
параметрами: они покрывают 3 из 108 возможных комбинаций входов и молчат
о том, как модель ведёт себя в остальных 105. Здесь systematic-проверка:

1. `one_at_a_time` -- размах отклика каждого выхода на каждый вход
   (классическая «торнадо»-диаграмма для БС). Ловит и «мёртвые» входы,
   не влияющие ни на что, и входы с подозрительно сильным влиянием.

2. `mutual_information` -- сколько бит о выходе несёт каждый датчик.
   Метрика структурная: не зависит от того, какие решения принимаются.

3. `value_of_information` -- то же, но в деньгах: на сколько снижается
   ожидаемый байесовский риск (см. src/decision.py), если датчик доступен.
   Отвечает на прикладной вопрос «какой канал АИС стоит своих денег».

Корневые узлы в этой сети независимы между собой (у них нет родителей),
поэтому вероятность комбинации свидетельств -- произведение приоров, и все
величины ниже считаются точно, без сэмплирования.

Запуск: python -m src.sensitivity
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.decision import ACTIONS, Costs, NEED
from src.model import BayesNet
from src.network_spec import PRIOR, ROOTS, RUSSIAN_NAME, STATES

TARGETS = ["failure_prob", "maintenance", "anomaly"]
HIGH_STATE = {"failure_prob": "высокая", "maintenance": "срочное",
              "anomaly": "возможная_утечка"}


def _combo_prob(vars_: list[str], combo: tuple[str, ...]) -> float:
    return float(np.prod([PRIOR[v][STATES[v].index(s)] for v, s in zip(vars_, combo)]))


# ---------------------------------------------------------------------------
def one_at_a_time(net: BayesNet, targets: list[str] = TARGETS) -> pd.DataFrame:
    """P(target = «тревожное состояние» | наблюдён только один вход)."""
    rows = []
    for v in ROOTS:
        for s in STATES[v]:
            post = net.posterior({v: s}, targets)
            rows.append({"вход": v, "состояние": s,
                         **{t: float(post[t][STATES[t].index(HIGH_STATE[t])])
                            for t in targets}})
    return pd.DataFrame(rows)


def tornado(net: BayesNet, targets: list[str] = TARGETS) -> pd.DataFrame:
    """Размах отклика: max - min по состояниям входа. Ноль = вход ни на что не влияет."""
    oat = one_at_a_time(net, targets)
    agg = oat.groupby("вход")[targets].agg(lambda c: c.max() - c.min())
    agg["суммарный_размах"] = agg.sum(axis=1)
    return agg.sort_values("суммарный_размах", ascending=False).round(4)


# ---------------------------------------------------------------------------
def mutual_information(net: BayesNet, sensor: str, target: str) -> float:
    """I(sensor; target) в битах, точно по совместному распределению."""
    p_marg = net.posterior({}, [target])[target]
    mi = 0.0
    for i, s in enumerate(STATES[sensor]):
        ps = PRIOR[sensor][i]
        if ps == 0:
            continue
        p_cond = net.posterior({sensor: s}, [target])[target]
        for j in range(len(p_marg)):
            joint = ps * p_cond[j]
            if joint > 0 and p_marg[j] > 0:
                mi += joint * np.log2(p_cond[j] / p_marg[j])
    # взаимная информация неотрицательна; отрицательные значения порядка 1e-17 --
    # это шум суммирования float, а не содержательный результат
    return float(max(mi, 0.0))


def information_table(net: BayesNet, targets: list[str] = TARGETS) -> pd.DataFrame:
    rows = [{"вход": v, "название": RUSSIAN_NAME[v],
             **{t: round(mutual_information(net, v, t), 4) for t in targets}}
            for v in ROOTS]
    df = pd.DataFrame(rows)
    df["итого_бит"] = df[targets].sum(axis=1).round(4)
    return df.sort_values("итого_бит", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
def expected_risk(net: BayesNet, observed: list[str], costs: Costs) -> float:
    """Ожидаемый байесовский риск, если наблюдаются только узлы `observed`.

    Риск = E_свидетельство [ min_a E_состояние [ L(a, состояние) | свидетельство ] ].
    Чем больше наблюдений, тем ниже риск: информация не может навредить
    байесовскому решателю -- этим свойством и тестируется модуль.
    """
    L = costs.matrix()
    if not observed:
        belief = net.posterior({}, [NEED])[NEED]
        return float((L @ belief).min())
    total = 0.0
    for combo in itertools.product(*[STATES[v] for v in observed]):
        p = _combo_prob(observed, combo)
        if p == 0:
            continue
        belief = net.posterior(dict(zip(observed, combo)), [NEED])[NEED]
        total += p * float((L @ belief).min())
    return total


def value_of_information(net: BayesNet, costs: Costs = Costs()) -> pd.DataFrame:
    """Два взгляда на ценность датчика, оба в у.е. на объект.

    `в_одиночку`     -- снижение риска, если доступен ТОЛЬКО этот датчик;
    `вклад_сверх_остальных` -- снижение риска от его добавления к остальным
                       четырём (leave-one-out). Второй столбец и есть ответ
                       на вопрос «что мы потеряем, если этот канал отключить».
    """
    r_none = expected_risk(net, [], costs)
    r_all = expected_risk(net, list(ROOTS), costs)
    rows = []
    for v in ROOTS:
        others = [u for u in ROOTS if u != v]
        rows.append({
            "вход": v,
            "название": RUSSIAN_NAME[v],
            # VOI не может быть отрицательной (информация не вредит байесовскому
            # решателю); clip убирает шум суммирования порядка 1e-17
            "в_одиночку": round(max(r_none - expected_risk(net, [v], costs), 0.0), 4),
            "вклад_сверх_остальных": round(max(expected_risk(net, others, costs) - r_all, 0.0), 4),
        })
    df = pd.DataFrame(rows).sort_values("вклад_сверх_остальных", ascending=False)
    df.attrs["risk_none"] = r_none
    df.attrs["risk_all"] = r_all
    return df.reset_index(drop=True)


def inert_inputs(net: BayesNet, tol: float = 1e-9) -> list[str]:
    """Входы, не влияющие ни на один выход. Должно быть пусто в исправной модели."""
    t = tornado(net)
    return sorted(t.index[t["суммарный_размах"] <= tol].tolist())


# ---------------------------------------------------------------------------
def plot(outdir: str = "reports/figures", net: BayesNet | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = net or BayesNet.from_spec()
    t = tornado(net)
    voi = value_of_information(net).set_index("вход").loc[t.index]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for tgt, color in zip(TARGETS, ["#4c72b0", "#dd8452", "#55a868"]):
        ax[0].barh(y, t[tgt], left=left, height=.6, color=color, label=tgt)
        left = left + t[tgt].to_numpy()
    ax[0].set_yticks(y, t.index)
    ax[0].set_xlabel("размах P(тревожное состояние) по состояниям входа")
    ax[0].set_title("Чувствительность выходов к каждому входу")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="x", alpha=.3)

    ax[1].barh(y, voi["вклад_сверх_остальных"], height=.6, color="#c44e52")
    ax[1].set_yticks(y, voi.index)
    ax[1].set_xlabel("снижение ожидаемых потерь, у.е. на объект")
    ax[1].set_title("Ценность датчика сверх остальных четырёх (VOI)")
    ax[1].grid(axis="x", alpha=.3)

    fig.tight_layout()
    path = f"{outdir}/sensitivity_voi.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


if __name__ == "__main__":
    import os

    net = BayesNet.from_spec()
    pd.set_option("display.width", 150)

    print("=" * 78)
    print("РАЗМАХ ОТКЛИКА (торнадо): max-min P(тревожное состояние) по входу")
    print("=" * 78)
    print(tornado(net).to_string())

    print("\n" + "=" * 78)
    print("ВЗАИМНАЯ ИНФОРМАЦИЯ ВХОД -> ВЫХОД, бит")
    print("=" * 78)
    print(information_table(net).to_string(index=False))

    print("\n" + "=" * 78)
    print("ЦЕННОСТЬ ИНФОРМАЦИИ, у.е. на объект")
    print("=" * 78)
    voi = value_of_information(net)
    print(voi.to_string(index=False))
    print(f"\nриск без наблюдений: {voi.attrs['risk_none']:.4f}  ->  "
          f"со всеми пятью: {voi.attrs['risk_all']:.4f}  "
          f"(снижение {voi.attrs['risk_none'] - voi.attrs['risk_all']:.4f})")

    dead = inert_inputs(net)
    if dead:
        print("\nВНИМАНИЕ: входы, не влияющие НИ НА ЧТО: " + ", ".join(dead))
        print("Каждый такой узел -- канал АИС, который опрашивается впустую.")
        print("До пересмотра структуры сюда попадала температура; если узел")
        print("появился здесь снова, значит очередная правка CPT его обесточила.")
    else:
        print("\nМёртвых входов нет: каждый из пяти каналов АИС влияет на выводы.")

    os.makedirs("reports/figures", exist_ok=True)
    for p in plot(net=net):
        print(f"\nГрафик -> {p}")
