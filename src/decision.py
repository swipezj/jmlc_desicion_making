"""
Решающий слой: от вероятностей к диспетчерскому решению.

Чего не хватало до этого модуля. Сеть выдаёт P(maintenance | показания),
а Табл. 3.5 диплома переводит эти вероятности в действия жёстким правилом
("высокая -> срочный выезд"). Но правило `argmax` неявно предполагает,
что ошибки стоят одинаково: пропустить аварию так же плохо, как съездить
зря. На практике это не так, и весь эффект пилота (ложные выезды 6-8/мес
-> 1-2/мес при том, что пропущенных отказов стало НЕ больше) -- это
история именно про асимметрию цен ошибок, а не про точность классификации.

Здесь эта асимметрия сделана явной: матрица потерь + байесовское решение
   a*(e) = argmin_a  Σ_s P(s | e) · L(a, s)
Тогда «порог срочного выезда» перестаёт быть магической константой и
становится следствием одного управленческого параметра -- отношения цены
пропущенного отказа к цене выезда (`miss_to_visit`). Это та ручка, которую
руководитель предприятия может крутить осознанно.

ВАЖНО: все числа ниже -- синтетический прогон по сети из диплома, а не
измерения. Реальные результаты пилота (см. README) получены на 32 объектах
ООО «Спецмастер» и здесь не воспроизводятся и не заменяются.

Запуск: python -m src.decision
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model import BayesNet
from src.network_spec import STATES, ROOTS

# Действия диспетчера. Порядок фиксирован -- он же порядок строк матрицы потерь.
ACTIONS = ["не_выезжать", "плановый_выезд", "срочный_выезд"]
ACTION_LABEL = {
    "не_выезжать": "Плановый режим наблюдения",
    "плановый_выезд": "Включение объекта в план ТО",
    "срочный_выезд": "Немедленная диспетчеризация выезда",
}
# Состояние мира, относительно которого считаются потери, -- узел maintenance
# («какое обслуживание объекту в действительности требовалось»).
NEED = "maintenance"


@dataclass(frozen=True)
class Costs:
    """Цены в условных единицах: за 1.0 принята стоимость срочного выезда.

    urgent_visit    -- срочный выезд бригады (внеплановый маршрут, переработка)
    planned_visit   -- тот же объём работ, но совмещённый с плановым маршрутом
    miss_to_visit   -- ВО СКОЛЬКО РАЗ пропущенный срочный случай дороже выезда
                       (авария, спор с заказчиком, безучётное потребление)
    miss_planned    -- цена пропущенного планового ТО, в тех же единицах
    late_fraction   -- какая доля ущерба всё же реализуется, если на срочный
                       случай приехали плановым порядком (с задержкой)
    """

    urgent_visit: float = 1.0
    planned_visit: float = 0.4
    miss_to_visit: float = 5.0
    miss_planned: float = 1.5
    late_fraction: float = 0.5

    def matrix(self) -> np.ndarray:
        """L[действие, истинная_потребность], столбцы в порядке STATES[NEED]."""
        miss = self.miss_to_visit
        return np.array([
            # не_требуется       плановое            срочное
            [0.0, self.miss_planned, miss],
            [self.planned_visit, self.planned_visit,
             self.planned_visit + self.late_fraction * miss],
            [self.urgent_visit, self.urgent_visit, self.urgent_visit],
        ])


# ---------------------------------------------------------------------------
# Апостериорные распределения кэшируются: входных комбинаций всего
# 2*2*3*3*3 = 108, поэтому парк любого размера обсчитывается мгновенно.
# ---------------------------------------------------------------------------
def posterior_lookup(net: BayesNet, target: str = NEED) -> dict[tuple[str, ...], np.ndarray]:
    return {combo: net.posterior(dict(zip(ROOTS, combo)), [target])[target]
            for combo in itertools.product(*[STATES[v] for v in ROOTS])}


def expected_losses(belief: np.ndarray, costs: Costs) -> np.ndarray:
    """Ожидаемые потери каждого действия при апостериорном распределении belief."""
    return costs.matrix() @ belief


def bayes_action(belief: np.ndarray, costs: Costs) -> str:
    return ACTIONS[int(np.argmin(expected_losses(belief, costs)))]


def explain_decision(net: BayesNet, evidence: dict[str, str], costs: Costs) -> pd.DataFrame:
    """Разбор одного решения: во что обходится каждое действие и почему."""
    belief = net.posterior(evidence, [NEED])[NEED]
    losses = expected_losses(belief, costs)
    best = int(np.argmin(losses))
    return pd.DataFrame({
        "действие": [ACTION_LABEL[a] for a in ACTIONS],
        "ожидаемые_потери": losses.round(3),
        "выбор": ["<-- минимум" if i == best else "" for i in range(len(ACTIONS))],
    })


def breakeven_ratio(net: BayesNet, evidence: dict[str, str],
                    base: Costs = Costs(), lo: float = 0.0, hi: float = 200.0) -> dict[str, float]:
    """При каком `miss_to_visit` решение по данному объекту переключается.

    Возвращает границы интервалов по R для каждого действия. Это и есть
    содержательный ответ на вопрос «почему порог именно такой»: порога нет,
    есть цена ошибки, при которой выезд начинает окупаться.
    """
    grid = np.linspace(lo, hi, 2001)
    out: dict[str, float] = {}
    prev = None
    for r in grid:
        a = bayes_action(net.posterior(evidence, [NEED])[NEED],
                         Costs(base.urgent_visit, base.planned_visit, float(r),
                               base.miss_planned, base.late_fraction))
        if a != prev:
            out.setdefault(a, float(r))
            prev = a
    return out


# ---------------------------------------------------------------------------
# Стратегии
# ---------------------------------------------------------------------------
def policy_actions(df: pd.DataFrame, policy: str, net: BayesNet,
                   costs: Costs, lookup=None) -> np.ndarray:
    """Действия стратегии на парке объектов df (строки -- наблюдения из АИС)."""
    if policy == "reactive":
        # «До внедрения»: реакция на превышение порога расхода (§3.1 диплома)
        return np.where(df["flow"].to_numpy() == "anomal_high",
                        "срочный_выезд", "не_выезжать")
    lookup = lookup or posterior_lookup(net)
    keys = list(zip(*[df[v].to_numpy() for v in ROOTS]))
    beliefs = np.stack([lookup[k] for k in keys])
    if policy == "map":
        # Табл. 3.5 в лоб: действие = наиболее вероятная потребность
        return np.asarray(ACTIONS, dtype=object)[beliefs.argmax(axis=1)]
    if policy == "bayes":
        return np.asarray(ACTIONS, dtype=object)[(costs.matrix() @ beliefs.T).argmin(axis=0)]
    raise ValueError(f"Неизвестная стратегия: {policy}")


def evaluate(df: pd.DataFrame, actions: np.ndarray, costs: Costs,
             per: int = 100) -> dict[str, float]:
    """Метрики стратегии. Считаются по фактической потребности из симуляции."""
    need = df[NEED].to_numpy()
    L = costs.matrix()
    a_idx = pd.Index(ACTIONS).get_indexer(actions)
    s_idx = pd.Index(STATES[NEED]).get_indexer(need)
    n = len(df)
    scale = per / n
    return {
        "выездов": float(((actions != "не_выезжать").sum()) * scale),
        "срочных_выездов": float((actions == "срочный_выезд").sum() * scale),
        "ложных_срочных": float(((actions == "срочный_выезд") & (need == "не_требуется")).sum() * scale),
        "пропущено_срочных": float(((need == "срочное") & (actions == "не_выезжать")).sum() * scale),
        "средние_потери": float(L[a_idx, s_idx].mean()),
    }


def compare_policies(n: int = 20000, seed: int = 42, costs: Costs = Costs(),
                     net: BayesNet | None = None) -> pd.DataFrame:
    net = net or BayesNet.from_spec()
    df = net.sample(n, np.random.default_rng(seed))
    lookup = posterior_lookup(net)
    rows = []
    for name, label in [("reactive", "Реактивное правило (порог расхода)"),
                        ("map", "БСД + argmax (Табл. 3.5)"),
                        ("bayes", f"БСД + мин. ожид. потерь (R={costs.miss_to_visit:g})")]:
        acts = policy_actions(df, name, net, costs, lookup)
        rows.append({"стратегия": label, **evaluate(df, acts, costs)})
    return pd.DataFrame(rows)


def triage(df: pd.DataFrame, capacity: int, net: BayesNet, costs: Costs,
           lookup=None) -> np.ndarray:
    """Ранжирование заявок, когда бригада может сделать только `capacity` выездов.

    У ООО «Спецмастер» бригада одна, поэтому «выехать на всё, что выгодно»
    физически недостижимо -- реальная задача диспетчера не «ехать или нет»,
    а «на какие объекты ехать в первую очередь». Ценность выезда на объект --
    это снижение ожидаемых потерь относительно бездействия:
        v(e) = L_ожид(не_выезжать | e) - min(L_ожид(выезд | e))
    Объекты сортируются по v, первые `capacity` получают выезд.
    """
    lookup = lookup or posterior_lookup(net)
    beliefs = np.stack([lookup[k] for k in zip(*[df[v].to_numpy() for v in ROOTS])])
    losses = costs.matrix() @ beliefs.T           # (действие, объект)
    value = losses[0] - losses[1:].min(axis=0)    # выгода от лучшего выезда
    actions = np.full(len(df), "не_выезжать", dtype=object)
    chosen = np.argsort(-value)[:capacity]
    chosen = chosen[value[chosen] > 0]            # отрицательную выгоду не берём
    actions[chosen] = np.asarray(ACTIONS[1:], dtype=object)[losses[1:, chosen].argmin(axis=0)]
    return actions


def capacity_sweep(capacities=(2, 5, 10, 20, 40, 80), n: int = 20000, seed: int = 42,
                   costs: Costs = Costs(), net: BayesNet | None = None,
                   per: int = 100) -> pd.DataFrame:
    """Сколько пропущенных срочных случаев остаётся при разном лимите выездов.

    Сравниваются два способа потратить один и тот же бюджет выездов:
    реактивный (кого поймал порог расхода) и байесовский (кто выше по
    ожидаемой выгоде). Это самая близкая к пилоту постановка.
    """
    net = net or BayesNet.from_spec()
    df = net.sample(n, np.random.default_rng(seed))
    lookup = posterior_lookup(net)
    rng = np.random.default_rng(seed + 1)
    flagged = np.flatnonzero(df["flow"].to_numpy() == "anomal_high")
    rows = []
    for k in capacities:
        budget = int(round(k * n / per))
        bayes_acts = triage(df, budget, net, costs, lookup)
        # реактивная альтернатива: тот же бюджет, но приоритет случайный среди
        # сработавших по порогу -- у порогового правила нет ранжирования
        react = np.full(len(df), "не_выезжать", dtype=object)
        react[rng.permutation(flagged)[:budget]] = "срочный_выезд"
        rows.append({
            "лимит_выездов_на_100": k,
            "пропущено_срочных__реактивно": evaluate(df, react, costs, per)["пропущено_срочных"],
            "пропущено_срочных__БСД": evaluate(df, bayes_acts, costs, per)["пропущено_срочных"],
            "потери__реактивно": evaluate(df, react, costs, per)["средние_потери"],
            "потери__БСД": evaluate(df, bayes_acts, costs, per)["средние_потери"],
        })
    return pd.DataFrame(rows)


def cost_ratio_sweep(ratios=(1, 2, 3, 5, 8, 12, 20, 35, 60), n: int = 20000,
                     seed: int = 42, net: BayesNet | None = None) -> pd.DataFrame:
    """Как меняется поведение байесовской стратегии с ценой пропущенного отказа."""
    net = net or BayesNet.from_spec()
    df = net.sample(n, np.random.default_rng(seed))
    lookup = posterior_lookup(net)
    rows = []
    for r in ratios:
        c = Costs(miss_to_visit=float(r))
        acts = policy_actions(df, "bayes", net, c, lookup)
        m = evaluate(df, acts, c)
        rows.append({"R": r, **{k: m[k] for k in
                                ["выездов", "срочных_выездов", "ложных_срочных",
                                 "пропущено_срочных"]}})
    return pd.DataFrame(rows)


def plot(outdir: str = "reports/figures", n: int = 20000, seed: int = 42) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = BayesNet.from_spec()
    df = net.sample(n, np.random.default_rng(seed))
    lookup = posterior_lookup(net)

    ratios = np.geomspace(0.5, 80, 40)
    fp, miss = [], []
    for r in ratios:
        c = Costs(miss_to_visit=float(r))
        m = evaluate(df, policy_actions(df, "bayes", net, c, lookup), c)
        fp.append(m["ложных_срочных"])
        miss.append(m["пропущено_срочных"])

    base = Costs()
    pts = {}
    for name, label in [("reactive", "реактивное правило"), ("map", "argmax (Табл. 3.5)")]:
        m = evaluate(df, policy_actions(df, name, net, base, lookup), base)
        pts[label] = (m["ложных_срочных"], m["пропущено_срочных"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(fp, miss, "-o", ms=3, color="#1f77b4", label="БСД + мин. ожид. потерь")
    for i in (0, 12, 24, 39):
        ax[0].annotate(f"R={ratios[i]:.0f}", (fp[i], miss[i]),
                       textcoords="offset points", xytext=(6, 6), fontsize=8)
    for (label, (x, y)), mk in zip(pts.items(), ["s", "^"]):
        ax[0].plot(x, y, mk, ms=10, label=label)
    ax[0].set_xlabel("ложных срочных выездов на 100 объектов")
    ax[0].set_ylabel("пропущенных срочных случаев на 100 объектов")
    ax[0].set_title("Компромисс, которым управляет цена ошибки")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    caps = [1, 2, 3, 5, 8, 12, 20, 30, 45, 65, 90]
    sweep = capacity_sweep(caps, n=n, seed=seed, net=net)
    ax[1].plot(sweep["лимит_выездов_на_100"], sweep["пропущено_срочных__реактивно"],
               "-s", ms=4, label="реактивное правило")
    ax[1].plot(sweep["лимит_выездов_на_100"], sweep["пропущено_срочных__БСД"],
               "-o", ms=4, label="БСД, ранжирование по ожид. выгоде")
    ax[1].set_xlabel("лимит выездов бригады на 100 объектов")
    ax[1].set_ylabel("пропущенных срочных случаев на 100 объектов")
    ax[1].set_title("Один и тот же бюджет выездов, разное распределение")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    fig.suptitle("Решающий слой: синтетический прогон по сети из диплома "
                 "(не результаты пилота)", fontsize=10)
    fig.tight_layout()
    path = f"{outdir}/decision_tradeoff.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


if __name__ == "__main__":
    import os

    net = BayesNet.from_spec()
    pd.set_option("display.width", 150)

    print("=" * 78)
    print("СРАВНЕНИЕ СТРАТЕГИЙ (синтетический парк 20 000 наблюдений, на 100 объектов)")
    print("=" * 78)
    print(compare_policies(net=net).to_string(index=False))

    print("\n" + "=" * 78)
    print("ЧУВСТВИТЕЛЬНОСТЬ К ЦЕНЕ ОШИБКИ (байесовская стратегия)")
    print("=" * 78)
    print(cost_ratio_sweep(net=net).to_string(index=False))

    print("\n" + "=" * 78)
    print("ОГРАНИЧЕННАЯ БРИГАДА: как потратить один и тот же бюджет выездов")
    print("=" * 78)
    print(capacity_sweep(net=net).round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("РАЗБОР ОДНОГО РЕШЕНИЯ -- Сценарий 1 Табл. 3.6")
    print("=" * 78)
    ev = dict(pressure="otklonenie", temperature="norma", flow="anomal_high",
              calibration="priblizhaetsya", age="3_7")
    print(explain_decision(net, ev, Costs()).to_string(index=False))
    print("\nКакое решение оптимально при какой цене пропущенного отказа:")
    switch = breakeven_ratio(net, ev)
    bounds = list(switch.items()) + [(None, None)]
    for (a, lo), (_, hi) in zip(bounds[:-1], bounds[1:]):
        rng_s = f"R < {hi:.1f}" if hi is not None else f"R >= {lo:.1f}"
        print(f"  {rng_s:12s} ->  {ACTION_LABEL[a]}")

    os.makedirs("reports/figures", exist_ok=True)
    for p in plot():
        print(f"\nГрафик -> {p}")

    base_rate = net.posterior({}, [NEED])[NEED][STATES[NEED].index("срочное")]
    print(f"""
ОГОВОРКА О БАЗОВОЙ ЧАСТОТЕ. Нормировка «на 100 объектов» выбрана не условно:
в отопительный сезон парк предприятия -- около 120 объектов, поэтому цифры
в таблицах выше читаются практически как месячные абсолютные значения.

Осторожность нужна в другом. В синтетическом парке P(срочное) = {base_rate:.1%},
потому что априорные вероятности входных узлов (network_spec.PRIOR) -- это
допущение для симуляции, а не измеренные частоты. На пилоте срочных случаев
на объект было заметно меньше, то есть модель завышает долю «тяжёлых»
объектов. Следствие: абсолютные уровни выездов и пропусков в таблицах выше
реальных, тогда как СРАВНЕНИЕ стратегий между собой корректно -- все они
работают на одной и той же базе. Как только появится анонимизированная
выгрузка из АИС, PRIOR заменяется частотами из неё, и уровни становятся
сопоставимы с пилотом напрямую.""")
