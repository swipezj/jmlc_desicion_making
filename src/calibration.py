"""
Калибровка CPT_FAILURE по Табл. 3.6 и проверка структурной гипотезы.

ПРОБЛЕМА. Реконструированная сеть воспроизводит все три сценария Табл. 3.6
качественно, но в сценарии 3 P(отказ = высокая) выходит 0.37 против 0.70 в
дипломе. Разрыв в 0.33 -- слишком большой, чтобы списать на округление.

ГИПОТЕЗА. Посмотрим на числа диплома. В сценариях 2 и 3 состояние прибора
одинаково (срок > 7 лет, поверка истекла), а P(отказ) отличается вдвое:
0.52 против 0.70. Различаются они расходом (норма против аномально
высокого) и давлением. При этом в сценарии 3 трубопровод ВЕРОЯТНЕЕ исправен,
чем в сценарии 2, -- значит, разницу не объяснить через pipe_cond. Остаётся
предположить, что в исходной сети у `failure_prob` есть родитель, которого
нет в текстовом описании структуры §3.2: расход напрямую либо достоверность
показаний (недостоверные показания -- сами по себе симптом отказа прибора).

МЕТОД. Перебираем три варианта набора родителей failure_prob и для каждого
подбираем CPT, минимизируя
    L = Σ (p_модель - p_диплом)^2  +  λ · ||logit(CPT) - logit(CPT_эксперт)||^2
Второе слагаемое принципиально: 6 целевых чисел и до 81 свободного
параметра -- задача недоопределена, и без штрафа подойдёт любая гипотеза.
Штраф превращает вопрос «можно ли подогнать?» в «НАСКОЛЬКО СИЛЬНО придётся
исказить экспертную таблицу, чтобы подогнать?». Гипотеза, которой хватает
малого искажения, и есть правдоподобная.

Что в целевой функции: P(failure_prob=высокая) и P(maintenance=срочное) в
трёх сценариях -- 6 чисел. P(anomaly) от CPT_FAILURE не зависит вовсе
(anomaly не является потомком failure_prob), поэтому в подгонку не входит и
служит контрольной величиной: она не должна измениться.

Запуск: python -m src.calibration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.model import BayesNet
from src.network_spec import (
    CPT_DEVICE, CPT_MAINT, CPT_PIPE, CPT_RELIABILITY, SCENARIOS_TABLE_3_6,
    STATES, TARGET_HIGH_STATE, cpt_anomaly,
)

DEV, PIPE, REL, FLOW, FAIL = (STATES["device_cond"], STATES["pipe_cond"],
                              STATES["reliability"], STATES["flow"],
                              STATES["failure_prob"])
# Наборы родителей failure_prob, которые сравниваются.
HYPOTHESES = {
    "A: device, pipe (как в §3.2)": ("device_cond", "pipe_cond"),
    "B: device, pipe, flow": ("device_cond", "pipe_cond", "flow"),
    "C: device, pipe, reliability": ("device_cond", "pipe_cond", "reliability"),
}
AXES = ("device_cond", "pipe_cond", "reliability", "flow")  # порядок осей тензора


def _full_shape() -> tuple[int, ...]:
    return (len(DEV), len(PIPE), len(REL), len(FLOW), len(FAIL))


def expand(compact: np.ndarray, parents: tuple[str, ...]) -> np.ndarray:
    """Развернуть CPT с подмножеством родителей в полный тензор (d, pipe, rel, flow, f)."""
    shape = [len(STATES[a]) if a in parents else 1 for a in AXES] + [len(FAIL)]
    return np.broadcast_to(compact.reshape(shape), _full_shape())


def compact_shape(parents: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(len(STATES[a]) for a in AXES if a in parents) + (len(FAIL),)


# ---------------------------------------------------------------------------
# Тензоры, не зависящие от подгоняемой CPT, собираются один раз: внутри
# оптимизатора scenario_outputs вызывается десятки тысяч раз, и пересборка
# таблиц из словарей на каждом вызове была основной статьёй расходов.
_MAINT = np.stack([[CPT_MAINT[(p, x)] for x in FAIL] for p in PIPE])         # (pipe,fail,m)
_REL = {f: np.stack([CPT_RELIABILITY[(d, f)] for d in DEV]) for f in FLOW}   # (d,rel)
_ANOM = {f: np.stack([[cpt_anomaly(p, r, f) for r in REL] for p in PIPE])    # (pipe,rel,a)
         for f in FLOW}


def scenario_outputs(evidence: dict[str, str], failure_full: np.ndarray) -> dict[str, np.ndarray]:
    """Замкнутая формула для апостериорных выходов при ПОЛНОМ свидетельстве на входах.

    Когда наблюдены все пять корневых узлов, device_cond и pipe_cond условно
    независимы (общих родителей нет, общие потомки не наблюдаются), поэтому
        P(d, pipe, rel | e) = P(d | e) · P(pipe | e) · P(rel | d, flow)
    и всё считается сворачиванием четырёх маленьких тензоров вместо перебора
    79 тыс. комбинаций. Эквивалентность полному перебору проверяется в
    tests/test_calibration.py.
    """
    f = evidence["flow"]
    p_d = np.asarray(CPT_DEVICE[(evidence["age"], evidence["calibration"])])      # (d,)
    p_pipe = np.asarray(CPT_PIPE[(evidence["pressure"], f)])                      # (pipe,)
    p_rel = _REL[f]                                                               # (d, rel)
    fi = FLOW.index(f)
    fail = failure_full[:, :, :, fi, :]                                           # (d,pipe,rel,f)

    joint = np.einsum("d,p,dr->dpr", p_d, p_pipe, p_rel)                          # (d,pipe,rel)
    p_pipe_fail = np.einsum("dpr,dprf->pf", joint, fail)                          # (pipe, fail)
    p_pipe_rel = joint.sum(axis=0)                                                # (pipe, rel)
    return {
        "failure_prob": p_pipe_fail.sum(axis=0),
        "maintenance": np.einsum("pf,pfm->m", p_pipe_fail, _MAINT),
        "anomaly": np.einsum("pr,pra->a", p_pipe_rel, _ANOM[f]),
    }


def targets_vector(failure_full: np.ndarray) -> np.ndarray:
    """6 чисел: P(отказ=высокая) и P(ТО=срочное) в трёх сценариях."""
    out = []
    for spec in SCENARIOS_TABLE_3_6.values():
        res = scenario_outputs(spec["evidence"], failure_full)
        for t in ("failure_prob", "maintenance"):
            out.append(res[t][STATES[t].index(TARGET_HIGH_STATE[t])])
    return np.asarray(out)


THESIS = np.asarray([spec["thesis_posterior"][t]
                     for spec in SCENARIOS_TABLE_3_6.values()
                     for t in ("failure_prob", "maintenance")])


# ---------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fit(parents: tuple[str, ...], lam: float, expert_full: np.ndarray,
        seed: int = 0) -> dict:
    """Подгонка CPT_FAILURE с данным набором родителей и силой штрафа lam."""
    shape = compact_shape(parents)
    # старт -- экспертная таблица, усреднённая по осям, которых нет в гипотезе
    keep = [i for i, a in enumerate(AXES) if a in parents]
    drop = tuple(i for i, a in enumerate(AXES) if a not in parents)
    init = expert_full.mean(axis=drop) if drop else expert_full
    z0 = np.log(np.clip(init, 1e-6, None)).reshape(-1)
    z_ref = z0.copy()

    def objective(z: np.ndarray) -> float:
        cpt = _softmax(z.reshape(shape))
        resid = targets_vector(expand(cpt, parents)) - THESIS
        return float(resid @ resid + lam * ((z - z_ref) ** 2).mean())

    # L-BFGS-B с численным градиентом: целевая функция гладкая (softmax +
    # свёртки), поэтому квазиньютон сходится на порядок быстрее Powell.
    res = minimize(objective, z0, method="L-BFGS-B",
                   options=dict(maxiter=2000, maxfun=200000, ftol=1e-14, gtol=1e-10))
    cpt = _softmax(res.x.reshape(shape))
    full = expand(cpt, parents)
    pred = targets_vector(full)
    return {
        "cpt": cpt, "full": np.asarray(full), "pred": pred,
        "rmse": float(np.sqrt(((pred - THESIS) ** 2).mean())),
        "max_abs": float(np.abs(pred - THESIS).max()),
        "distortion": float(np.abs(np.asarray(full) - expert_full).mean()),
        "distortion_max": float(np.abs(np.asarray(full) - expert_full).max()),
    }


def monotonicity_report(full: np.ndarray) -> dict[str, bool]:
    """Осталась ли таблица качественно осмысленной после подгонки.

    Ожидания из §3.2: чем хуже прибор -- тем выше P(отказ); нарушение
    герметичности не должно снижать P(отказ) относительно нормы.
    """
    high = full[..., FAIL.index("высокая")]
    by_device = high.mean(axis=(1, 2, 3))
    by_pipe = high.mean(axis=(0, 2, 3))
    return {
        "P(отказ) растёт с деградацией прибора": bool(np.all(np.diff(by_device) > -1e-9)),
        "нарушение герметичности не снижает P(отказ)": bool(
            by_pipe[PIPE.index("нарушена_герметичность")] >= by_pipe[PIPE.index("норма")] - 1e-9),
    }


def compare_hypotheses(lams=(0.0, 0.01, 0.05, 0.2, 1.0)) -> pd.DataFrame:
    expert_full = expand(_expert_compact(), ("device_cond", "pipe_cond"))
    rows = []
    for name, parents in HYPOTHESES.items():
        for lam in lams:
            r = fit(parents, lam, expert_full)
            rows.append({"гипотеза": name, "λ": lam,
                         "RMSE_к_дипл": round(r["rmse"], 4),
                         "макс_ошибка": round(r["max_abs"], 4),
                         "искажение_ср": round(r["distortion"], 4),
                         "искажение_макс": round(r["distortion_max"], 4)})
    return pd.DataFrame(rows)


def _expert_compact() -> np.ndarray:
    from src.network_spec import CPT_FAILURE
    return np.stack([[CPT_FAILURE[(d, p)] for p in PIPE] for d in DEV])


def calibrated_network(hypothesis: str = "B: device, pipe, flow", lam: float = 0.05,
                       base: BayesNet | None = None) -> tuple[BayesNet, dict]:
    """Сеть с подогнанной CPT_FAILURE -- для сравнения выводов, не для замены.

    Возвращает (сеть, отчёт о подгонке). Экспертная сеть не мутируется:
    решения, посчитанные на калиброванной версии, должны публиковаться
    отдельно и с пометкой о происхождении таблицы.
    """
    base = base or BayesNet.from_spec()
    parents = HYPOTHESES[hypothesis]
    report = fit(parents, lam, expand(_expert_compact(), ("device_cond", "pipe_cond")))
    new_parents = dict(base.parents)
    new_parents["failure_prob"] = parents
    new_cpt = dict(base.cpt)
    new_cpt["failure_prob"] = report["cpt"]
    return BayesNet(base.states, new_parents, new_cpt), report


def plot(outdir: str = "reports/figures", lams=(0.0, 0.005, 0.01, 0.02, 0.05,
                                                0.1, 0.2, 0.5, 1.0, 2.0)) -> list[str]:
    """Парето: насколько сильно нужно исказить экспертную CPT ради согласия с Табл. 3.6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expert_full = expand(_expert_compact(), ("device_cond", "pipe_cond"))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for (name, parents), color in zip(HYPOTHESES.items(), ["#c44e52", "#4c72b0", "#55a868"]):
        xs, ys = [], []
        for lam in lams:
            r = fit(parents, lam, expert_full)
            xs.append(r["distortion"])
            ys.append(r["rmse"])
        ax.plot(xs, ys, "-o", ms=4, color=color, label=name)
    base_rmse = float(np.sqrt(((targets_vector(expert_full) - THESIS) ** 2).mean()))
    ax.axhline(base_rmse, ls="--", c="gray", lw=1)
    ax.annotate("исходная реконструкция", (0.002, base_rmse), fontsize=8,
                va="bottom", color="gray")
    ax.set_xlabel("среднее искажение экспертной CPT, |Δ вероятности|")
    ax.set_ylabel("RMSE относительно шести чисел Табл. 3.6")
    ax.set_title("Цена согласия с дипломом для трёх гипотез о родителях failure_prob")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    path = f"{outdir}/calibration_pareto.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    expert_full = expand(_expert_compact(), ("device_cond", "pipe_cond"))

    print("=" * 78)
    print("ИСХОДНОЕ СОСТОЯНИЕ: экспертная (реконструированная) CPT")
    print("=" * 78)
    pred = targets_vector(expert_full)
    labels = [f"{i + 1}:{t}" for i in range(3) for t in ("отказ", "ТО")]
    print(pd.DataFrame({"сценарий": labels, "модель": pred.round(3),
                        "диплом": THESIS, "Δ": (pred - THESIS).round(3)}).to_string(index=False))
    print(f"RMSE = {np.sqrt(((pred - THESIS) ** 2).mean()):.4f}")

    print("\n" + "=" * 78)
    print("СРАВНЕНИЕ СТРУКТУРНЫХ ГИПОТЕЗ (λ -- сила притяжения к экспертной CPT)")
    print("=" * 78)
    cmp = compare_hypotheses()
    print(cmp.to_string(index=False))

    print("\n" + "=" * 78)
    print("ЛУЧШАЯ ГИПОТЕЗА ПРИ УМЕРЕННОМ ШТРАФЕ (λ = 0.05)")
    print("=" * 78)
    sub = cmp[cmp["λ"] == 0.05].sort_values("RMSE_к_дипл")
    print(sub.to_string(index=False))
    best_name = sub.iloc[0]["гипотеза"]
    best = fit(HYPOTHESES[best_name], 0.05, expert_full)
    print(f"\nПобедитель: {best_name}")
    pred = best["pred"]
    print(pd.DataFrame({"сценарий": labels, "модель": pred.round(3),
                        "диплом": THESIS, "Δ": (pred - THESIS).round(3)}).to_string(index=False))
    print("\nКачественные проверки подогнанной таблицы:")
    for k, v in monotonicity_report(best["full"]).items():
        print(f"  [{'OK ' if v else 'НЕТ'}] {k}")

    print("""
ЧТО ИЗ ЭТОГО СЛЕДУЕТ, А ЧТО НЕТ. При λ = 0 под шесть чисел Табл. 3.6
подстраивается ЛЮБАЯ из трёх гипотез, включая исходную структуру §3.2:
шесть уравнений против 27-81 параметра -- задача недоопределена, и «удалось
подогнать» само по себе ничего не доказывает. Содержательна только цена
подгонки. Исходная структура достигает согласия, лишь переписав таблицу
почти целиком (среднее искажение ~0.4 при λ = 0 -- это уже не та CPT, что
следует из качественной логики §3.2). Гипотеза с дополнительным родителем
даёт то же согласие вдвое-втрое дешевле и доминирует по обеим осям сразу
(см. reports/figures/calibration_pareto.png).

Между гипотезами B (расход) и C (достоверность показаний) шести чисел не
хватает, чтобы уверенно выбрать: они идут вплотную. Это ожидаемо -- в
исходной сети reliability сама зависит от flow, поэтому оба варианта
описывают близкие зависимости. Разрешается это не подгонкой, а данными:
достаточно выгрузки из АИС с зафиксированными исходами отказов.

ВАЖНО. Подогнанная таблица НЕ подставляется в network_spec.py автоматически.
Она получена подгонкой под шесть чисел из текста диплома, а не выгружена из
исходного файла HUGIN, поэтому её статус -- «обоснованная гипотеза о
недостающей дуге», а не «восстановленный оригинал». В network_spec.py она
доступна отдельной константой с явной пометкой о происхождении; переключение
модели на неё -- сознательное действие, а не умолчание.""")
