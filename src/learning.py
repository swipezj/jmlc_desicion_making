"""
Обучение параметров (CPT) по синтетическим данным -- MLE и Bayesian
estimation -- и сравнение с экспертными CPT из network_spec.py.

Отвечает на вопрос "а если бы мы не знали CPT заранее, смогла бы модель
их восстановить по данным?" -- отдельно от вопроса "правильно ли заданы
экспертные CPT" (это уже проверяет scripts/verify_reference.py).

ВАЖНО: часть комментариев ниже я не мог исполнить сам (см. историю чата --
pgmpy недоступен в моей песочнице без интернета), поэтому сравнение
намеренно построено так, чтобы даже при сбое в блоке количественного
диффа (try/except ниже) ты всё равно увидел таблицы CPT напечатанными
бок к боку через встроенный __str__ pgmpy -- это самая надёжная часть.

Запуск: python -m src.learning
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.network import BayesianNetwork  # уже содержит фолбэк-импорт нужного класса
from src.network_spec import EDGES, STATES, PARENTS, CPT_DEVICE, CPT_PIPE
from src.data_generator import generate


def _bare_structure_model() -> "BayesianNetwork":
    """Модель ТОЛЬКО со структурой (без CPT) -- то, что нужно эстиматорам."""
    model = BayesianNetwork(EDGES)
    model.add_node("temperature")
    return model


def learn(df: pd.DataFrame, estimator_cls, **kwargs):
    """Обучить все CPT сети по данным df. Возвращает dict {var: TabularCPD}."""
    model = _bare_structure_model()
    estimator = estimator_cls(model, df)
    cpds = estimator.get_parameters(**kwargs)
    return {cpd.variable: cpd for cpd in cpds}


def print_side_by_side(node: str, expert_table: dict[tuple, list[float]], learned_cpd) -> None:
    """Печатает экспертную CPT (наша таблица) и обученную (объект pgmpy) --
    основной, надёжный способ увидеть разницу глазами."""
    parents = PARENTS[node]
    print(f"\n--- {node} | родители {parents} ---")
    print("Экспертная CPT (из диплома/реконструкции):")
    for combo, probs in expert_table.items():
        probs_str = ", ".join(f"{STATES[node][i]}={p:.2f}" for i, p in enumerate(probs))
        print(f"  {dict(zip(parents, combo))} -> {probs_str}")
    print("\nОбученная CPT (pgmpy, из синтетических данных):")
    print(learned_cpd)


def max_abs_diff(node: str, expert_table: dict[tuple, list[float]], learned_cpd) -> float | None:
    """Максимальное |learned - expert| по всем ячейкам CPT, или None при сбое
    сопоставления (например, если pgmpy вернул родителей в ином порядке)."""
    try:
        learned_parents = list(learned_cpd.variables[1:])
        expert_parents = list(PARENTS[node])
        if set(learned_parents) != set(expert_parents):
            return None
        values = np.asarray(learned_cpd.get_values())
        parent_states = [learned_cpd.state_names[p] for p in learned_parents]
        node_states = learned_cpd.state_names[node]
        diff = 0.0
        for col_i, combo in enumerate(itertools.product(*parent_states)):
            combo_as_dict = dict(zip(learned_parents, combo))
            expert_key = tuple(combo_as_dict[p] for p in PARENTS[node])
            expert_probs = expert_table[expert_key]
            for row_i, state in enumerate(node_states):
                diff = max(diff, abs(values[row_i, col_i] - expert_probs[STATES[node].index(state)]))
        return diff
    except Exception:
        return None


def try_quantitative_diff(node: str, expert_table: dict[tuple, list[float]], learned_cpd) -> None:
    """Best-effort численное сравнение, вывод в stdout (обёртка над max_abs_diff)."""
    diff = max_abs_diff(node, expert_table, learned_cpd)
    if diff is None:
        print("\n[диагностика] количественное сравнение не выполнено; "
              "смотри таблицы выше глазами -- они точно верны")
    else:
        print(f"\n[диагностика] максимальное |learned - expert| по всем ячейкам: {diff:.3f}")


def sample_efficiency_experiment(
    node: str = "device_cond",
    expert_table: dict[tuple, list[float]] | None = None,
    sizes=(20, 50, 100, 300, 1000, 5000),
    seed: int = 0,
) -> pd.DataFrame:
    """MLE vs Bayesian estimation: насколько точно каждый метод восстанавливает
    экспертную CPT для `node` в зависимости от объёма обучающей выборки.

    Ожидаемый эффект (см. README): при малых n Bayesian estimation (BDeu-приор)
    должен быть точнее MLE, потому что часть комбинаций родителей почти не
    встречается и MLE на них скатывается в 0/1; при больших n методы сходятся
    (что мы и увидели на n=5000 в первом прогоне: 0.051 vs 0.051).
    """
    if expert_table is None:
        expert_table = CPT_DEVICE
    rows = []
    for n in sizes:
        df = generate(n=n, seed=seed)
        for label, cls, kwargs in [
            ("MLE", MaximumLikelihoodEstimator, {}),
            ("Bayesian (BDeu)", BayesianEstimator, {"prior_type": "BDeu", "equivalent_sample_size": 10}),
        ]:
            try:
                cpds = learn(df, cls, **kwargs)
                diff = max_abs_diff(node, expert_table, cpds[node])
            except Exception as e:
                print(f"[ОШИБКА] {label} при n={n}: {e!r}")
                diff = None
            rows.append(dict(n=n, method=label, max_abs_diff=diff))
    return pd.DataFrame(rows)


def plot_sample_efficiency(df: pd.DataFrame, node: str, out_path: str = "reports/figures/mle_vs_bayesian.png") -> None:
    import os
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in df.groupby("method"):
        group = group.dropna(subset=["max_abs_diff"]).sort_values("n")
        ax.plot(group["n"], group["max_abs_diff"], marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Размер обучающей выборки (n)")
    ax.set_ylabel(f"Макс. |learned - expert| для {node}")
    ax.set_title("MLE vs Bayesian estimation: точность восстановления CPT от объёма данных")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено -> {out_path}")


def run_estimator(label: str, df: pd.DataFrame, estimator_cls, **kwargs) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    try:
        cpds = learn(df, estimator_cls, **kwargs)
    except Exception as e:
        print(f"[ОШИБКА] {label} не выполнен: {e!r}")
        print("(это не должно затронуть второй метод оценки ниже, если он идёт следующим)")
        return
    for node, expert_table in [("device_cond", CPT_DEVICE), ("pipe_cond", CPT_PIPE)]:
        try:
            print_side_by_side(node, expert_table, cpds[node])
            try_quantitative_diff(node, expert_table, cpds[node])
        except Exception as e:
            print(f"[ОШИБКА] вывод для {node} не удался: {e!r}")


if __name__ == "__main__":
    print("Генерирую синтетические данные для обучения параметров...")
    df = generate(n=5000, seed=1)

    run_estimator("Maximum Likelihood Estimation", df, MaximumLikelihoodEstimator)
    run_estimator("Bayesian Estimation (prior_type='BDeu')", df, BayesianEstimator,
                  prior_type="BDeu", equivalent_sample_size=10)

    print("\nЗачем сравнивать именно device_cond и pipe_cond: это единственные")
    print("два узла, для которых у нас есть ТОЧНЫЕ экспертные CPT (Табл. 3.3-3.4")
    print("диплома) -- сравнение с остальными узлами было бы сравнением с")
    print("моей же реконструкцией, что бессмысленно.")

    print("\n" + "=" * 70)
    print("MLE vs Bayesian estimation по размеру выборки (device_cond)")
    print("=" * 70)
    sweep = sample_efficiency_experiment(node="device_cond", expert_table=CPT_DEVICE)
    print(sweep.pivot(index="n", columns="method", values="max_abs_diff").round(3))
    plot_sample_efficiency(sweep, node="device_cond")
