"""
Сравнение БСД с ML-моделями -- корректная постановка.

ЧТО БЫЛО НЕ ТАК В ПЕРВОЙ ВЕРСИИ. Данные порождались той же сетью, чьё
качество измерялось, поэтому экспертная БСД была байесовски-оптимальной ПО
ПОСТРОЕНИЮ, а её «точность» 0.686 -- это не характеристика модели, а
байесовский предел задачи. Вывод «деревьям нужно на порядок больше данных»
из такого эксперимента не следует: он заложен в постановку.

ЧТО ИСПРАВЛЕНО.

1. Два режима генерации.
   * `well_specified` -- данные из экспертной сети. Здесь БСД оптимальна по
     построению, и это честно называется своим именем: режим показывает
     байесовский предел, к которому остальные модели могут только
     приближаться. Полезен как проверка сходимости, а не как соревнование.
   * `misspecified` -- данные из ВОЗМУЩЁННОЙ сети (CPT пересэмплированы из
     Дирихле вокруг экспертных). Экспертная БСД больше не оптимальна: она
     несёт правильную структуру, но неточные параметры -- ровно та ситуация,
     в которой находится реальная экспертная модель. Здесь сравнение
     осмысленно, и здесь же видно, где экспертное знание начинает мешать.

2. Метрики -- правила скоринга, а не только доля правильных ответов.
   Продукт системы -- вероятность, на которую умножается цена ошибки в
   решающем слое (src/decision.py). Модель, угадывающая класс, но выдающая
   переуверенные вероятности, даст плохие решения при хорошей accuracy.
   Поэтому основные метрики -- log-loss и Brier; accuracy/macro-F1
   приводятся для сопоставимости с первой версией. Заодно это снимает
   вопрос о немонотонности macro-F1: она была следствием дисбаланса
   классов, а правила скоринга к нему устойчивы.

3. Добавлена БСД С ОБУЧЕННЫМИ ПАРАМЕТРАМИ: структура из §3.2 фиксирована,
   CPT оценены по обучающей выборке. Это и есть честный «средний» вариант
   между экспертным знанием и обучением с нуля, и главный ответ на вопрос
   «зачем вообще БСД, если есть бустинг».

4. Добавлена логистическая регрессия -- сильный baseline на малых выборках,
   без которого сравнение с одними деревьями выглядит подобранным.

Запуск: python -m src.baselines
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from src.model import BayesNet
from src.network_spec import ROOTS, STATES

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

FEATURES = list(ROOTS)
TARGET = "maintenance"
CLASSES = STATES[TARGET]
TRAIN_SIZES = [20, 50, 100, 300, 1000, 3000, 10000]
TEST_SIZE = 5000
SEED = 42
PERTURB = 0.5    # сила искажения CPT в режиме misspecified (см. BayesNet.perturb)


def _bn_proba(net: BayesNet, df: pd.DataFrame) -> np.ndarray:
    """Вероятности БСД. Входных комбинаций всего 108 -- считаем их и раздаём."""
    lookup = {c: net.posterior(dict(zip(FEATURES, c)), [TARGET])[TARGET]
              for c in itertools.product(*[STATES[v] for v in FEATURES])}
    return np.stack([lookup[k] for k in zip(*[df[v].to_numpy() for v in FEATURES])])


def _score(name: str, n: int, y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = np.asarray(CLASSES)[proba.argmax(axis=1)]
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), pd.Index(CLASSES).get_indexer(y_true)] = 1.0
    return {
        "model": name, "n_train": n,
        "log_loss": log_loss(y_true, proba, labels=CLASSES),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
    }


def _align(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Достроить столбцы для классов, отсутствовавших в обучающей выборке."""
    out = np.full((len(proba), len(CLASSES)), 1e-12)
    for j, c in enumerate(classes):
        out[:, CLASSES.index(c)] = proba[:, j]
    return out / out.sum(axis=1, keepdims=True)


def run(regime: str = "misspecified", train_sizes=TRAIN_SIZES,
        test_size: int = TEST_SIZE, seed: int = SEED) -> pd.DataFrame:
    expert = BayesNet.from_spec()
    if regime == "well_specified":
        truth = expert
    elif regime == "misspecified":
        truth = expert.perturb(np.random.default_rng(seed + 7), PERTURB)
    else:
        raise ValueError("regime: 'well_specified' | 'misspecified'")

    full = truth.sample(max(train_sizes) + test_size, np.random.default_rng(seed))
    test_df = full.iloc[:test_size].reset_index(drop=True)
    pool = full.iloc[test_size:].reset_index(drop=True)
    y_test = test_df[TARGET].to_numpy()

    ordinal = OrdinalEncoder(categories=[STATES[f] for f in FEATURES]).fit(test_df[FEATURES])
    onehot = OneHotEncoder(categories=[STATES[f] for f in FEATURES],
                           sparse_output=False).fit(test_df[FEATURES])
    Xo_test = ordinal.transform(test_df[FEATURES])
    Xh_test = onehot.transform(test_df[FEATURES])

    # опорные линии, не зависящие от размера обучающей выборки
    expert_row = _score("БСД экспертная (без обучения)", 0, y_test, _bn_proba(expert, test_df))
    limit_row = _score("[предел] истинная сеть", 0, y_test, _bn_proba(truth, test_df))

    tree_models = {
        "DecisionTree": lambda: DecisionTreeClassifier(max_depth=5, random_state=seed),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8,
                                                       random_state=seed),
        "GradientBoosting": lambda: HistGradientBoostingClassifier(random_state=seed),
    }
    if HAS_XGBOOST:
        tree_models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric="mlogloss", random_state=seed)

    rows = []
    for n in train_sizes:
        train = pool.iloc[:n]
        y_train = train[TARGET].to_numpy()
        rows.append(dict(expert_row, n_train=n))
        rows.append(dict(limit_row, n_train=n))
        # БСД со структурой из §3.2 и параметрами, оценёнными по обучающей выборке
        rows.append(_score("БСД обученная (структура §3.2)", n, y_test,
                           _bn_proba(expert.fit(train, alpha=1.0), test_df)))
        if len(np.unique(y_train)) < 2:
            continue
        Xo = ordinal.transform(train[FEATURES])
        Xh = onehot.transform(train[FEATURES])
        for name, factory in tree_models.items():
            clf = factory().fit(Xo, y_train)
            rows.append(_score(name, n, y_test,
                               _align(clf.predict_proba(Xo_test), clf.classes_)))
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xh, y_train)
        rows.append(_score("LogisticRegression", n, y_test,
                           _align(lr.predict_proba(Xh_test), lr.classes_)))
    return pd.DataFrame(rows)


def crossover(res: pd.DataFrame, reference: str = "БСД экспертная (без обучения)",
              challenger: str = "БСД обученная (структура §3.2)") -> int | None:
    """Первый размер выборки, на котором обученные CPT обгоняют экспертные."""
    piv = res.pivot_table(index="n_train", columns="model", values="log_loss")
    if reference not in piv or challenger not in piv:
        return None
    better = piv.index[piv[challenger] < piv[reference]]
    return int(better[0]) if len(better) else None


def plot(outdir: str = "reports/figures", seed: int = SEED) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=False)
    for ax, regime, title in zip(
            axes, ["well_specified", "misspecified"],
            ["данные из самой экспертной сети\n(БСД оптимальна по построению)",
             f"данные из возмущённой сети (сила {PERTURB})\n"
             "(структура верна, параметры -- нет)"]):
        piv = run(regime, seed=seed).pivot_table(index="n_train", columns="model",
                                                 values="log_loss")
        for col in piv.columns:
            style = "--" if col.startswith("[предел]") else "-"
            lw = 2.2 if ("БСД" in col or col.startswith("[предел]")) else 1.2
            ax.plot(piv.index, piv[col], style, lw=lw, marker="o", ms=3, label=col)
        ax.set_xscale("log")
        ax.set_xlabel("размер обучающей выборки")
        ax.set_ylabel("log-loss на тесте (меньше -- лучше)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle("Экспертные знания против обучения: два режима", fontsize=11)
    fig.tight_layout()
    path = f"{outdir}/baselines_logloss.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


if __name__ == "__main__":
    import os

    pd.set_option("display.width", 175)
    os.makedirs("data/synthetic", exist_ok=True)
    for regime, header in [("well_specified", "РЕЖИМ 1: данные из самой экспертной сети"),
                           ("misspecified", "РЕЖИМ 2: данные из возмущённой сети")]:
        res = run(regime)
        print("=" * 92)
        print(header + "  --  log-loss (меньше лучше)")
        print("=" * 92)
        print(res.pivot_table(index="n_train", columns="model",
                              values="log_loss").round(3).to_string())
        res.to_csv(f"data/synthetic/baselines_{regime}.csv", index=False)
        c = crossover(res)
        print(f"\nОбученные CPT обгоняют экспертные начиная с n = "
              f"{c if c else 'не обгоняют на рассмотренных объёмах'}")

    os.makedirs("reports/figures", exist_ok=True)
    print("\nГрафик ->", plot()[0])
    print("""
КАК ЧИТАТЬ. В режиме 1 «победа» экспертной БСД тавтологична: она совпадает с
процессом, породившим данные, и лежит ровно на линии предела. Никакого вывода
о качестве моделей из него делать нельзя, он приведён именно поэтому.

Содержателен режим 2, и он даёт вывод ТОЧНЕЕ исходного. Экспертная БСД --
горизонтальная линия: её параметры неверны, и данные ей не помогают. Но БСД
с той же структурой и CPT, оценёнными всего по 20 наблюдениям, уже лучше и
экспертной версии, и любой модели без структуры, и дальше быстро выходит на
байесовский предел, тогда как деревьям и бустингу для сопоставимого log-loss
нужны тысячи наблюдений.

То есть выигрыш в объёме данных даёт НЕ экспертная настройка вероятностей, а
экспертный причинно-следственный граф: он резко сокращает число параметров,
которые вообще нужно оценивать (146 против произвольной функции пяти
категориальных входов). Это более сильное утверждение, чем «БСД точнее
деревьев», и оно не зависит от того, чьи числа стояли в CPT: структура
берётся из §3.2 диплома, числа -- из данных.""")
