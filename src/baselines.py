"""
Сравнение БСД с классификаторами (Decision Tree / Random Forest /
опционально XGBoost) на СИНТЕТИЧЕСКИХ данных.

ЧЕСТНОЕ ОГРАНИЧЕНИЕ ЭТОГО СРАВНЕНИЯ (прочитать перед тем, как цитировать
цифры где-либо): синтетические данные сгенерированы ИЗ ТОЙ ЖЕ сети,
предсказание которой мы сравниваем. Поэтому "точность БСД" здесь по
построению близка к байесовски-оптимальной -- это не честное соревнование
"чей алгоритм лучше в вакууме", а способ ПРОИЛЛЮСТРИРОВАТЬ конкретный,
содержательный аргумент из диплома (§3.1): при малом количестве размеченных
примеров экспертно-заданная БСД уже работает хорошо, тогда как деревьям/
ансамблям для сопоставимого качества нужно заметно больше данных. Ценность
здесь -- в ФОРМЕ кривой "точность vs. размер обучающей выборки", а не в
абсолютном значении точности БСД.

Запуск: python -m src.baselines
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.network_spec import STATES, ROOTS
from src.data_generator import generate
from src.reference_inference import posterior

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

FEATURES = list(ROOTS)          # pressure, temperature, flow, calibration, age
TARGET = "maintenance"          # 3-класс: не_требуется / плановое / срочное
TRAIN_SIZES = [20, 50, 100, 300, 1000, 3000]
TEST_SIZE = 2000
SEED = 42


def _encode(df: pd.DataFrame, encoder: OrdinalEncoder | None = None):
    if encoder is None:
        encoder = OrdinalEncoder(categories=[STATES[f] for f in FEATURES])
        encoder.fit(df[FEATURES])
    X = encoder.transform(df[FEATURES])
    y = df[TARGET].values
    return X, y, encoder


def bn_predict(df: pd.DataFrame) -> np.ndarray:
    """Предсказание БСД: argmax апостериорного распределения по свидетельству."""
    preds = []
    for _, row in df.iterrows():
        evidence = {f: row[f] for f in FEATURES}
        post = posterior(evidence, targets=[TARGET])[TARGET]
        preds.append(STATES[TARGET][int(np.argmax(post))])
    return np.array(preds)


def run(train_sizes: list[int] = TRAIN_SIZES, test_size: int = TEST_SIZE, seed: int = SEED) -> pd.DataFrame:
    full = generate(max(train_sizes) + test_size, seed=seed)
    test_df = full.iloc[:test_size].reset_index(drop=True)
    pool_df = full.iloc[test_size:].reset_index(drop=True)

    _, y_test, encoder = _encode(test_df)
    X_test, _, _ = _encode(test_df, encoder)

    bn_pred = bn_predict(test_df)
    bn_acc = accuracy_score(test_df[TARGET], bn_pred)
    bn_f1 = f1_score(test_df[TARGET], bn_pred, average="macro")

    models = {
        "DecisionTree": lambda: DecisionTreeClassifier(max_depth=5, random_state=seed),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed),
        "GradientBoosting": lambda: HistGradientBoostingClassifier(random_state=seed),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric="mlogloss",
            use_label_encoder=False, random_state=seed,
        )

    rows = []
    for n in train_sizes:
        train_df = pool_df.iloc[:n]
        X_train, y_train, _ = _encode(train_df, encoder)
        rows.append(dict(model="БСД (эталон, без обучения)", n_train=n, accuracy=bn_acc, macro_f1=bn_f1))
        for name, factory in models.items():
            clf = factory()
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            rows.append(dict(
                model=name, n_train=n,
                accuracy=accuracy_score(y_test, pred),
                macro_f1=f1_score(y_test, pred, average="macro"),
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run()
    pd.set_option("display.width", 120)
    print(results.pivot(index="n_train", columns="model", values="accuracy").round(3))
    results.to_csv("data/synthetic/baseline_sample_efficiency.csv", index=False)
    print("\nСохранено -> data/synthetic/baseline_sample_efficiency.csv")
    print("\nНапоминание: см. docstring файла -- сравнение иллюстративное (данные ")
    print("сгенерированы из самой БСД), демонстрирует форму кривой, а не 'победу' БСД.")
