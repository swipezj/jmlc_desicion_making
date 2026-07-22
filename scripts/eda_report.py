"""
Мини-EDA по синтетическим данным: распределения переменных + два
кросс-таба, иллюстрирующие причинные эффекты, заложенные в CPT.
Не зависит от pgmpy.

Запуск: python -m scripts.eda_report
"""
import os
import matplotlib.pyplot as plt

from src.data_generator import generate
from src.network_spec import STATES, RUSSIAN_NAME, ROOTS

OUT_DIR = "reports/figures"


def plot_distributions(df, out_path=f"{OUT_DIR}/eda_distributions.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    variables = list(ROOTS) + ["failure_prob", "maintenance", "anomaly"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, var in zip(axes.flat, variables):
        counts = df[var].value_counts().reindex(STATES[var]).fillna(0)
        ax.bar(range(len(counts)), counts.values, color="#3E8E7E")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(RUSSIAN_NAME[var], fontsize=9)
    for ax in axes.flat[len(variables):]:
        ax.axis("off")
    fig.suptitle(f"Распределения переменных на синтетической выборке (n={len(df)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено -> {out_path}")


def plot_causal_crosstabs(df, out_path=f"{OUT_DIR}/eda_causal_effects.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ct1 = df.groupby(["age", "calibration"])["device_cond"].apply(
        lambda s: (s == "исправен").mean()
    ).unstack().reindex(index=STATES["age"], columns=STATES["calibration"])
    im1 = axes[0].imshow(ct1.values, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(ct1.columns))); axes[0].set_xticklabels(ct1.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(len(ct1.index))); axes[0].set_yticklabels(ct1.index)
    axes[0].set_title("P(прибор исправен) | возраст, поверка")
    for i in range(ct1.shape[0]):
        for j in range(ct1.shape[1]):
            axes[0].text(j, i, f"{ct1.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=axes[0], fraction=0.046)

    ct2 = df.groupby(["pressure", "flow"])["pipe_cond"].apply(
        lambda s: (s == "нарушена_герметичность").mean()
    ).unstack().reindex(index=STATES["pressure"], columns=STATES["flow"])
    im2 = axes[1].imshow(ct2.values, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(ct2.columns))); axes[1].set_xticklabels(ct2.columns, rotation=30, ha="right")
    axes[1].set_yticks(range(len(ct2.index))); axes[1].set_yticklabels(ct2.index)
    axes[1].set_title("P(нарушена герметичность) | давление, расход")
    for i in range(ct2.shape[0]):
        for j in range(ct2.shape[1]):
            axes[1].text(j, i, f"{ct2.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    fig.suptitle("Причинные эффекты, восстановленные из синтетической выборки (сверка с Табл. 3.3-3.4)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено -> {out_path}")


if __name__ == "__main__":
    df = generate(n=5000, seed=1)
    print(df.describe(include="all").T[["count", "unique", "top", "freq"]])
    plot_distributions(df)
    plot_causal_crosstabs(df)
