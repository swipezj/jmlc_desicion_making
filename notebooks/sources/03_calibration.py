# %% [markdown]
# # 3. Разрыв в сценарии 3: поиск недостающей дуги
#
# Реконструированная сеть воспроизводит Табл. 3.6 качественно, но в
# сценарии 3 даёт `P(отказ = высокая) = 0.37` против `0.70` в дипломе.
# Расхождение в 0.33 слишком велико, чтобы списать на округление.
#
# Этот ноутбук — небольшое расследование: сначала показывается, что разрыв
# **не может** быть объяснён в рамках заявленной структуры, затем
# формулируется и проверяется гипотеза о родителе, которого нет в тексте §3.2.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import BayesNet
from src.network_spec import SCENARIOS_TABLE_3_6, STATES
from src.calibration import (HYPOTHESES, THESIS, _expert_compact, calibrated_network,
                             expand, fit, monotonicity_report, scenario_outputs,
                             targets_vector)

pd.set_option("display.width", 160)
net = BayesNet.from_spec()
expert_full = expand(_expert_compact(), ("device_cond", "pipe_cond"))

# %% [markdown]
# ## 3.1 Улика: числа диплома несовместимы с заявленной структурой
#
# Выпишем, что различается между сценариями, и что при этом делает `P(отказ)`
# в дипломе.

# %%
rows = []
for name, spec in SCENARIOS_TABLE_3_6.items():
    ev = spec["evidence"]
    post = net.posterior(ev, ["device_cond", "pipe_cond"])
    rows.append({
        "сценарий": name.split("(")[0].strip(),
        "возраст": ev["age"], "поверка": ev["calibration"],
        "давление": ev["pressure"], "расход": ev["flow"],
        "P(прибор неисправен)": round(float(post["device_cond"][2]), 3),
        "P(труба: норма)": round(float(post["pipe_cond"][0]), 3),
        "P(отказ) диплом": spec["thesis_posterior"]["failure_prob"],
    })
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# Сравним сценарии 2 и 3. Состояние прибора **одинаковое** (срок > 7 лет,
# поверка истекла — значит, и распределение `device_cond` совпадает), а
# `P(отказ)` в дипломе отличается почти в полтора раза: 0.52 против 0.70.
#
# Объяснить это через трубопровод нельзя: в сценарии 3 труба **вероятнее**
# исправна (0.70 против 0.55). То есть оба родителя `failure_prob`,
# заявленные в §3.2, в сценарии 3 «лучше», а вероятность отказа при этом
# выше. В рамках структуры `failure_prob ← (device_cond, pipe_cond)` такое
# невозможно ни при каких CPT.
#
# Что ещё различается? Расход: в сценарии 3 он аномально высокий, в
# сценарии 2 — нормальный. Отсюда две гипотезы: у `failure_prob` есть
# дополнительный родитель — либо расход напрямую, либо достоверность
# показаний (недостоверные показания сами по себе симптом отказа прибора).

# %% [markdown]
# ## 3.2 Инструмент: замкнутая формула вместо перебора
#
# При полном свидетельстве на входах `device_cond` и `pipe_cond` условно
# независимы, поэтому апостериорные выходы считаются свёрткой четырёх
# маленьких тензоров. Это в сотни раз быстрее перебора — что критично, раз
# формулу предстоит вызывать десятки тысяч раз внутри оптимизатора.
#
# Прежде чем ей доверять, сверим с точным перебором.

# %%
worst = 0.0
for spec in SCENARIOS_TABLE_3_6.values():
    a = scenario_outputs(spec["evidence"], expert_full)
    b = net.posterior(spec["evidence"], list(a))
    for t in a:
        worst = max(worst, float(np.abs(a[t] - b[t]).max()))
print(f"максимальное расхождение с полным перебором: {worst:.2e}")

# %% [markdown]
# ## 3.3 Постановка подгонки
#
# Для каждой гипотезы о наборе родителей подбираем CPT, минимизируя
#
# $$L = \sum_{\text{сценарии}} (p_{\text{модель}} - p_{\text{диплом}})^2
#      + \lambda \, \|\mathrm{logit}(CPT) - \mathrm{logit}(CPT_{\text{эксперт}})\|^2$$
#
# **Второе слагаемое — суть метода, а не техническая деталь.** Целевых чисел
# всего шесть, а свободных параметров от 27 до 81: без штрафа подойдёт любая
# гипотеза, и «удалось подогнать» не будет значить ничего. Штраф превращает
# вопрос «можно ли подогнать?» в «насколько сильно придётся исказить
# экспертную таблицу, чтобы подогнать?». Правдоподобна та гипотеза, которой
# хватает малого искажения.
#
# `P(anomaly)` в целевую функцию не входит: `anomaly` не потомок
# `failure_prob`, и от этой CPT не зависит вовсе — служит контрольной
# величиной.

# %%
before = targets_vector(expert_full)
labels = [f"сценарий {i+1}: {t}" for i in range(3) for t in ("P(отказ=высокая)", "P(ТО=срочное)")]
print(pd.DataFrame({"величина": labels, "модель": before.round(3), "диплом": THESIS,
                    "Δ": (before - THESIS).round(3)}).to_string(index=False))
print(f"\nRMSE исходной реконструкции: {np.sqrt(((before - THESIS) ** 2).mean()):.4f}")

# %% [markdown]
# ## 3.4 Сравнение гипотез

# %%
lams = (0.0, 0.01, 0.05, 0.2, 1.0)
rows = []
for name, parents in HYPOTHESES.items():
    for lam in lams:
        r = fit(parents, lam, expert_full)
        rows.append({"гипотеза": name, "λ": lam, "RMSE": round(r["rmse"], 4),
                     "искажение CPT": round(r["distortion"], 4)})
cmp = pd.DataFrame(rows)
print(cmp.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(7.5, 5))
for (name, parents), color in zip(HYPOTHESES.items(), ["#c44e52", "#4c72b0", "#55a868"]):
    sub = cmp[cmp["гипотеза"] == name].sort_values("искажение CPT")
    ax.plot(sub["искажение CPT"], sub["RMSE"], "-o", ms=5, color=color, label=name)
ax.axhline(np.sqrt(((before - THESIS) ** 2).mean()), ls="--", c="gray", lw=1)
ax.annotate("исходная реконструкция", (0.005, np.sqrt(((before - THESIS) ** 2).mean())),
            fontsize=8, va="bottom", color="gray")
ax.set_xlabel("среднее искажение экспертной CPT, |Δ вероятности|")
ax.set_ylabel("RMSE относительно шести чисел Табл. 3.6")
ax.set_title("Цена согласия с дипломом: ниже и левее — лучше")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# Кривая гипотезы с дополнительным родителем лежит **ниже и левее** — то есть
# при любом уровне искажения даёт лучшее согласие. Исходная структура §3.2
# способна достичь согласия только ценой переписывания таблицы почти целиком.
#
# При этом двух конкурирующих гипотез — «расход» и «достоверность показаний» —
# шесть чисел разделить не могут: кривые идут вплотную. Это ожидаемо, ведь в
# сети `reliability` сама зависит от `flow`. Разрешается это данными, а не
# оптимизацией.

# %% [markdown]
# ## 3.5 Результат

# %%
best_name = "B: device, pipe, flow"
cal_net, report = calibrated_network(best_name, lam=0.05)
after = report["pred"]
print(pd.DataFrame({"величина": labels, "было": before.round(3),
                    "стало": after.round(3), "диплом": THESIS,
                    "Δ итог": (after - THESIS).round(3)}).to_string(index=False))
print(f"\nRMSE: {np.sqrt(((before - THESIS) ** 2).mean()):.4f} → {report['rmse']:.4f}")
print("\nКачественные проверки подогнанной таблицы (не должны нарушиться):")
for k, v in monotonicity_report(report["full"]).items():
    print(f"  [{'OK ' if v else 'НЕТ'}] {k}")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
y = np.arange(len(labels))
ax.barh(y - 0.26, THESIS, height=0.25, label="диплом (HUGIN)", color="#8172b3")
ax.barh(y, before, height=0.25, label="реконструкция", color="#dd8452")
ax.barh(y + 0.26, after, height=0.25, label="после калибровки", color="#55a868")
ax.set_yticks(y, labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("вероятность")
ax.set_title("Разрыв со сценарием 3 закрыт большей частью")
ax.legend(fontsize=8)
ax.grid(axis="x", alpha=.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3.6 Что из этого следует, а что нет
#
# **Следует:** числа диплома внутренне несовместимы со структурой,
# описанной в §3.2 словами; у `failure_prob` почти наверняка есть третий
# родитель, связанный с расходом. Это содержательное наблюдение об
# оригинальной модели, полученное из одних опубликованных чисел.
#
# **Не следует:** что найденная таблица — «восстановленный оригинал». Она
# подогнана под шесть чисел из текста, а не выгружена из файла HUGIN.
# Поэтому `calibrated_network()` возвращает отдельную сеть, а
# `network_spec.py` не меняется: переключение на калиброванную версию должно
# быть сознательным действием, а не умолчанием.
