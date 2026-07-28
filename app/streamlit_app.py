"""
Интерактивная демонстрация: диспетчер выбирает показания приборов ->
сеть выдаёт апостериорные вероятности -> решающий слой переводит их в
действие с учётом цены ошибки.

Запуск:  streamlit run app/streamlit_app.py

Приложение не зависит от pgmpy: вывод идёт через src/reference_inference.py
(чистый numpy), поэтому демо запускается там, где pgmpy не установлен.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from src.decision import ACTION_LABEL, ACTIONS, Costs, breakeven_ratio, expected_losses
from src.model import BayesNet
from src.network_spec import DECISION_RULES, ROOTS, RUSSIAN_NAME, STATES
from src.reference_inference import posterior

st.set_page_config(page_title="БСД — предиктивное ТО инженерных систем", layout="wide")
st.title("Байесовская сеть доверия — предиктивное ТО")
st.caption(
    "Python-реконструкция модели из ВКР «Разработка системы управления качеством "
    "технического обслуживания инженерных систем с использованием байесовских "
    "сетей доверия» (СПбГЭУ, 2026). Исходная модель реализована в HUGIN Researcher; "
    "это независимая программная демонстрация той же логики."
)


@st.cache_resource
def _net() -> BayesNet:
    return BayesNet.from_spec()


with st.sidebar:
    st.header("Цены ошибок")
    st.caption(
        "За единицу принята стоимость срочного выезда. Именно эти числа, а не "
        "порог вероятности, определяют, когда система советует выехать."
    )
    miss = st.slider("Пропущенный отказ дороже выезда в … раз", 0.5, 40.0, 5.0, 0.5)
    planned = st.slider("Плановый выезд (доля от срочного)", 0.05, 1.0, 0.4, 0.05)
    late = st.slider("Доля ущерба при опоздании на срочный случай", 0.0, 1.0, 0.5, 0.05)
    costs = Costs(planned_visit=planned, miss_to_visit=miss, late_fraction=late)

st.subheader("Наблюдаемые параметры (то, что реально приходит с АИС/ЛЭРС УЧЁТ)")
evidence = {}
cols = st.columns(3)
for i, var in enumerate(ROOTS):
    with cols[i % 3]:
        evidence[var] = st.selectbox(RUSSIAN_NAME[var], STATES[var], key=var)
        if var == "temperature":
            st.caption("норма определяется температурным графиком по уличной "
                       "температуре, допуск ±3% (ПТЭ ТЭ, приказ Минэнерго № 115)")

if st.button("Выполнить вероятностный вывод", type="primary"):
    targets = ["device_cond", "pipe_cond", "reliability",
               "failure_prob", "maintenance", "anomaly"]
    result = posterior(evidence, targets=targets)

    left, right = st.columns(2)
    with left:
        st.subheader("Скрытые состояния системы")
        for t in ["device_cond", "pipe_cond", "reliability"]:
            st.write(f"**{RUSSIAN_NAME[t]}**")
            st.bar_chart({s: float(p) for s, p in zip(STATES[t], result[t])})
    with right:
        st.subheader("Управленческие гипотезы")
        for t in ["failure_prob", "maintenance", "anomaly"]:
            st.write(f"**{RUSSIAN_NAME[t]}**")
            st.bar_chart({s: float(p) for s, p in zip(STATES[t], result[t])})

    st.divider()
    st.subheader("Решающий слой: во что обходится каждое действие")
    losses = expected_losses(result["maintenance"], costs)
    best = int(losses.argmin())
    st.dataframe(
        pd.DataFrame({
            "действие": [ACTION_LABEL[a] for a in ACTIONS],
            "ожидаемые потери, у.е.": losses.round(3),
            "": ["<-- минимум" if i == best else "" for i in range(len(ACTIONS))],
        }),
        hide_index=True, use_container_width=True,
    )
    st.success(f"Решение при заданных ценах: {ACTION_LABEL[ACTIONS[best]]}")

    switch = breakeven_ratio(_net(), evidence, costs)
    bounds = list(switch.items()) + [(None, None)]
    st.caption("При каком отношении «цена пропуска / цена выезда» что оптимально:")
    st.write("  ·  ".join(
        f"**{f'R < {hi:.1f}' if hi is not None else f'R ≥ {lo:.1f}'}** → {ACTION_LABEL[a]}"
        for (a, lo), (_, hi) in zip(bounds[:-1], bounds[1:])
    ))

    with st.expander("Для сравнения: жёсткие правила из Табл. 3.5 диплома"):
        rules = [action for node, state, action in DECISION_RULES
                 if state == STATES[node][int(result[node].argmax())]]
        for r in rules:
            st.write("· " + r)
        st.caption(
            "Эти правила выбирают действие по наиболее вероятному состоянию и "
            "неявно считают все ошибки равноценными. На синтетическом прогоне "
            "(python -m src.decision) такое правило пропускает больше срочных "
            "случаев, чем даже пороговое правило по расходу."
        )

st.divider()
st.caption(
    "Числа для узлов device_cond и pipe_cond воспроизводят диплом (Табл. 3.3-3.4) "
    "точно; остальные CPT реконструированы (полные таблицы из HUGIN не были "
    "опубликованы) — подробности в README, раздел «Происхождение чисел». "
    "Цены ошибок в боковой панели — иллюстративные, а не тарифы предприятия."
)
