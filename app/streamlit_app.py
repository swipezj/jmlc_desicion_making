"""
Интерактивная демонстрация: диспетчер выбирает показания приборов ->
сеть выдаёт апостериорные вероятности и рекомендацию.

Запуск:  streamlit run app/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.network_spec import STATES, RUSSIAN_NAME, ROOTS, DECISION_RULES
from src.reference_inference import posterior

st.set_page_config(page_title="БСД — предиктивное ТО инженерных систем", layout="centered")
st.title("Байесовская сеть доверия — предиктивное ТО")
st.caption(
    "Python-реконструкция модели из ВКР «Разработка системы управления качеством "
    "технического обслуживания инженерных систем с использованием байесовских "
    "сетей доверия» (СПбГЭУ, 2026). Исходная модель реализована в HUGIN Researcher; "
    "это независимая программная демонстрация той же логики."
)

st.subheader("Наблюдаемые параметры (то, что реально приходит с АИС/ЛЭРС УЧЁТ)")
evidence = {}
cols = st.columns(2)
for i, var in enumerate(ROOTS):
    with cols[i % 2]:
        evidence[var] = st.selectbox(RUSSIAN_NAME[var], STATES[var], key=var)

if st.button("Выполнить вероятностный вывод", type="primary"):
    targets = ["device_cond", "pipe_cond", "reliability", "failure_prob", "maintenance", "anomaly"]
    result = posterior(evidence, targets=targets)

    st.subheader("Скрытые состояния системы")
    for t in ["device_cond", "pipe_cond", "reliability"]:
        st.write(f"**{RUSSIAN_NAME[t]}**")
        st.bar_chart({s: float(p) for s, p in zip(STATES[t], result[t])})

    st.subheader("Управленческие гипотезы")
    recommendations = []
    for t in ["failure_prob", "maintenance", "anomaly"]:
        st.write(f"**{RUSSIAN_NAME[t]}**")
        st.bar_chart({s: float(p) for s, p in zip(STATES[t], result[t])})
        best_state = STATES[t][int(result[t].argmax())]
        for node, state, action in DECISION_RULES:
            if node == t and state == best_state:
                recommendations.append(action)

    st.subheader("Рекомендуемое действие")
    if recommendations:
        for r in recommendations:
            st.success(r)
    else:
        st.info("Существенных отклонений не выявлено, плановый режим наблюдения.")

st.divider()
st.caption(
    "Числа для узлов device_cond и pipe_cond воспроизводят диплом (Табл. 3.3-3.4) "
    "точно; остальные CPT реконструированы (полные таблицы из HUGIN не были "
    "опубликованы) — подробности в README."
)
