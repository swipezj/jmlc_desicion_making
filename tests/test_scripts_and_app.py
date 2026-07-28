"""Тесты на служебный слой: пересчёт PRIOR, изоляция структуры, живость UI.

Эти три вещи объединены в один модуль не по смыслу, а по роли: всё это
проверки, что репозиторий не разваливается по краям. Именно края обычно и
ломаются при правке ядра, а замечают это позже всех.
"""
from __future__ import annotations

import runpy
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.model import BayesNet
from src.network_spec import PARENTS, ROOTS, STATES
from scripts.fit_priors import audit, estimate, render, wilson


# --------------------------------------------------------------------------
# 1. scripts/fit_priors.py
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_export() -> pd.DataFrame:
    """Синтетическая «выгрузка из АИС» с известным истинным PRIOR."""
    df = BayesNet.from_spec().sample(4000, np.random.default_rng(1))[list(ROOTS)]
    df["nomer_uzla"] = np.repeat(np.arange(200), 20)
    return df


def test_audit_passes_on_clean_export(sample_export):
    assert audit(sample_export) == []


def test_audit_catches_missing_column(sample_export):
    problems = audit(sample_export.drop(columns=["flow"]))
    assert any("flow" in p for p in problems)


def test_audit_catches_unknown_category(sample_export):
    bad = sample_export.copy()
    bad.loc[bad.index[0], "age"] = "неизвестно_сколько"
    problems = audit(bad)
    assert any("неизвестные категории" in p for p in problems)


def test_audit_reports_missing_values(sample_export):
    bad = sample_export.copy()
    bad.loc[bad.index[:10], "temperature"] = np.nan
    assert any("пропусков" in p for p in audit(bad))


def test_estimated_frequencies_recover_true_prior(sample_export):
    """На 4000 наблюдений оценка должна сойтись к тем числам, из которых
    данные и порождались. Это главный тест модуля: если он падает, скрипт
    считает не то, что обещает."""
    est = estimate(sample_export)
    for _, r in est.iterrows():
        assert abs(r["частота"] - r["было_в_модели"]) < 0.03, r.to_dict()


def test_estimates_are_normalised(sample_export):
    """Частоты в отчёте округлены до 4 знаков, поэтому сумма может отличаться
    от единицы на последнем знаке; `render` нормирует ещё раз перед вставкой
    в код, и вот там расхождения быть уже не должно."""
    est = estimate(sample_export)
    for v in ROOTS:
        assert est.loc[est["узел"] == v, "частота"].sum() == pytest.approx(1.0, abs=1e-3)


def test_confidence_interval_covers_estimate(sample_export):
    est = estimate(sample_export)
    for _, r in est.iterrows():
        assert r["ДИ_нижн"] <= r["частота"] <= r["ДИ_верх"] + 1e-9


def test_wilson_handles_degenerate_counts():
    """Ноль наблюдений и полное совпадение -- те случаи, на которых
    нормальный интервал даёт бессмыслицу вроде [0, 0]."""
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson(50, 50)
    assert lo < 1.0 and hi == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_thin_categories_are_flagged(sample_export):
    """Категория с горсткой наблюдений должна быть помечена, иначе шум
    попадёт в модель под видом измерения."""
    thin = sample_export[sample_export["age"] != "gt_7"].copy()
    thin = pd.concat([thin, sample_export[sample_export["age"] == "gt_7"].head(3)])
    est = estimate(thin)
    row = est[(est["узел"] == "age") & (est["состояние"] == "gt_7")].iloc[0]
    assert bool(row["мало_данных"])


def test_render_produces_importable_python(sample_export):
    block = render(estimate(sample_export))
    ns: dict = {}
    exec(block, {"dict": dict, "str": str, "list": list, "float": float}, ns)
    prior = ns["PRIOR"]
    assert set(prior) == set(ROOTS)
    for v in ROOTS:
        assert len(prior[v]) == len(STATES[v])
        assert sum(prior[v]) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# 2. Изоляция структуры от глобальной спецификации
# --------------------------------------------------------------------------
def test_network_does_not_share_structure_with_spec():
    """`from_spec()` обязана отдавать собственные словари.

    Иначе эксперименты со структурой (structure_learning их ставит) молча
    портили бы network_spec для всего остального процесса.
    """
    net = BayesNet.from_spec()
    net.parents["maintenance"] = ("device_cond",)
    net.states["flow"].append("мусор")

    assert PARENTS["maintenance"] == ("device_cond", "pipe_cond",
                                      "reliability", "failure_prob")
    assert "мусор" not in STATES["flow"]
    BayesNet.from_spec().validate()   # спецификация всё ещё собирается


def test_copy_is_independent():
    net = BayesNet.from_spec()
    clone = net.copy()
    clone.parents["anomaly"] = ()
    clone.cpt["pipe_cond"][:] = 0.0
    assert net.parents["anomaly"] == PARENTS["anomaly"]
    net.validate()


# --------------------------------------------------------------------------
# 3. Streamlit-приложение
# --------------------------------------------------------------------------
def _streamlit_stub() -> types.ModuleType:
    """Заглушка streamlit: позволяет прогнать app/ без установки UI-стека."""
    st = types.ModuleType("streamlit")

    class Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, _): return noop

    def noop(*a, **k): return Ctx()

    for name in ("set_page_config", "title", "caption", "markdown", "header",
                 "subheader", "write", "dataframe", "bar_chart", "pyplot",
                 "divider", "metric", "info", "warning", "success", "expander",
                 "container", "table", "text", "json", "code", "latex", "stop"):
        setattr(st, name, noop)
    st.selectbox = lambda label, options, **k: list(options)[0]
    st.radio = lambda label, options, **k: list(options)[0]
    st.slider = lambda label, lo, hi, *a, **k: lo
    st.number_input = lambda label, *a, **k: k.get("value", 1.0)
    st.checkbox = lambda *a, **k: False
    st.button = lambda *a, **k: False
    st.columns = lambda n, **k: [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels, **k: [Ctx() for _ in labels]
    st.sidebar = Ctx()
    for attr in ("selectbox", "radio", "slider", "checkbox", "number_input",
                 "markdown", "header", "caption", "title", "write", "divider", "button"):
        setattr(st.sidebar, attr, getattr(st, attr))
    st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.cache_resource = st.cache_data
    return st


def test_streamlit_app_runs_end_to_end(monkeypatch):
    """Приложение исполняется целиком на заглушке UI.

    Ловит ровно тот класс поломок, который иначе всплывает у проверяющего:
    правка структуры сети в src/ рассогласована с обращениями к узлам в app/.
    """
    monkeypatch.setitem(sys.modules, "streamlit", _streamlit_stub())
    monkeypatch.setattr(sys, "argv", ["streamlit_app.py"])
    runpy.run_path("app/streamlit_app.py", run_name="__main__")
