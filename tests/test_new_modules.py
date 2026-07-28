"""
Тесты новых модулей. Ни один из них не зависит от pgmpy, поэтому набор
проходит в любом окружении, где есть numpy/pandas/scipy/sklearn.

Главный тест здесь -- `test_engine_matches_reference_oracle`: он связывает
обобщённый движок (src/model.py) с независимо написанным перебором
(src/reference_inference.py) на всех 108 комбинациях входных свидетельств.
Пока он зелёный, третья реализация вывода не размывает единый источник
правды, а служит ещё одной взаимной проверкой.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.calibration import (HYPOTHESES, THESIS, _current_compact, _expert_compact, expand, fit,
                             scenario_outputs, targets_vector)
from src.decision import (ACTIONS, Costs, NEED, bayes_action, evaluate,
                          expected_losses, policy_actions, posterior_lookup, triage)
from src.model import BayesNet
from src.network_spec import ROOTS, SCENARIOS_TABLE_3_6, STATES
from src.reference_inference import posterior as ref_posterior
from src.sensitivity import (expected_risk, inert_inputs, mutual_information,
                             one_at_a_time, value_of_information)
from src.structure_learning import BICScorer, compare_to_expert, hill_climb

TARGETS = ["failure_prob", "maintenance", "anomaly"]
ALL_EVIDENCE = list(itertools.product(*[STATES[v] for v in ROOTS]))


@pytest.fixture(scope="module")
def net() -> BayesNet:
    return BayesNet.from_spec()


# --------------------------------------------------------------- src/model.py
def test_engine_matches_reference_oracle(net):
    """Обобщённый движок и референсный перебор совпадают на всех 108 входах."""
    for combo in ALL_EVIDENCE:
        ev = dict(zip(ROOTS, combo))
        a, b = net.posterior(ev, TARGETS), ref_posterior(ev, TARGETS)
        for t in TARGETS:
            np.testing.assert_allclose(a[t], b[t], atol=1e-12)


def test_cpt_shapes_and_normalisation(net):
    net.validate()          # не должно бросать
    # 428, а не прежние 146: температура и давление добавились в родители
    # device_cond, pipe_cond получил температуру, reliability и maintenance --
    # по два новых родителя каждый, у anomaly расход сохранён. См. README,
    # «Пересмотр структуры».
    assert net.n_free_params == 428


def test_validate_rejects_broken_cpt(net):
    bad = net.copy()
    bad.cpt["pipe_cond"] = bad.cpt["pipe_cond"] * 0.9
    with pytest.raises(ValueError, match="не нормирована"):
        bad.validate()


def test_sampling_reproduces_marginals(net):
    df = net.sample(40000, np.random.default_rng(0))
    for v in ["device_cond", "maintenance", "anomaly"]:
        empirical = df[v].value_counts(normalize=True).reindex(STATES[v]).fillna(0).to_numpy()
        np.testing.assert_allclose(empirical, net.posterior({}, [v])[v], atol=0.015)


def test_fit_recovers_cpt_on_large_sample(net):
    # 200k, а не 50k: пространство конфигураций родителей выросло вчетверо
    # (device_cond: 9 строк -> 36), и на 50k редкие комбинации вроде
    # "давление откл. + температура откл. + поверка истекла + прибор до 3 лет"
    # набирают несколько десятков наблюдений -- оценка по ним шумная.
    df = net.sample(200_000, np.random.default_rng(1))
    learned = net.fit(df, alpha=1.0)
    for v in ["device_cond", "pipe_cond"]:
        assert np.abs(learned.cpt[v] - net.cpt[v]).max() < 0.12


def test_fit_does_not_mutate_original(net):
    before = net.cpt["device_cond"].copy()
    net.fit(net.sample(200, np.random.default_rng(2)))
    np.testing.assert_array_equal(net.cpt["device_cond"], before)


def test_perturb_changes_cpt_but_keeps_validity(net):
    p = net.perturb(np.random.default_rng(3), 0.5)
    p.validate()
    assert np.abs(p.cpt["failure_prob"] - net.cpt["failure_prob"]).mean() > 0.01


def test_posterior_rejects_unknown_evidence(net):
    with pytest.raises(KeyError):
        net.posterior({"нет_такого_узла": "x"})
    with pytest.raises(KeyError):
        net.posterior({"flow": "нет_такого_состояния"})


# ------------------------------------------------------------ src/decision.py
def test_loss_matrix_orientation():
    """Пропустить срочный случай дороже, чем съездить зря."""
    L = Costs().matrix()
    urgent_col = STATES[NEED].index("срочное")
    none_col = STATES[NEED].index("не_требуется")
    assert L[ACTIONS.index("не_выезжать"), urgent_col] > L[ACTIONS.index("срочный_выезд"), none_col]
    assert L[ACTIONS.index("плановый_выезд"), none_col] < L[ACTIONS.index("срочный_выезд"), none_col]


def test_certain_urgent_need_triggers_urgent_visit():
    belief = np.zeros(3)
    belief[STATES[NEED].index("срочное")] = 1.0
    assert bayes_action(belief, Costs()) == "срочный_выезд"


def test_certain_no_need_triggers_no_visit():
    belief = np.zeros(3)
    belief[STATES[NEED].index("не_требуется")] = 1.0
    assert bayes_action(belief, Costs()) == "не_выезжать"


def test_higher_miss_cost_never_reduces_dispatch(net):
    """Монотонность: дороже пропуск -> выездов не меньше. Свойство матрицы потерь."""
    df = net.sample(4000, np.random.default_rng(5))
    lookup = posterior_lookup(net)
    prev = -1
    for r in [0.5, 1, 2, 5, 10, 30]:
        c = Costs(miss_to_visit=float(r))
        n_visits = (policy_actions(df, "bayes", net, c, lookup) != "не_выезжать").sum()
        assert n_visits >= prev
        prev = n_visits


def test_bayes_policy_beats_alternatives_on_its_own_criterion(net):
    """Байесовское решение минимизирует ожидаемые потери -- иначе оно неверно."""
    df = net.sample(4000, np.random.default_rng(6))
    lookup = posterior_lookup(net)
    costs = Costs()
    losses = {p: evaluate(df, policy_actions(df, p, net, costs, lookup), costs)["средние_потери"]
              for p in ["reactive", "map", "bayes"]}
    assert losses["bayes"] <= min(losses["reactive"], losses["map"]) + 1e-9


def test_triage_respects_capacity(net):
    df = net.sample(2000, np.random.default_rng(8))
    acts = triage(df, capacity=37, net=net, costs=Costs())
    assert (acts != "не_выезжать").sum() <= 37


def test_triage_beats_random_allocation_of_same_budget(net):
    df = net.sample(4000, np.random.default_rng(9))
    costs = Costs()
    budget = 400
    smart = evaluate(df, triage(df, budget, net, costs), costs)["средние_потери"]
    rng = np.random.default_rng(10)
    acts = np.full(len(df), "не_выезжать", dtype=object)
    acts[rng.permutation(len(df))[:budget]] = "срочный_выезд"
    assert smart < evaluate(df, acts, costs)["средние_потери"]


def test_expected_losses_length_matches_actions(net):
    belief = net.posterior({}, [NEED])[NEED]
    assert expected_losses(belief, Costs()).shape == (len(ACTIONS),)


# --------------------------------------------------------- src/sensitivity.py
def test_no_input_is_inert(net):
    """Все пять входных каналов АИС влияют хоть на что-то.

    Раньше здесь стоял обратный тест: температура была изолирована, и это
    фиксировалось как известное ограничение структуры §3.2. После
    восстановления дуг по экспертному опросу мёртвых входов быть не должно
    -- если тест упадёт, значит очередная правка снова обесточила канал.
    """
    assert inert_inputs(net) == []


def test_flow_is_the_strongest_input(net):
    oat = one_at_a_time(net)
    spread = oat.groupby("вход")["anomaly"].agg(lambda c: c.max() - c.min())
    assert spread.idxmax() == "flow"


def test_mutual_information_non_negative(net):
    for v in ROOTS:
        for t in TARGETS:
            assert mutual_information(net, v, t) >= 0.0


def test_information_never_hurts(net):
    """Ожидаемый риск не растёт при добавлении наблюдений -- базовое свойство VOI."""
    costs = Costs()
    r_none = expected_risk(net, [], costs)
    r_one = expected_risk(net, ["pressure"], costs)
    r_all = expected_risk(net, list(ROOTS), costs)
    assert r_all <= r_one + 1e-12 <= r_none + 1e-12


def test_temperature_has_positive_value(net):
    """Опрос канала температуры перестал быть бесполезным.

    Эффект слабый по построению (отклонение от температурного графика чаще
    означает ошибку теплоснабжающей организации, чем отказ прибора), поэтому
    проверяется только знак, а не величина.
    """
    voi = value_of_information(net).set_index("вход")
    assert voi.loc["temperature", "вклад_сверх_остальных"] > 0.0


# --------------------------------------------------------- src/calibration.py
def test_closed_form_matches_enumeration(net):
    """Аналитическая формула для сценариев эквивалентна полному перебору.

    Берём CPT_FAILURE из действующей спецификации (после перехода на
    гипотезу B у неё три родителя), а не экспертную двухродительскую
    базу: иначе тест сверял бы формулу с сетью, которой уже нет.
    """
    full = expand(_current_compact(), HYPOTHESES["B: device, pipe, flow"])
    for spec in SCENARIOS_TABLE_3_6.values():
        analytic = scenario_outputs(spec["evidence"], full)
        exact = net.posterior(spec["evidence"], list(analytic))
        for t in analytic:
            np.testing.assert_allclose(analytic[t], exact[t], atol=1e-10)


def test_calibration_reduces_gap_to_thesis():
    full = expand(_expert_compact(), ("device_cond", "pipe_cond"))
    before = float(np.sqrt(((targets_vector(full) - THESIS) ** 2).mean()))
    after = fit(HYPOTHESES["B: device, pipe, flow"], 0.05, full)["rmse"]
    assert after < before / 2


def test_extra_parent_dominates_original_structure():
    """Ключевой вывод модуля: тот же λ, лучше согласие И меньше искажение."""
    full = expand(_expert_compact(), ("device_cond", "pipe_cond"))
    a = fit(HYPOTHESES["A: device, pipe (как в §3.2)"], 0.05, full)
    b = fit(HYPOTHESES["B: device, pipe, flow"], 0.05, full)
    assert b["rmse"] < a["rmse"] and b["distortion"] < a["distortion"]


def test_fitted_cpt_is_a_valid_distribution():
    full = expand(_expert_compact(), ("device_cond", "pipe_cond"))
    cpt = fit(HYPOTHESES["C: device, pipe, reliability"], 0.05, full)["cpt"]
    np.testing.assert_allclose(cpt.sum(axis=-1), 1.0, atol=1e-9)
    assert (cpt >= 0).all()


# -------------------------------------------------- src/structure_learning.py
def test_bic_prefers_true_edge_over_none(net):
    df = net.sample(3000, np.random.default_rng(11))
    s = BICScorer(df)
    assert s.local("device_cond", ("age",)) > s.local("device_cond", ())


def test_bic_penalises_irrelevant_parent(net):
    df = net.sample(3000, np.random.default_rng(12))
    s = BICScorer(df)
    # flow, а не temperature: температура теперь настоящий родитель
    # device_cond, и штрафовать её было бы неверно по существу. flow с
    # device_cond не смежен и не имеет с ним общих предков.
    assert s.local("device_cond", ("age", "flow")) < s.local("device_cond", ("age",))


def test_hill_climb_recovers_most_of_the_skeleton(net):
    df = net.sample(5000, np.random.default_rng(13))
    learned, _ = hill_climb(df)
    # Порог снижен с 0.75: у reliability и maintenance стало по четыре
    # родителя, а восстанавливать плотные родительские множества жадным
    # поиском заметно труднее, чем разреженные.
    assert compare_to_expert(learned)["скелет_F1"] >= 0.70


def test_hill_climb_does_not_claim_temperature_is_dead(net):
    """Раньше здесь проверялось, что алгоритм оставляет температуру изолированной.

    Теперь у температуры есть дуги, но эффект слабый, и требовать от жадного
    поиска по BIC их обнаружения на 5000 наблюдений было бы флаки-тестом.
    Проверяем то, что действительно должно выполняться: если алгоритм всё же
    нашёл дугу с температурой, она соединяет её с одним из двух настоящих
    потомков, а не с произвольным узлом.
    """
    df = net.sample(5000, np.random.default_rng(14))
    learned, _ = hill_climb(df)
    edges = {(p, c) for c, ps in learned.items() for p in ps}
    found = {a if b == "temperature" else b for a, b in edges if "temperature" in (a, b)}
    assert found <= {"device_cond", "pipe_cond"}, f"температура связалась с {found}"


def test_hill_climb_result_is_acyclic(net):
    df = net.sample(2000, np.random.default_rng(15))
    learned, _ = hill_climb(df)
    order, pending = [], list(STATES)
    while pending:
        ready = [v for v in pending if all(p in order for p in learned.get(v, ()))]
        assert ready, "в найденном графе цикл"
        order += ready
        pending = [v for v in pending if v not in order]


# --------------------------------------------------------- app/streamlit_app.py
def test_app_computation_path(net):
    """Повторяет вычисления демо-приложения без streamlit.

    Само приложение здесь не импортируется (это потянуло бы streamlit в
    зависимости тестов), но последовательность вызовов та же: вывод ->
    ожидаемые потери -> выбор действия -> границы по цене ошибки. Тест
    ловит рассинхронизацию API между src/decision.py и app/.
    """
    from src.decision import breakeven_ratio
    evidence = dict(pressure="otklonenie", temperature="norma", flow="anomal_high",
                    calibration="priblizhaetsya", age="3_7")
    result = ref_posterior(evidence, ["device_cond", "pipe_cond", "reliability",
                                      "failure_prob", "maintenance", "anomaly"])
    costs = Costs(planned_visit=0.4, miss_to_visit=5.0, late_fraction=0.5)
    losses = expected_losses(result[NEED], costs)
    assert losses.shape == (len(ACTIONS),)
    assert ACTIONS[int(losses.argmin())] in ACTIONS
    switch = breakeven_ratio(net, evidence, costs)
    assert switch and all(a in ACTIONS for a in switch)
