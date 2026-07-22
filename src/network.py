"""
Байесовская сеть на pgmpy -- воспроизводит модель из диплома (§3.2) в
виде исполняемого, тестируемого Python-кода вместо GUI-конфигурации в
HUGIN Researcher.

Использование:
    from src.network import build_network
    model = build_network()
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    result = infer.query(["maintenance"], evidence={"pressure": "otklonenie", ...})
"""
from __future__ import annotations
try:
    # pgmpy >= 0.1.25: BayesianNetwork renamed, old alias now raises on import
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    # older pgmpy: DiscreteBayesianNetwork doesn't exist yet
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from src.network_spec import (
    STATES, EDGES, PARENTS, PRIOR,
    CPT_DEVICE, CPT_PIPE, CPT_RELIABILITY, CPT_FAILURE, CPT_MAINT, cpt_anomaly,
)


def _root_cpd(var: str) -> TabularCPD:
    return TabularCPD(variable=var, variable_card=len(STATES[var]),
                       values=[[p] for p in PRIOR[var]],
                       state_names={var: STATES[var]})


def _child_cpd(var: str, table: dict[tuple, list[float]]) -> TabularCPD:
    """Собрать TabularCPD из словаря {(состояния родителей,): [P(состояние var)]}.

    pgmpy хочет values в форме (card[var], произведение card[родителей]),
    со столбцами в порядке itertools.product(*[states родителя_i]).
    state_names обязателен: без него pgmpy трактует состояния как индексы
    0,1,2,... и evidence со строками ("otklonenie" и т.п.) не резолвится.
    """
    import itertools
    parents = list(PARENTS[var])
    parent_states = [STATES[p] for p in parents]
    columns = []
    for combo in itertools.product(*parent_states):
        columns.append(table[combo])
    values = list(map(list, zip(*columns)))  # transpose -> (card[var], n_combos)
    state_names = {var: STATES[var]}
    state_names.update({p: STATES[p] for p in parents})
    return TabularCPD(
        variable=var, variable_card=len(STATES[var]), values=values,
        evidence=parents, evidence_card=[len(STATES[p]) for p in parents],
        state_names=state_names,
    )


def _anomaly_cpd() -> TabularCPD:
    import itertools
    parents = list(PARENTS["anomaly"])  # pipe_cond, reliability, flow
    parent_states = [STATES[p] for p in parents]
    columns = [cpt_anomaly(pipe, rel, flow) for pipe, rel, flow in itertools.product(*parent_states)]
    values = list(map(list, zip(*columns)))
    state_names = {"anomaly": STATES["anomaly"]}
    state_names.update({p: STATES[p] for p in parents})
    return TabularCPD(
        variable="anomaly", variable_card=len(STATES["anomaly"]), values=values,
        evidence=parents, evidence_card=[len(STATES[p]) for p in parents],
        state_names=state_names,
    )


def build_network() -> BayesianNetwork:
    """Собрать и вернуть готовую к вероятностному выводу pgmpy-модель."""
    model = BayesianNetwork(EDGES)
    # temperature участвует в CPT ни одного узла (см. network_spec.py --
    # "известное ограничение"), поэтому в списке EDGES её вообще нет, а
    # BayesianNetwork(edges) создаёт узлы только из рёбер. Без явного
    # add_node изолированный узел отсутствует в графе, и add_cpds ниже
    # упадёт с "CPD defined on variable not in the model".
    model.add_node("temperature")

    cpds = [_root_cpd(v) for v in ["pressure", "temperature", "flow", "calibration", "age"]]
    cpds.append(_child_cpd("device_cond", CPT_DEVICE))
    cpds.append(_child_cpd("pipe_cond", CPT_PIPE))
    cpds.append(_child_cpd("reliability", CPT_RELIABILITY))
    cpds.append(_child_cpd("failure_prob", CPT_FAILURE))
    cpds.append(_child_cpd("maintenance", CPT_MAINT))
    cpds.append(_anomaly_cpd())

    model.add_cpds(*cpds)
    assert model.check_model(), "Модель некорректна: сумма CPT по столбцам должна быть 1"
    return model


if __name__ == "__main__":
    from pgmpy.inference import VariableElimination

    model = build_network()
    infer = VariableElimination(model)

    evidence = {"pressure": "otklonenie", "temperature": "norma", "flow": "anomal_high",
                "calibration": "priblizhaetsya", "age": "3_7"}
    for target in ["failure_prob", "maintenance", "anomaly"]:
        print(infer.query([target], evidence=evidence))
