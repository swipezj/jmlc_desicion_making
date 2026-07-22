"""
Тесты pgmpy-модели. Требуют `pip install pgmpy` (см. requirements.txt).
Если pgmpy не установлен, тесты аккуратно скипаются, а не падают --
tests/test_reference.py работает и без pgmpy.
"""
import pytest

pgmpy = pytest.importorskip("pgmpy", reason="pgmpy не установлен -- `pip install pgmpy`")

from pgmpy.inference import VariableElimination

from src.network import build_network
from src.network_spec import SCENARIOS_TABLE_3_6, TARGET_HIGH_STATE, STATES
from src.reference_inference import posterior as reference_posterior

TOLERANCE = {"failure_prob": 0.35, "maintenance": 0.15, "anomaly": 0.15}


@pytest.fixture(scope="module")
def infer():
    model = build_network()
    return VariableElimination(model)


def test_model_is_valid(infer):
    assert infer.model.check_model()


@pytest.mark.parametrize("name", list(SCENARIOS_TABLE_3_6.keys()))
def test_pgmpy_matches_reference_implementation(infer, name):
    """pgmpy и независимая numpy-реализация должны давать ОДИНАКОВЫЙ ответ
    (это те же CPT) -- если тест падает, значит где-то разошлись CPT/структура
    между src/network.py и src/network_spec.py.
    """
    spec = SCENARIOS_TABLE_3_6[name]
    ref = reference_posterior(spec["evidence"], targets=list(spec["thesis_posterior"].keys()))
    for target in spec["thesis_posterior"]:
        pg = infer.query([target], evidence=spec["evidence"], show_progress=False)
        for i, state in enumerate(STATES[target]):
            assert pg.values[i] == pytest.approx(ref[target][i], abs=1e-6), (
                f"{name}/{target}/{state}: pgmpy={pg.values[i]:.4f} vs reference={ref[target][i]:.4f}"
            )


@pytest.mark.parametrize("name", list(SCENARIOS_TABLE_3_6.keys()))
def test_pgmpy_matches_thesis_within_tolerance(infer, name):
    spec = SCENARIOS_TABLE_3_6[name]
    for target, thesis_p in spec["thesis_posterior"].items():
        hi_state = TARGET_HIGH_STATE[target]
        result = infer.query([target], evidence=spec["evidence"], show_progress=False)
        model_p = result.values[STATES[target].index(hi_state)]
        assert abs(model_p - thesis_p) <= TOLERANCE[target]
