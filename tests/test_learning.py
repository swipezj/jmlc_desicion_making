"""Тесты обучения параметров. Требуют pgmpy -- скипаются, если его нет."""
import pytest

pytest.importorskip("pgmpy", reason="pgmpy не установлен -- `pip install pgmpy`")

from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.learning import learn
from src.network_spec import STATES
from src.data_generator import generate


@pytest.fixture(scope="module")
def synthetic_df():
    return generate(n=2000, seed=7)


@pytest.mark.parametrize("estimator_cls,kwargs", [
    (MaximumLikelihoodEstimator, {}),
    (BayesianEstimator, {"prior_type": "BDeu", "equivalent_sample_size": 10}),
])
def test_learn_returns_all_nodes(synthetic_df, estimator_cls, kwargs):
    cpds = learn(synthetic_df, estimator_cls, **kwargs)
    assert set(cpds.keys()) == set(STATES.keys())


@pytest.mark.parametrize("estimator_cls,kwargs", [
    (MaximumLikelihoodEstimator, {}),
    (BayesianEstimator, {"prior_type": "BDeu", "equivalent_sample_size": 10}),
])
def test_learned_cpts_are_valid_distributions(synthetic_df, estimator_cls, kwargs):
    """Столбцы каждой обученной CPT должны суммироваться в 1 (это же самое
    проверяет model.check_model() для экспертных CPT в test_network_pgmpy.py).
    """
    import numpy as np
    cpds = learn(synthetic_df, estimator_cls, **kwargs)
    for node, cpd in cpds.items():
        values = cpd.get_values()
        col_sums = np.asarray(values).sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-6), f"{node}: столбцы CPT не суммируются в 1"


def test_sample_efficiency_experiment_runs():
    from src.learning import sample_efficiency_experiment
    df = sample_efficiency_experiment(sizes=(20, 200), seed=3)
    assert set(df["n"]) == {20, 200}
    assert set(df["method"]) == {"MLE", "Bayesian (BDeu)"}
    # хотя бы часть запусков должна дать численный результат, а не сплошные None
    assert df["max_abs_diff"].notna().sum() >= 2
