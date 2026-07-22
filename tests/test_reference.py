"""
Тесты референсной реализации (numpy-only, без pgmpy).
Запуск: pytest tests/test_reference.py -v
"""
import numpy as np
import pytest

from src.network_spec import SCENARIOS_TABLE_3_6, TARGET_HIGH_STATE, STATES
from src.reference_inference import posterior, joint_prob, VARS


def test_prior_distributions_sum_to_one():
    post = posterior({}, targets=["failure_prob", "maintenance", "anomaly"])
    for t, dist in post.items():
        assert np.isclose(dist.sum(), 1.0), f"{t} posterior does not sum to 1"


def test_healthy_system_is_low_risk():
    """Новый прибор, свежая поверка, всё в норме -> низкий риск по всем узлам."""
    evidence = {"pressure": "norma", "temperature": "norma", "flow": "norma",
                "calibration": "v_norme", "age": "lt_3"}
    post = posterior(evidence, targets=["failure_prob", "maintenance"])
    assert post["failure_prob"][STATES["failure_prob"].index("высокая")] < 0.15
    assert post["maintenance"][STATES["maintenance"].index("срочное")] < 0.15


# Допуски по узлам. failure_prob шире остальных: в дипломе он резко
# растёт от одного лишь возраста прибора и просроченной поверки (сценарий
# 3), а значит в HUGIN у него, вероятнее всего, есть ещё связи помимо
# опосредования через device_cond, которых нет в текстовом описании §3.2.
# Это самый явный TODO по калибровке -- см. README, "Что доделать за неделю".
TOLERANCE = {"failure_prob": 0.35, "maintenance": 0.15, "anomaly": 0.15}


@pytest.mark.parametrize("name", list(SCENARIOS_TABLE_3_6.keys()))
def test_scenarios_match_thesis_table_3_6_within_tolerance(name):
    """Апостериорные вероятности близки к диплому (Табл. 3.6).

    Допуски обусловлены тем, что 4 из 6 CPT реконструированы (полные
    таблицы из HUGIN не публиковались) -- см. README, 'Происхождение чисел'.
    Цель теста -- зафиксировать НАПРАВЛЕНИЕ и порядок величины и поймать
    регрессии при будущих правках CPT, а не бит-в-бит совпадение.
    """
    spec = SCENARIOS_TABLE_3_6[name]
    post = posterior(spec["evidence"], targets=list(spec["thesis_posterior"].keys()))
    for target, thesis_p in spec["thesis_posterior"].items():
        hi_state = TARGET_HIGH_STATE[target]
        model_p = post[target][STATES[target].index(hi_state)]
        assert abs(model_p - thesis_p) <= TOLERANCE[target], (
            f"{name}/{target}: модель={model_p:.2f}, диплом={thesis_p:.2f}"
        )


def test_explaining_away_reduces_leak_probability():
    """Ключевой качественный эффект сети (§3.1, 'объяснения прочь'):
    при недостоверных показаниях вероятность утечки должна быть заметно
    ниже, чем при тех же самых наблюдаемых входах, но достоверном приборе.
    """
    base_evidence = {"pressure": "norma", "flow": "anomal_high", "temperature": "norma"}

    reliable = dict(base_evidence, calibration="v_norme", age="lt_3")
    unreliable = dict(base_evidence, calibration="istek", age="gt_7")

    p_reliable = posterior(reliable, targets=["anomaly"])["anomaly"]
    p_unreliable = posterior(unreliable, targets=["anomaly"])["anomaly"]

    leak_idx = STATES["anomaly"].index("возможная_утечка")
    assert p_unreliable[leak_idx] < p_reliable[leak_idx], (
        "Explaining-away не работает: недостоверный прибор не должен давать "
        "более высокую P(утечка), чем достоверный, при одинаковых симптомах"
    )
