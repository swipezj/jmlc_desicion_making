"""
Проверка референсной реализации (src/reference_inference.py) на трёх
сценариях из Таблицы 3.6 диплома. Запуск: `python -m scripts.verify_reference`
из корня репозитория.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.network_spec import SCENARIOS_TABLE_3_6, TARGET_HIGH_STATE, STATES
from src.reference_inference import posterior


def main():
    print("=" * 70)
    print("Сравнение с Таблицей 3.6 диплома (реконструкция vs. HUGIN)")
    print("=" * 70)
    for name, spec in SCENARIOS_TABLE_3_6.items():
        ev = spec["evidence"]
        thesis = spec["thesis_posterior"]
        post = posterior(ev, targets=list(thesis.keys()))
        print(f"\n{name}")
        print(f"  свидетельство: {ev}")
        for t, thesis_p in thesis.items():
            hi = TARGET_HIGH_STATE[t]
            model_p = post[t][STATES[t].index(hi)]
            flag = "OK " if abs(model_p - thesis_p) <= 0.10 else "gap"
            print(f"  [{flag}] {t:12s} P({hi})={model_p:.2f}  диплом={thesis_p:.2f}  Δ={model_p - thesis_p:+.2f}")

    print("\n" + "-" * 70)
    print("[gap] = реконструированная CPT пока не дотягивает до точного значения")
    print("        из диплома -- ожидаемо, т.к. полные CPT из HUGIN не были")
    print("        опубликованы в тексте работы. См. README, 'Происхождение чисел'.")


if __name__ == "__main__":
    main()
