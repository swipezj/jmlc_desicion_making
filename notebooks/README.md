# Ноутбуки

Вся логика проекта живёт в `src/` и покрыта тестами; ноутбуки нужны как
витрина, а не как источник истины, поэтому появятся последними.

Планируется:
- `01_network_and_validation.ipynb` — сборка сети, проверка против Табл. 3.6, explaining away
- `02_decision_layer.ipynb` — матрица потерь, компромисс «ложные выезды / пропуски»
- `03_calibration.ipynb` — разбор гипотезы о недостающей дуге у `failure_prob`
- `04_learning_and_baselines.ipynb` — обучение параметров и структуры, сравнение с ML

Пока то же самое доступно как CLI: `python -m src.decision`, `python -m src.sensitivity`,
`python -m src.calibration`, `python -m src.structure_learning`, `python -m src.baselines`.
