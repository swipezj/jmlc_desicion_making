"""
Визуализация графа сети (замена рис. 3.1 из диплома средствами кода, а не
HUGIN GUI). Не зависит от pgmpy -- только networkx + matplotlib, поэтому
проверяется и работает без версионных сюрпризов.

Запуск: python -m src.visualization
"""
from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt

from src.network_spec import EDGES, RUSSIAN_NAME, ROOTS

TIER = {
    **{v: 0 for v in ROOTS},
    "device_cond": 1, "pipe_cond": 1, "reliability": 1,
    "failure_prob": 2, "maintenance": 2, "anomaly": 2,
}
TIER_COLOR = {0: "#4A7FA6", 1: "#3E8E7E", 2: "#C77F2B"}
TIER_LABEL = {
    0: "ВХОДНОЙ УРОВЕНЬ — наблюдаемые переменные (АИС / ЛЭРС УЧЁТ)",
    1: "ПРОМЕЖУТОЧНЫЙ УРОВЕНЬ — скрытые состояния системы",
    2: "ВЫХОДНОЙ УРОВЕНЬ — управленческие гипотезы",
}


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("temperature")  # изолированный узел, см. network_spec.py
    g.add_edges_from(EDGES)
    return g


def draw(output_path: str = "reports/figures/network_graph.png") -> None:
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    g = build_graph()
    nodes_by_tier = {0: [], 1: [], 2: []}
    for n in g.nodes():
        nodes_by_tier[TIER[n]].append(n)
    for tier in nodes_by_tier:
        nodes_by_tier[tier].sort()

    pos = {}
    for tier, nodes in nodes_by_tier.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i - (n - 1) / 2) * 2.6, -tier * 3.0)

    fig, ax = plt.subplots(figsize=(13, 8))
    for tier, color in TIER_COLOR.items():
        nx.draw_networkx_nodes(g, pos, nodelist=nodes_by_tier[tier], node_color=color,
                                node_size=3200, node_shape="s", ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=18, edge_color="#555555",
                            connectionstyle="arc3,rad=0.08", width=1.4)
    labels = {n: RUSSIAN_NAME[n].replace(" ", "\n", 1) for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color="white", ax=ax)

    for tier, y in [(0, 1.6), (1, -1.4), (2, -4.4)]:
        ax.text(0, y, TIER_LABEL[tier], ha="center", fontsize=9, style="italic", color="#333333")

    ax.set_title("Граф байесовской сети — предиктивное ТО инженерных систем\n"
                  "(воспроизводит структуру §3.2 диплома; изолированный узел temperature — известное ограничение)",
                  fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Граф сохранён -> {output_path}")
    print(f"Узлов: {g.number_of_nodes()}, рёбер: {g.number_of_edges()}, "
          f"изолированных узлов: {list(nx.isolates(g))}")


if __name__ == "__main__":
    draw()
