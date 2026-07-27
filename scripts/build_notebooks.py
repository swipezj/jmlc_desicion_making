"""
Сборка ноутбуков из исходников-повествований.

ПОЧЕМУ ТАК, А НЕ ПРОСТО .ipynb В РЕПОЗИТОРИИ. Ноутбук -- плохой носитель
истории изменений: JSON с вперемешку кодом, выводами и метаданными даёт
нечитаемые диффы и конфликты при слиянии. Здесь принят подход «ноутбук как
артефакт сборки»: повествование живёт в `notebooks/sources/*.py` в виде
обычного Python с разметкой ячеек, а `.ipynb` собирается этим скриптом --
вместе с уже выполненными выводами и картинками, чтобы результаты были
видны прямо на GitHub, без запуска.

Тот же приём известен как jupytext; здесь он реализован без внешних
зависимостей, чтобы сборка работала в CI, где jupyter не установлен.

Разметка в исходниках:
    # %% [markdown]
    # текст ячейки (каждая строка -- с решёткой)

    # %%
    код ячейки

Запуск:  python -m scripts.build_notebooks
Проверка без перезаписи:  python -m scripts.build_notebooks --check
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = ROOT / "notebooks" / "sources"
OUT = ROOT / "notebooks"
MAX_TEXT_CHARS = 12000


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Разобрать размеченный Python на список (тип ячейки, исходник)."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))

    out = []
    for kind, lines in cells:
        if kind == "markdown":
            body = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                             for ln in lines)
        else:
            body = "\n".join(lines)
        body = body.strip("\n")
        if body.strip():
            out.append((kind, body))
    return out


def _capture_figures() -> list[str]:
    """Снять все открытые matplotlib-фигуры как base64 PNG и закрыть их."""
    import matplotlib.pyplot as plt
    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        plt.close(fig)
    return images


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    head = text[: MAX_TEXT_CHARS // 2]
    tail = text[-MAX_TEXT_CHARS // 2:]
    return f"{head}\n\n[... вывод усечён при сборке ноутбука ...]\n\n{tail}"


def execute(cells: list[tuple[str, str]], name: str) -> list[dict]:
    """Выполнить кодовые ячейки в общем пространстве имён, собрав выводы."""
    import matplotlib
    matplotlib.use("Agg")

    namespace: dict = {"__name__": "__notebook__"}
    result: list[dict] = []
    counter = 0
    for kind, source in cells:
        if kind == "markdown":
            result.append({"cell_type": "markdown", "metadata": {},
                           "source": source.splitlines(keepends=True)})
            continue
        counter += 1
        stdout = io.StringIO()
        outputs: list[dict] = []
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(source, f"<{name}:{counter}>", "exec"), namespace)
        except Exception:
            outputs.append({
                "output_type": "stream", "name": "stderr",
                "text": traceback.format_exc().splitlines(keepends=True),
            })
            print(f"  ! ячейка {counter} упала:\n{traceback.format_exc()}",
                  file=sys.stderr)
        text = stdout.getvalue()
        if text.strip():
            outputs.insert(0, {"output_type": "stream", "name": "stdout",
                               "text": _truncate(text).splitlines(keepends=True)})
        for png in _capture_figures():
            outputs.append({"output_type": "display_data",
                            "data": {"image/png": png}, "metadata": {}})
        result.append({
            "cell_type": "code", "execution_count": counter,
            "metadata": {}, "outputs": outputs,
            "source": source.splitlines(keepends=True),
        })
    return result


def build(path: pathlib.Path) -> dict:
    cells = parse_cells(path.read_text(encoding="utf-8"))
    return {
        "cells": execute(cells, path.stem),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="только проверить, что все ячейки выполняются без ошибок")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    failures = 0
    for src in sorted(SOURCES.glob("*.py")):
        print(f"Сборка {src.name} ...")
        nb = build(src)
        failed = sum(1 for c in nb["cells"] if c["cell_type"] == "code"
                     and any(o.get("name") == "stderr" for o in c["outputs"]))
        failures += failed
        if not args.check:
            target = OUT / f"{src.stem}.ipynb"
            target.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            print(f"  -> {target.relative_to(ROOT)} "
                  f"({len(nb['cells'])} ячеек, ошибок: {failed})")
        else:
            print(f"  проверено, ошибок: {failed}")
    if failures:
        print(f"\nОШИБКА: ячеек с исключениями -- {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
