"""Shared output helpers. Saved settings accompany every new calculation."""
import csv
import json
from pathlib import Path


def write_rows(path, rows, config):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError("No calculated rows to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    snapshot = path.with_suffix(".parameters.json")
    snapshot.write_text(json.dumps(config.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def setup_plot(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = config.get("numerics.plot")
    plt.rcParams.update({"font.family": settings["font_family"],
                         "font.size": settings["font_size"],
                         "axes.unicode_minus": False})
    return plt
