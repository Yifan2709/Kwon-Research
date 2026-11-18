"""Plot newly calculated internal-state diagnostics; never load archive data implicitly."""
import argparse
import csv
from pathlib import Path

from .output import setup_plot
from .parameters import Parameters, ParameterError


def read_columns(path, columns):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(name not in rows[0] for name in columns):
        raise ParameterError(f"{path} must contain newly calculated columns: {', '.join(columns)}")
    return [[float(row[name]) for row in rows] for name in columns]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--coherence-csv", type=Path, default=Path("outputs/coherence_scan.csv"))
    parser.add_argument("--depumping-csv", type=Path, default=Path("outputs/depumping.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/internal_scan.png"))
    args = parser.parse_args(argv)
    try:
        config = Parameters(args.config_dir)
        cx, events = read_columns(args.coherence_csv, (
            "quantization_voltage_V", "prob_emitted_at_least_one_noncycling_photon"))
        dx, dark = read_columns(args.depumping_csv, (
            "quantization_voltage_V", "p_lower_F_from_initial_with_dephased_trap"))
        plt = setup_plot(config)
        fig, axes = plt.subplots(2, 1, sharex=True, layout="constrained",
                                 figsize=config.get("numerics.plot.figure_size_inches"))
        axes[0].plot(cx, events)
        axes[0].set_ylabel("At least one noncycling photon")
        axes[1].plot(dx, dark)
        axes[1].set_ylabel("Lower-F probability (dephased model)")
        axes[1].set_xlabel("Quantization-field control voltage (V)")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=config.number("numerics.plot.dpi"))
        plt.close(fig)
        print(args.output)
    except (ParameterError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
