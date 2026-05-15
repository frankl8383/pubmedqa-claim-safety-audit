"""
Plot the hypothesis-first autoresearch audit workflow.

Usage:
    uv run python analysis/plot_hypothesis_first_workflow.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "figures" / "hypothesis_first_workflow"


def save_all(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 8.5

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.06, 0.84)

    main_color = "#E8F1FA"
    audit_color = "#F7F1E1"
    stop_color = "#F8E6E7"
    edge = "#2B3A42"

    def box(x: float, y: float, w: float, h: float, text: str, face: str, lw: float = 1.2) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.18)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color="#455A64",
            )
        )

    steps = [
        ("Agent prior\nknowledge", 0.04, 0.68, 0.12, 0.13, main_color),
        ("Hypothesis\nexperiment card", 0.20, 0.68, 0.14, 0.13, main_color),
        ("5-min locked\ntraining run", 0.39, 0.68, 0.13, 0.13, main_color),
        ("Repeated-seed\nvolatility gate", 0.56, 0.68, 0.14, 0.13, audit_color),
        ("Prediction-level\nclass audit", 0.75, 0.68, 0.14, 0.13, audit_color),
    ]
    for text, x, y, w, h, color in steps:
        box(x, y, w, h, text, color)
    for i in range(len(steps) - 1):
        _, x, y, w, h, _ = steps[i]
        _, x2, y2, w2, h2, _ = steps[i + 1]
        arrow(x + w + 0.01, y + h / 2, x2 - 0.01, y2 + h2 / 2)

    lower = [
        ("Proxy-metric\nmismatch gate", 0.17, 0.38, 0.16, 0.12, audit_color),
        ("Context-use\nsanity check", 0.39, 0.38, 0.15, 0.12, audit_color),
        ("Calibration\nsanity check", 0.60, 0.38, 0.15, 0.12, audit_color),
        ("Claim-safety\nmatrix", 0.81, 0.38, 0.14, 0.12, audit_color),
    ]
    for text, x, y, w, h, color in lower:
        box(x, y, w, h, text, color)
    arrow(0.82, 0.68, 0.25, 0.51)
    for i in range(len(lower) - 1):
        _, x, y, w, h, _ = lower[i]
        _, x2, y2, w2, h2, _ = lower[i + 1]
        arrow(x + w + 0.01, y + h / 2, x2 - 0.01, y2 + h2 / 2)

    box(
        0.05,
        0.12,
        0.22,
        0.13,
        "Builder role\nproposes one code change\nin train_biomed.py",
        "#EEF7EE",
    )
    box(
        0.38,
        0.12,
        0.24,
        0.13,
        "Auditor role\nchecks locks, seeds,\ncontrols, and trade-offs",
        "#EEF7EE",
    )
    box(
        0.73,
        0.12,
        0.21,
        0.13,
        "Forbidden claims\nno agent superiority;\nno clinical utility",
        stop_color,
    )
    arrow(0.27, 0.185, 0.38, 0.185)
    arrow(0.62, 0.185, 0.73, 0.185)

    save_all(fig, OUTDIR, "hypothesis_first_workflow")
    readme = """# Hypothesis-First Workflow Figure

This figure summarizes the audit workflow added after the initial PubMedQA
autoresearch experiments. It should be used as the main workflow figure when
the manuscript is positioned as a biomedical informatics claim-safety audit.
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote hypothesis-first workflow figure to {OUTDIR}")


if __name__ == "__main__":
    main()
