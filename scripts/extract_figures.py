"""Extract the plots embedded in the executed notebooks into paper/figures/*.pdf.

The paper includes figures/<name>.pdf; the notebooks display their plots inline
rather than saving them, so this script pulls the PNG outputs out of the .ipynb
files (run the notebooks first) and writes them under the names the paper expects.
Two paper figures combine two notebook plots side by side.

Usage:  python scripts/extract_figures.py
"""
import base64, io, json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

# paper figure -> list of (notebook, code-cell index) panels, left to right
FIGURES = {
    "fig_bandit_rewards": [("bandit", 3)],
    "fig_bandit_kl":      [("bandit", 7)],
    "fig_bandit_exact":   [("bandit", 12)],
    "fig_bandit_sampled": [("bandit", 15)],
    "fig_bandit_rescale": [("bandit", 19), ("bandit", 21)],
    "fig_bandit_refarm":  [("bandit", 23)],
    "fig_grid_layout":    [("gridworld", 3), ("gridworld", 9)],
    "fig_grid_exact":     [("gridworld", 12)],
    "fig_grid_rescale":   [("gridworld", 18)],
    "fig_grid_sampled":   [("gridworld", 21)],
}
NOTEBOOKS = {
    "bandit": ROOT / "notebooks" / "natural_gradient_bandit.ipynb",
    "gridworld": ROOT / "notebooks" / "natural_gradient_gridworld.ipynb",
}


def cell_image(nb, idx):
    for out in nb["cells"][idx].get("outputs", []):
        png = out.get("data", {}).get("image/png")
        if png:
            return Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")
    raise RuntimeError(f"cell {idx} has no image output - run the notebook first")


def hstack(images, gap=20):
    h = max(im.height for im in images)
    images = [im.resize((round(im.width * h / im.height), h)) for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in images) + gap * (len(images) - 1), h), "white")
    x = 0
    for im in images:
        canvas.paste(im, (x, 0)); x += im.width + gap
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nbs = {k: json.loads(p.read_text()) for k, p in NOTEBOOKS.items()}
    for name, panels in FIGURES.items():
        img = hstack([cell_image(nbs[nb], i) for nb, i in panels])
        img.save(OUT / f"{name}.pdf", resolution=200)
        print("wrote", (OUT / f"{name}.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
