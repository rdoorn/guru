"""Render benchmark results as speed-vs-accuracy scatter plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # headless backend
import matplotlib.pyplot as plt             # noqa: E402


def points(records: list, x_key: str) -> list:
    """[(x, accuracy, 'model (ctx)')] for records with a numeric accuracy."""
    out = []
    for r in records:
        acc = r.get('accuracy')
        if acc is None:
            continue
        out.append((r.get(x_key, 0), acc,
                    f"{r['model']} ({r.get('num_ctx', '?')})"))
    return out


def _scatter(records: list, x_key: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for x, y, label in points(records, x_key):
        ax.scatter(x, y)
        ax.annotate(label, (x, y), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('accuracy (0-100)')
    ax.set_ylim(0, 100)
    ax.set_title('guru coding-model benchmark')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render(results_path, out_dir=Path('bench')) -> list:
    """Write two scatter PNGs (tokens/sec and wall-time) and return paths."""
    records = json.loads(Path(results_path).read_text(encoding='utf-8'))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a = out_dir / 'speed_tokens.png'
    b = out_dir / 'speed_walltime.png'
    _scatter(records, 'tokens_per_sec', 'speed (tokens/sec)', a)
    _scatter(records, 'seconds', 'wall-time (seconds, lower=faster)', b)
    return [a, b]


if __name__ == '__main__':
    import sys
    render(Path(sys.argv[1]))
