"""Loading + aggregation shared by all analyses.  Everything reads results/raw/*.json only."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RAW = os.path.join(ROOT, "results", "raw")
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")
for d in (TABLES, FIGS):
    os.makedirs(d, exist_ok=True)

TIER_ORDER = ["full", "no_direction", "no_direction_noisy", "no_visual"]
HELDOUT = [f"heldout{i:02d}" for i in range(10)]


def load(method=None, tier=None, target=None, split="test", prefix=None):
    """All raw result dicts matching the filters (method exact, or prefix)."""
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "*__*__s*__*__*.json"))):
        m, t, s, tg, sp = os.path.basename(f)[:-5].split("__")
        if split and sp != split: continue
        if method and m != method: continue
        if prefix and not m.startswith(prefix): continue
        if tier and t != tier: continue
        if target and tg != target: continue
        out.append(json.load(open(f)))
    return out


def seed_means(results, key):
    """One value per result (= per seed for learned methods): summary[key]."""
    return np.array([r["summary"][key] for r in results], dtype=float)


def boot_ci(x, n=10000, seed=0):
    """95% percentile bootstrap CI of the mean over seed means."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def agg(results, key):
    x = seed_means(results, key)
    lo, hi = boot_ci(x)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
                lo=lo, hi=hi)


def fmt(a, d=1):
    if a["n"] < 2:
        return f"{a['mean']:.{d}f}"
    return f"{a['mean']:.{d}f} $\\pm$ {a['sd']:.{d}f} [{a['lo']:.{d}f}, {a['hi']:.{d}f}]"


def write(name, text):
    p = os.path.join(TABLES, name)
    open(p, "w").write(text)
    return p
