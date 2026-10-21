"""Training/eval progress + ETA.  python status.py"""
import glob, os, re, time
now = time.time()
TOTAL, ORDER = 1_003_520, [f"{m}_s{s}" for m, ss in [("no_direction", [42, 1, 2, 3, 4]), ("full", [1, 2, 3, 4]),
    ("no_direction_noisy", [42, 1, 2]), ("no_visual", [42, 1, 2])] for s in ss]
steps = {}
for r in ORDER:
    d = f"runs/{r}"
    if os.path.exists(f"{d}/final.zip"):
        steps[r] = TOTAL
    else:
        ck = [int(re.search(r"_(\d+)_steps", f).group(1)) for f in glob.glob(f"{d}/checkpoints/*.zip")]
        lg = f"runs/logs/{r}.log"
        cur = 0
        if os.path.exists(lg):
            t = re.findall(r"total_timesteps\s*\|\s*(\d+)", open(lg, errors="ignore").read().replace("\r", "\n"))
            cur = int(t[-1]) if t else 0
        steps[r] = cur
active = [r for r in ORDER if 0 < steps[r] < TOTAL]
done = [r for r in ORDER if steps[r] >= TOTAL]
# rate: from checkpoint mtimes of active runs
rates = []
for r in active:
    f = sorted(glob.glob(f"runs/{r}/checkpoints/*.zip"), key=os.path.getmtime)
    if len(f) >= 3:
        rates.append(50000 * (len(f) - 1) / (os.path.getmtime(f[-1]) - os.path.getmtime(f[0])))
rate = sum(rates) / len(rates) if rates else 240.0   # steps/s per run
par = max(len(active), 1)
remaining = sum(TOTAL - v for v in steps.values())
eta_h = remaining / (rate * par) / 3600
print(time.strftime("%H:%M:%S"), f"| done {len(done)}/15, active {active}")
for r in active:
    print(f"  {r}: {steps[r]:,}/{TOTAL:,} ({100*steps[r]/TOTAL:.0f}%)  ~{rate:.0f} steps/s")
print(f"  remaining {remaining/1e6:.1f}M steps -> training ETA ~{eta_h:.1f} h ({par} parallel)")
ev = glob.glob("results/raw/ppo_*__*.json")
print(f"  eval result files: {len(ev)}; selection files: {len(glob.glob('results/raw/selection__*'))}")
for lg in glob.glob("runs/logs/eval_*.log"):
    tail = open(lg, errors="ignore").read().strip().splitlines()[-1:] 
    print(f"  {os.path.basename(lg)}: {tail[0][:90] if tail else '(running val checkpoints)'}")
