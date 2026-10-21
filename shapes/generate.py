"""
Procedural closed-contour generator (radial Fourier series).

    r(theta) = R0 * (1 + sum_{k=2}^{K} a_k cos(k*theta + phi_k))

  K = 7 harmonics (k = 2..7); k = 1 omitted (it only shifts the centre)
  a_k ~ U(0, 0.40/k)          amplitudes decay with frequency
  phi_k ~ U(0, 2*pi)
  R0 ~ U(150, 190) px on a 512x512 canvas (path1.png: ~180-230 px extent)
  accept only if min r >= 0.45*R0 (no pinched shapes) and max extent fits the canvas with a 20 px margin
  stroke: black, 5 px, anti-aliased (matches path1.png), white background, centre = canvas centre + U(-12,12)px
r(theta) is single-valued so the curve never self-intersects.

Seeds (disjoint ranges):  train bank : 1000..1099   (train_000..train_099)
                          held-out   : 9000..9009   (heldout00..heldout09)
Generation is deterministic (numpy default_rng(seed) + cv2 drawing).

NOTE: path1-3 were hand-drawn by the author, so they are NOT reproducible from this generator; the PNGs in
target_images/ remain the reference.  These shapes only mimic path1's style (closed star-like blob).

    python -m shapes.generate
"""
import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K, SIZE, STROKE = 7, 512, 5
TRAIN_SEEDS = range(1000, 1100)
HELDOUT_SEEDS = range(9000, 9010)


def shape_points(seed, n=720):
    rng = np.random.default_rng(seed)
    while True:
        R0 = rng.uniform(150, 190)
        ks = np.arange(2, K + 1)
        a = rng.uniform(0, 0.40 / ks)
        ph = rng.uniform(0, 2 * np.pi, len(ks))
        c = SIZE / 2 + rng.uniform(-12, 12, 2)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = R0 * (1 + (a[None] * np.cos(ks[None] * th[:, None] + ph[None])).sum(1))
        pts = np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)])
        if r.min() >= 0.45 * R0 and pts.min() >= 20 and pts.max() <= SIZE - 20:
            return pts


def render(seed):
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    cv2.polylines(img, [np.round(shape_points(seed)).astype(np.int32).reshape(-1, 1, 2)], True,
                  (20, 20, 20), STROKE, cv2.LINE_AA)
    return img


def main():
    for sub, seeds, pre, w in (("train", TRAIN_SEEDS, "train_", 3), ("heldout", HELDOUT_SEEDS, "heldout", 2)):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
        for i, s in enumerate(seeds):
            cv2.imwrite(os.path.join(HERE, sub, f"{pre}{i:0{w}d}.png"), render(s))
    print("wrote", len(TRAIN_SEEDS), "train +", len(HELDOUT_SEEDS), "held-out shapes")


if __name__ == "__main__":
    main()
