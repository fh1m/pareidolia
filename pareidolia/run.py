"""Open up a 2001 face detector and see what it actually looks for.

1. Draw the cascade's first-stage Haar features on a real face.
2. Rotate the face and find where the detector stops believing in it.
3. Point it at things that are not faces and collect what it sees anyway.

    python -m pareidolia.run          # prints findings, writes assets/*.png
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage.data as sk

ROOT = Path(__file__).resolve().parent.parent
CASCADE = ROOT / "cascade.xml"  # OpenCV's haarcascade_frontalface_default (Lienhart)
ASSETS = ROOT / "assets"
ANGLES = np.arange(-60, 61, 2)

# Things with no faces in them. (skimage ships these, so no downloads.)
NOT_FACES = ["coffee", "rocket", "coins", "clock", "brick", "gravel", "grass",
             "moon", "hubble_deep_field", "immunohistochemistry", "horse", "page"]


def detector() -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(str(CASCADE))
    assert not c.empty(), "cascade failed to load"
    return c


def gray(img: np.ndarray) -> np.ndarray:
    from skimage.util import img_as_ubyte
    img = img_as_ubyte(img)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY)


def first_stage_features() -> list[list[tuple[float, ...]]]:
    """Return the rectangles of the features used by stage 0 (24x24 window)."""
    root = ET.parse(CASCADE).getroot()
    casc = root.find("cascade")
    feats = casc.find("features")
    all_rects = [[tuple(float(v) for v in r.text.split()) for r in f.find("rects")] for f in feats]
    stage0 = casc.find("stages")[0]
    idx = [int(w.find("internalNodes").text.split()[2]) for w in stage0.find("weakClassifiers")]
    return [all_rects[i] for i in idx]


def detect(det, g, neighbours=5):
    return det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=neighbours, minSize=(24, 24))


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    det = detector()
    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 10})

    # --- 1. what stage 0 looks at -------------------------------------------
    face_rgb = sk.astronaut()
    g = gray(face_rgb)
    (x, y, w, h) = max(detect(det, g), key=lambda r: r[2] * r[3])
    crop = cv2.resize(g[y:y + h, x:x + w], (240, 240), interpolation=cv2.INTER_AREA)
    feats = first_stage_features()
    fig, axes = plt.subplots(1, len(feats), figsize=(2.0 * len(feats), 2.3), dpi=150)
    for ax, rects in zip(axes, feats):
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        for rx, ry, rw, rh, wt in rects:
            colour = "#f2efe6" if wt > 0 else "#14110f"
            ax.add_patch(plt.Rectangle((rx * 10, ry * 10), rw * 10, rh * 10,
                                       facecolor=colour, alpha=0.72, edgecolor="none"))
        ax.axis("off")
    fig.suptitle("Stage 0: nine light-vs-dark comparisons on a 24×24 window", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ASSETS / "stage0_features.png")
    plt.close(fig)

    # --- 2. rotation tolerance ---------------------------------------------
    # Rotate about the face's own centre, pad with flat grey (a reflected
    # border would mirror the face and hand the detector free copies), and
    # only count a detection that sits on the real face at a plausible size.
    hits = []
    pad = cv2.copyMakeBorder(g, 128, 128, 128, 128, cv2.BORDER_CONSTANT, value=int(g.mean()))
    cx, cy = x + w / 2 + 128, y + h / 2 + 128
    for a in ANGLES:
        m = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(pad, m, pad.shape[::-1], borderMode=cv2.BORDER_CONSTANT,
                             borderValue=int(g.mean()))
        hits.append(any(abs(dx + dw / 2 - cx) < w / 3 and abs(dy + dh / 2 - cy) < h / 3
                        and 0.5 * w < dw < 2 * w for dx, dy, dw, dh in detect(det, rot)))
    hits = np.array(hits)
    print("tilts where the face was found:", [int(a) for a in ANGLES[hits]])
    ok = ANGLES[hits]
    fig, ax = plt.subplots(figsize=(8, 1.9), dpi=150)
    ax.bar(ANGLES, np.where(hits, 1, 0.12), width=1.7, color=np.where(hits, "#3f6b2a", "#b3261e"))
    ax.set_yticks([])
    ax.set_xlabel("in-plane head tilt (degrees)")
    ax.set_title("found the face?  green = yes", fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "rotation.png")
    plt.close(fig)

    # --- 3. pareidolia ------------------------------------------------------
    found = []
    for name in NOT_FACES:
        img = getattr(sk, name)()
        gg = gray(img)
        for (fx, fy, fw, fh) in detect(det, gg, neighbours=3):
            patch = cv2.resize(gg[fy:fy + fh, fx:fx + fw], (160, 160), interpolation=cv2.INTER_AREA)
            found.append((name, int(fw), patch))
    if found:
        n = min(len(found), 8)
        fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.3), dpi=150)
        axes = np.atleast_1d(axes)
        for ax, (name, size, patch) in zip(axes, found[:n]):
            ax.imshow(patch, cmap="gray")
            ax.set_title(name.replace("_", " "), fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(ASSETS / "pareidolia.png")
        plt.close(fig)

    print(f"stage 0 uses {len(feats)} features")
    print(f"face found from {ok.min()}° to {ok.max()}° tilt; misses inside that band: "
          f"{[int(a) for a in ANGLES[(ANGLES >= ok.min()) & (ANGLES <= ok.max()) & ~hits]]}")
    print(f"pareidolia (minNeighbors=3) over {len(NOT_FACES)} face-free images: {len(found)} 'faces'")
    for name, size, _ in found:
        print(f"  {name}: {size}px")


if __name__ == "__main__":
    main()
