"""The 2023 webcam loop, made portable and a little more honest.

origin/Main.py loaded '/home/fh1m/cascade.xml', a path that existed on exactly
one laptop. This uses the cascade in this repo, prints the frame rate, and
can loosen the neighbour vote so you can watch it find faces in your furniture.

    python -m pareidolia.live                  # normal
    python -m pareidolia.live --pareidolia     # minNeighbors 5 -> 2: see faces everywhere
    python -m pareidolia.live --image photo.jpg --out boxed.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2

from .run import CASCADE, detector


def boxes(det, frame, neighbours):
    g = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=neighbours, minSize=(30, 30))


def draw(frame, found, label):
    for (x, y, w, h) in found:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pareidolia", action="store_true", help="loosen the vote: minNeighbors 2")
    ap.add_argument("--image")
    ap.add_argument("--out")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    det = detector()
    n = 2 if a.pareidolia else 5
    print(f"cascade: {CASCADE.name}, minNeighbors={n}")

    if a.image:
        img = cv2.imread(a.image)
        found = boxes(det, img, n)
        print(f"{len(found)} face(s)")
        if a.out:
            cv2.imwrite(a.out, draw(img, found, f"{len(found)} face(s)"))
        return

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit(f"no camera at index {a.camera}")
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        found = boxes(det, frame, n)
        frames += 1
        fps = frames / max(time.time() - t0, 1e-6)
        cv2.imshow("pareidolia", draw(frame, found, f"{len(found)} face(s)  {fps:4.1f} fps"))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
