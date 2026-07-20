import os, numpy as np, pandas as pd
from PIL import Image

ROOT   = os.path.expanduser("~/MICCAI2026/FoundUS/labeled_only")
CSVDIR = os.path.join(ROOT, "csv")
TASKS  = ["PSAX", "IVC", "A4C", "PLAX"]

CENTER_BAND  = 0.10   # central +/-5% of width = "divider zone"
FLANK_FRAC   = 0.25   # each lobe measured over the outer 25%
FG_THRESH    = 15     # grayscale > this = ultrasound content (bg is ~black)
VALLEY_RATIO = 0.45   # centre < 0.45*flank -> dark divider -> split
MIN_FLANK    = 0.05   # both flanks need >5% content -> genuinely two lobes

def score(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    h, w = a.shape
    col = (a > FG_THRESH).astype(np.float32).mean(axis=0)   # per-column content
    k = max(3, int(w*0.02))
    col = np.convolve(col, np.ones(k)/k, mode="same")
    c0, c1 = int(w*(0.5-CENTER_BAND/2)), int(w*(0.5+CENTER_BAND/2))
    center = col[c0:c1].mean()
    left   = col[:int(w*FLANK_FRAC)].mean()
    right  = col[int(w*(1-FLANK_FRAC)):].mean()
    flank  = (left + right)/2.0
    ratio  = center/flank if flank > 0 else 1.0
    is_split = (flank > 0 and center < VALLEY_RATIO*flank
                and left > MIN_FLANK and right > MIN_FLANK)
    return bool(is_split), round(float(ratio), 3)

for t in TASKS:
    df = pd.read_csv(os.path.join(CSVDIR, f"{t}_train.csv"))
    res = [score(os.path.join(ROOT, p)) for p in df["image_path"]]
    df["is_split"] = [r[0] for r in res]
    df["valley_ratio"] = [r[1] for r in res]
    ns, n = int(df["is_split"].sum()), len(df)
    print(f"{t}: {ns}/{n} flagged split  ->  keeps {n-ns}")
    df.to_csv(os.path.join(CSVDIR, f"{t}_train_flagged.csv"), index=False)
    (df[~df["is_split"]].drop(columns=["is_split","valley_ratio"])
        .to_csv(os.path.join(CSVDIR, f"{t}_train_singleview.csv"), index=False))
    srt = df.sort_values("valley_ratio")
    print("   most split-like:", list(srt["image_path"].head(4)))
    print("   least split-like:", list(srt["image_path"].tail(4)))
