import os, ast, shutil, numpy as np, pandas as pd
from PIL import Image

ROOT   = os.path.expanduser("~/MICCAI2026/FoundUS/labeled_only")
CSVDIR = os.path.join(ROOT, "csv")
FG=18

def pts_cols(row):
    return [c for c in row.index if c.endswith("_xy")
            and isinstance(row[c],str) and row[c].strip().startswith("[")]

def find_div(a,w):
    prof=(a>FG).mean(axis=0); k=max(3,int(w*0.02))
    prof=np.convolve(prof,np.ones(k)/k,mode="same")
    lo,hi=int(w*0.35),int(w*0.65)
    return lo+int(np.argmin(prof[lo:hi]))

def crop_side(im, div, side):
    w,h=im.size
    box=(0,0,div,h) if side=="left" else (div,0,w,h)
    return im.crop(box), box[0]

def process(task, decide):
    src_csv=os.path.join(CSVDIR,f"{task}_train.csv")
    df=pd.read_csv(src_csv)
    shutil.copy(src_csv, os.path.join(CSVDIR,f"{task}_train.orig.csv"))
    outdir=os.path.join(ROOT,f"{task}_cropped"); os.makedirs(outdir,exist_ok=True)
    n_crop=0
    for i,row in df.iterrows():
        rel=row["image_path"]; src=os.path.join(ROOT,rel)
        if not os.path.exists(src): continue
        im=Image.open(src).convert("RGB"); w,h=im.size
        a=np.asarray(im.convert("L"),dtype=np.float32)
        div=find_div(a,w)
        action,side=decide(rel,row,div,a,w)
        newrel=f"{task}_cropped/"+os.path.basename(rel)
        if action=="crop":
            cim,x0=crop_side(im,div,side); cim.save(os.path.join(ROOT,newrel))
            for c in pts_cols(row):
                x,y=ast.literal_eval(row[c]); df.at[i,c]=str([round(x-x0,2),round(y,2)])
            df.at[i,"width"]=cim.size[0]; n_crop+=1
        else:
            im.save(os.path.join(ROOT,newrel))
        df.at[i,"image_path"]=newrel
    df.to_csv(src_csv,index=False)
    print(f"{task}: cropped {n_crop}/{len(df)}, rewrote CSV, backup=*.orig.csv")

# IVC: obey hand-checked triage VERDICT
tri=pd.read_csv(os.path.join(CSVDIR,"IVC_triage.csv")).set_index("image_path")
def ivc_decide(rel,row,div,a,w):
    if rel in tri.index and str(tri.loc[rel,"VERDICT"]).strip()=="crop":
        side=str(tri.loc[rel,"keep_side"]).strip() or ("right" if div<w/2 else "left")
        if side not in ("left","right"): side="right" if div<w/2 else "left"
        return "crop",side
    return "keep",None

# PSAX: auto (passed clean)
def psax_decide(rel,row,div,a,w):
    prof=(a>FG).mean(axis=0); k=max(3,int(w*0.02))
    prof=np.convolve(prof,np.ones(k)/k,mode="same")
    lo,hi=int(w*0.35),int(w*0.65); gap=prof[div]
    left=prof[:int(w*0.30)].mean(); right=prof[int(w*0.70):].mean()
    darkw=int(np.sum(prof[lo:hi]<0.15))
    xs=[ast.literal_eval(row[c])[0] for c in pts_cols(row)]
    both=(min(xs)<div<max(xs)) if xs else False
    near=any(abs(x-div)<0.06*w for x in xs)
    if gap<0.12 and left>0.10 and right>0.10 and darkw<int(w*0.18) and not both and not near:
        return "crop",("right" if np.mean(xs)>div else "left")
    return "keep",None

process("IVC", ivc_decide)
process("PSAX", psax_decide)
print("Done. Spot-check IVC_cropped/, then set NUM_EPOCHS=250 and queue behind PENGWIN.")
