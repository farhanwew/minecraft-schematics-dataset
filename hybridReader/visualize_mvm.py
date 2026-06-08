"""
MVM Visualization — Masked Voxel Modeling visual explainer.
Follows the same pattern as hybrid_render.py (Node.js + prismarine-viewer).

Renders each schematic 3 ways side-by-side:
  1. Original  — real Minecraft textures via prismarine-viewer
  2. Masked    — 20% non-air blocks set to air (holes = masked positions)
  3. Semantic  — each unique block type gets a distinct color (PIL overlay)

Input:  data.parquet  (32x32x32 flat voxel arrays, pre-processed block IDs)
Output: mvm_viz/<idx>_<title>/  →  original.jpg, masked.jpg, semantic.png, combined.png

Requirements:
    pip install javascript pyarrow pillow numpy
    cd hybridReader && npm install
"""

import os, re, sys, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from javascript import require

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), '..')
PARQUET_FILE = os.path.join(DATA_DIR, 'data.parquet')
OUT_DIR      = os.path.join(DATA_DIR, 'mvm_viz')
MAX_SCHEMAS  = 10
IMG_SIZE     = 512        # final panel size (px)
MASK_RATIO   = 0.20       # 20% of non-air blocks get masked
VOXEL_SIZE   = 32
RENDER_ANGLE = 1          # number of angles to render per panel (1 = front-ish)
# ─────────────────────────────────────────────────────────────────────────────

print("Loading Node.js rendering engine...")
render_engine = require('./render_engine.js')

# ── Semantic colormap (PIL only, no prismarine needed) ────────────────────────
_TAB20 = [
    (31,119,180),(174,199,232),(255,127,14),(255,187,120),(44,160,44),
    (152,223,138),(214,39,40),(255,152,150),(148,103,189),(197,176,213),
    (140,86,75),(196,156,148),(227,119,194),(247,182,210),(127,127,127),
    (199,199,199),(188,189,34),(219,219,141),(23,190,207),(158,218,229),
]

def _surface_mask(occ):
    s = np.zeros_like(occ)
    s[1:,:,:]  |= ~occ[:-1,:,:]
    s[:-1,:,:] |= ~occ[1:,:,:]
    s[:,1:,:]  |= ~occ[:,:-1,:]
    s[:,:-1,:] |= ~occ[:,1:,:]
    s[:,:,1:]  |= ~occ[:,:,:-1]
    s[:,:,:-1] |= ~occ[:,:,1:]
    return occ & s

def _compute_ao(occ):
    H, L, W = occ.shape
    nc = np.zeros((H,L,W), dtype=np.float32)
    for dy,dz,dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        ss = [slice(max(0,-dy),H+min(0,-dy)), slice(max(0,-dz),L+min(0,-dz)), slice(max(0,-dx),W+min(0,-dx))]
        sd = [slice(max(0, dy),H+min(0, dy)), slice(max(0, dz),L+min(0, dz)), slice(max(0, dx),W+min(0, dx))]
        nc[tuple(sd)] += occ[tuple(ss)].astype(np.float32)
    return 1.0 - nc/6.0*0.5

def _shade(color, f):
    return tuple(max(0, min(255, int(c*f))) for c in color)

def _sky_bg(size):
    top = np.array([160,210,250], dtype=np.float32)
    bot = np.array([ 80,140,200], dtype=np.float32)
    t   = np.linspace(0,1,size,dtype=np.float32)[:,None]
    row = (top+(bot-top)*t).astype(np.uint8)
    return Image.fromarray(np.broadcast_to(row[:,None,:],(size,size,3)).copy())

def render_semantic(blocks_3d, size=IMG_SIZE, angle_deg=30):
    """PIL isometric render with distinct color per unique block type."""
    unique_ids = sorted(set(int(v) for v in blocks_3d.flat if v != 0))
    sem_map = {bid: _TAB20[i % len(_TAB20)] for i, bid in enumerate(unique_ids)}

    H, L, W = blocks_3d.shape
    occ  = blocks_3d != 0
    surf = _surface_mask(occ)
    ao   = _compute_ao(occ)
    ys, zs, xs = np.where(surf)
    if len(xs) == 0:
        return _sky_bg(size)

    block_ids = blocks_3d[ys, zs, xs]
    ao_vals   = ao[ys, zs, xs]

    cx,cy,cz = W/2.0, H/2.0, L/2.0
    bx=(xs-cx).astype(np.float32); by=(ys-cy).astype(np.float32); bz=(zs-cz).astype(np.float32)
    az=np.radians(angle_deg)
    rx= np.cos(az)*bx+np.sin(az)*bz; ry=by; rz=-np.sin(az)*bx+np.cos(az)*bz

    rs=size*2; margin=40
    ixr=(rx-rz); iyr=(rx+rz)*0.5-ry
    xs_=max(ixr.max()-ixr.min()+2,1.0); ys_=max(iyr.max()-iyr.min()+2,1.0)
    us=rs-margin*2; s=max(1.0,min(us/xs_,us/ys_))
    cxo=rs/2-((ixr.min()+ixr.max())/2)*s; cyo=rs/2-((iyr.min()+iyr.max())/2)*s-s/2
    ix=ixr*s+cxo; iy=iyr*s+cyo

    order=np.argsort(rx+rz+ry*0.001)
    img=_sky_bg(rs); draw=ImageDraw.Draw(img); ow=max(1,int(s)//8)
    for i in order:
        ox,oy=float(ix[i]),float(iy[i]); ao_f=float(ao_vals[i])
        base=sem_map.get(int(block_ids[i]),(160,120,100))
        s2=s/2
        lp=[(ox-s,oy),(ox,oy+s2),(ox,oy+s2+s),(ox-s,oy+s)]
        rp=[(ox,oy+s2),(ox+s,oy),(ox+s,oy+s),(ox,oy+s2+s)]
        tp=[(ox,oy-s2),(ox+s,oy),(ox,oy+s2),(ox-s,oy)]
        draw.polygon(lp,fill=_shade(base,0.55*ao_f))
        draw.polygon(rp,fill=_shade(base,0.75*ao_f))
        draw.polygon(tp,fill=_shade(base,1.00*ao_f))
        if s>=4:
            draw.line(lp+[lp[0]],fill=_shade(base,0.35*ao_f),width=ow)
            draw.line(rp+[rp[0]],fill=_shade(base,0.35*ao_f),width=ow)
            draw.line(tp+[tp[0]],fill=_shade(base,0.45*ao_f),width=ow)
    return img.resize((size,size),Image.LANCZOS)


def apply_mask(voxel_list, mask_ratio=MASK_RATIO, seed=42):
    """Return masked list (masked positions → 0/air) + count."""
    arr     = np.array(voxel_list, dtype=np.int32)
    non_air = np.where(arr != 0)[0]
    rng     = np.random.default_rng(seed)
    n_mask  = max(1, int(len(non_air) * mask_ratio))
    idx     = rng.choice(len(non_air), n_mask, replace=False)
    masked  = arr.copy()
    masked[non_air[idx]] = 0   # set to air
    return masked.tolist(), n_mask


def add_label(img, text, font_size=26):
    bar_h = 42
    out   = Image.new('RGB', (img.width, img.height+bar_h), (25,25,25))
    out.paste(img, (0,0))
    draw  = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((img.width//2, img.height+bar_h//2), text,
              fill=(230,230,230), font=font, anchor='mm')
    return out


def safe_name(s):
    return re.sub(r'[^\w\-]', '_', str(s))[:40]


def load_view(folder, idx=0):
    """Load view_XX.jpg rendered by render_engine.js, resize to IMG_SIZE."""
    p = os.path.join(folder, f'view_{idx:02d}.jpg')
    if not os.path.exists(p):
        return _sky_bg(IMG_SIZE)
    return Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    import pyarrow.parquet as pq
    pf  = pq.ParquetFile(PARQUET_FILE)
    df  = next(pf.iter_batches(batch_size=MAX_SCHEMAS)).to_pandas()
    print(f"Loaded {len(df)} records from {PARQUET_FILE}\n")

    for done, (_, row) in enumerate(df.iterrows()):
        title = row.get('title', f'schematic_{done}')
        tags  = row.get('tags', '')
        print(f"[{done+1:02d}] {title}  |  tags: {tags}")

        # Prepare flat voxel list (x*H*L + y*L + z ordering, per render_engine.js)
        vd = row['voxel_data']
        if hasattr(vd, 'tolist'): vd = vd.tolist()
        voxel_list = [int(v) for v in vd]

        if len(voxel_list) != VOXEL_SIZE**3:
            print(f"  Unexpected size {len(voxel_list)}, skipping"); continue

        # Masked list (masked positions → 0/air)
        masked_list, n_masked = apply_mask(voxel_list, MASK_RATIO)
        n_nonair = sum(1 for v in voxel_list if v != 0)
        print(f"  Non-air: {n_nonair}  |  Masked: {n_masked} ({n_masked/max(n_nonair,1)*100:.1f}%)")

        folder      = os.path.join(OUT_DIR, f'{done+1:02d}_{safe_name(title)}')
        tmp_orig    = os.path.join(folder, '_tmp_orig')
        tmp_masked  = os.path.join(folder, '_tmp_masked')
        os.makedirs(tmp_orig,   exist_ok=True)
        os.makedirs(tmp_masked, exist_ok=True)

        # ── Render 1 & 2 via prismarine-viewer (real Minecraft textures) ──────
        try:
            render_engine.renderFromRaw(voxel_list,  VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, tmp_orig,   RENDER_ANGLE)
            render_engine.renderFromRaw(masked_list, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, tmp_masked, RENDER_ANGLE)
        except Exception as e:
            print(f"  Render error: {e}"); continue

        img_orig   = load_view(tmp_orig,   0)
        img_masked = load_view(tmp_masked, 0)

        # ── Render 3: Semantic (PIL, distinct color per block type) ───────────
        blocks_3d = np.array(voxel_list, dtype=np.int16).reshape(
            VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE).transpose(1, 2, 0)
        img_sem = render_semantic(blocks_3d)

        # ── Label ─────────────────────────────────────────────────────────────
        img_orig   = add_label(img_orig,   "Original")
        img_masked = add_label(img_masked, f"Masked ({MASK_RATIO*100:.0f}% → air)")
        img_sem    = add_label(img_sem,    "Semantic (block types)")

        # ── Combine ───────────────────────────────────────────────────────────
        h        = max(img_orig.height, img_masked.height, img_sem.height)
        combined = Image.new('RGB', (IMG_SIZE*3 + 20, h + 56), (20,20,20))
        draw_c   = ImageDraw.Draw(combined)
        try:
            tfont = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            tfont = ImageFont.load_default()
        draw_c.text((IMG_SIZE*3//2+10, 24), title, fill=(220,220,220), font=tfont, anchor='mm')
        combined.paste(img_orig,   (0,             52))
        combined.paste(img_masked, (IMG_SIZE+10,   52))
        combined.paste(img_sem,    (IMG_SIZE*2+20, 52))

        # ── Save & cleanup ────────────────────────────────────────────────────
        img_orig.save(  os.path.join(folder, 'original.jpg'))
        img_masked.save(os.path.join(folder, 'masked.jpg'))
        img_sem.save(   os.path.join(folder, 'semantic.png'))
        combined.save(  os.path.join(folder, 'combined.png'))
        shutil.rmtree(tmp_orig,   ignore_errors=True)
        shutil.rmtree(tmp_masked, ignore_errors=True)
        print(f"  Saved → {folder}/combined.png")

        if done+1 >= MAX_SCHEMAS:
            break

    print(f"\nDone → {OUT_DIR}")


if __name__ == '__main__':
    main()
