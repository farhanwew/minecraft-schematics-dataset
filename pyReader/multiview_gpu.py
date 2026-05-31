"""
MVCNN multi-view renderer — GPU version (moderngl / OpenGL).
Replaces PIL polygon loop with a single GPU draw call per view.
Reuses all parsing + config from multiview.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import moderngl
from PIL import Image

# ── Reuse everything from multiview.py except render_view ────────────────────
from multiview import (
    DATA_DIR, TFRECORD, META_FILE, N_VIEWS, ELEVATION, IMG_SIZE,
    MAX_SCHEMAS, MAX_VOL,
    read_schematics, parse_schematic,
    BLOCK_COLORS, BLOCK_FACE_COLORS, _DEFAULT_COLOR,
    BLOCK_NAMES, block_name,
    _surface_mask, _compute_ao,
    safe_name, make_grid,
    TOP_SHADE, RIGHT_SHADE, LEFT_SHADE,
)

OUT_DIR = os.path.join(DATA_DIR, 'multiview_gpu')

# ── Precompute color lookup tables as numpy arrays (indexed by block ID 0-255) ─
_TOP_TABLE  = np.zeros((256, 3), dtype=np.float32)
_SIDE_TABLE = np.zeros((256, 3), dtype=np.float32)
for _bid in range(256):
    _top, _side = (BLOCK_FACE_COLORS.get(_bid) or
                   (BLOCK_COLORS.get(_bid, _DEFAULT_COLOR),
                    BLOCK_COLORS.get(_bid, _DEFAULT_COLOR)))
    _TOP_TABLE[_bid]  = _top
    _SIDE_TABLE[_bid] = _side

# ── OpenGL setup (offscreen, no window needed) ────────────────────────────────
_CTX = moderngl.create_standalone_context()

_VERT = """
#version 330
in vec2  in_pos;
in vec3  in_color;
out vec3 v_color;
uniform vec2 u_res;
void main() {
    vec2 ndc  = in_pos / u_res * 2.0 - 1.0;
    ndc.y     = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color   = in_color;
}
"""

_FRAG = """
#version 330
in  vec3 v_color;
out vec4 out_color;
void main() {
    out_color = vec4(v_color / 255.0, 1.0);
}
"""

_PROG = _CTX.program(vertex_shader=_VERT, fragment_shader=_FRAG)


def _sky_gradient_pixels(size):
    """Sky gradient as (size,size,3) uint8 numpy array."""
    top    = np.array([160, 210, 250], dtype=np.float32)
    bottom = np.array([ 80, 140, 200], dtype=np.float32)
    t   = np.linspace(0, 1, size, dtype=np.float32)[:, None]     # (H,1)
    row = (top + (bottom - top) * t).astype(np.uint8)            # (H,3)
    return np.broadcast_to(row[:, None, :], (size, size, 3)).copy()


def _build_vbo(iso_x, iso_y, block_ids, ao_vals, order, s):
    """
    Vectorized: build float32 vertex buffer (x,y,r,g,b) for all blocks.
    Each block = 3 faces × 2 triangles × 3 verts = 18 vertices.
    Painter's order is preserved: block order[0] → order[1] → ...
    """
    n  = len(order)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float32)

    ox  = iso_x[order].astype(np.float32)
    oy  = iso_y[order].astype(np.float32)
    ao  = ao_vals[order].astype(np.float32)[:, None]   # (n,1)
    bid = block_ids[order].astype(np.int32)

    tc  = np.clip(_TOP_TABLE[bid]  * TOP_SHADE   * ao, 0, 255)  # (n,3)
    rc  = np.clip(_SIDE_TABLE[bid] * RIGHT_SHADE * ao, 0, 255)
    lc  = np.clip(_SIDE_TABLE[bid] * LEFT_SHADE  * ao, 0, 255)

    s2 = s / 2.0

    def face_verts(p0, p1, p2, p3):
        """Two triangles from quad (p0,p1,p2,p3) — shape (n,6,2)."""
        return np.stack([p0, p1, p2, p0, p2, p3], axis=1)

    # Left face  quad: p0=(ox-s,oy)  p1=(ox,oy+s2)  p2=(ox,oy+s2+s)  p3=(ox-s,oy+s)
    lv = face_verts(
        np.stack([ox-s, oy      ], -1),
        np.stack([ox,   oy+s2   ], -1),
        np.stack([ox,   oy+s2+s ], -1),
        np.stack([ox-s, oy+s    ], -1),
    )  # (n,6,2)

    # Right face quad: p0=(ox,oy+s2) p1=(ox+s,oy) p2=(ox+s,oy+s) p3=(ox,oy+s2+s)
    rv = face_verts(
        np.stack([ox,   oy+s2   ], -1),
        np.stack([ox+s, oy      ], -1),
        np.stack([ox+s, oy+s    ], -1),
        np.stack([ox,   oy+s2+s ], -1),
    )

    # Top face   quad: p0=(ox,oy-s2) p1=(ox+s,oy) p2=(ox,oy+s2) p3=(ox-s,oy)
    tv = face_verts(
        np.stack([ox,   oy-s2   ], -1),
        np.stack([ox+s, oy      ], -1),
        np.stack([ox,   oy+s2   ], -1),
        np.stack([ox-s, oy      ], -1),
    )

    # Positions: (n, 18, 2)  —  [left6 | right6 | top6] per block
    pos = np.concatenate([lv, rv, tv], axis=1)

    # Colors: (n, 18, 3)
    lc6 = np.tile(lc[:, None, :], (1, 6, 1))
    rc6 = np.tile(rc[:, None, :], (1, 6, 1))
    tc6 = np.tile(tc[:, None, :], (1, 6, 1))
    col = np.concatenate([lc6, rc6, tc6], axis=1)

    # Interleave: (n, 18, 5) → (n*18, 5)
    buf = np.concatenate([pos, col], axis=2)   # (n, 18, 5)
    return buf.reshape(-1, 5).astype(np.float32)


def render_view_gpu(blocks_3d, angle_deg, size=IMG_SIZE):
    H, L, W = blocks_3d.shape
    occ  = blocks_3d != 0
    surf = _surface_mask(occ)
    ao   = _compute_ao(occ)

    ys, zs, xs = np.where(surf)
    if len(xs) == 0:
        return Image.fromarray(_sky_gradient_pixels(size))

    block_ids = blocks_3d[ys, zs, xs]
    ao_vals   = ao[ys, zs, xs]

    # Rotate around Y
    cx, cy, cz = W/2.0, H/2.0, L/2.0
    bx = (xs - cx).astype(np.float32)
    by = (ys - cy).astype(np.float32)
    bz = (zs - cz).astype(np.float32)
    az = np.radians(angle_deg)
    rx =  np.cos(az)*bx + np.sin(az)*bz
    ry =  by
    rz = -np.sin(az)*bx + np.cos(az)*bz

    # Project to screen
    margin = 40
    iso_x_raw = (rx - rz)
    iso_y_raw = (rx + rz)*0.5 - ry

    x_span = max(iso_x_raw.max() - iso_x_raw.min() + 2, 1.0)
    y_span = max(iso_y_raw.max() - iso_y_raw.min() + 2, 1.0)
    usable = size - margin*2
    s      = max(1.0, min(usable/x_span, usable/y_span))

    cx_off = size/2 - ((iso_x_raw.min()+iso_x_raw.max())/2)*s
    cy_off = size/2 - ((iso_y_raw.min()+iso_y_raw.max())/2)*s - s/2
    iso_x  = iso_x_raw*s + cx_off
    iso_y  = iso_y_raw*s + cy_off

    # Painter's order
    depth = rx + rz + ry*0.001
    order = np.argsort(depth)

    # Build VBO
    vdata = _build_vbo(iso_x, iso_y, block_ids, ao_vals, order, s)

    # Offscreen framebuffer
    fbo = _CTX.framebuffer(
        color_attachments=[_CTX.texture((size, size), 3)]
    )
    fbo.use()

    # Clear with sky gradient
    sky = _sky_gradient_pixels(size)        # (H,W,3) uint8
    fbo.color_attachments[0].write(sky.tobytes())

    # Upload & draw
    vbo = _CTX.buffer(vdata.tobytes())
    vao = _CTX.vertex_array(
        _PROG,
        [(vbo, '2f 3f', 'in_pos', 'in_color')],
    )
    _PROG['u_res'].value = (float(size), float(size))

    _CTX.enable(moderngl.BLEND)
    _CTX.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    fbo.use()
    vao.render(moderngl.TRIANGLES)

    # Read back
    raw  = fbo.color_attachments[0].read()
    arr  = np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3)
    arr  = arr[::-1]   # OpenGL Y-flip

    vbo.release(); vao.release(); fbo.release()
    return Image.fromarray(arr)


def render_multiview_gpu(blocks_3d, n_views=N_VIEWS):
    angles = np.linspace(0, 360, n_views, endpoint=False)
    return [render_view_gpu(blocks_3d, a) for a in angles]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Reading {TFRECORD}')
    print(f'Output  → {OUT_DIR}')
    print(f'Views: {N_VIEWS}  |  Size: {IMG_SIZE}px  |  GPU: {_CTX.info["GL_RENDERER"]}\n')

    done = 0
    for raw, url, meta in read_schematics(TFRECORD, META_FILE):
        result = parse_schematic(raw)
        if result is None:
            continue

        W, H, L, blocks_3d = result
        title = meta.get('title', f'schematic_{done}')
        tags  = ', '.join(meta.get('tags') or []) or '—'
        print(f'[{done+1:02d}] "{title}"  {W}×{H}×{L}  tags: {tags}')

        ids, counts = np.unique(blocks_3d[blocks_3d != 0], return_counts=True)
        inventory   = sorted(zip(counts, ids), reverse=True)
        inv_str     = '  '.join(f'{block_name(bid)}×{cnt}' for cnt, bid in inventory[:8])
        if len(inventory) > 8:
            inv_str += f'  (+{len(inventory)-8} more)'
        print(f'       blocks: {inv_str}')

        folder = os.path.join(OUT_DIR, f'{done+1:02d}_{safe_name(title)}')
        os.makedirs(folder, exist_ok=True)

        block_inv = {block_name(int(bid)): int(cnt) for cnt, bid in inventory}
        with open(os.path.join(folder, 'blocks.json'), 'w') as f:
            json.dump(block_inv, f, indent=2)

        views = render_multiview_gpu(blocks_3d)
        for i, img in enumerate(views):
            img.save(os.path.join(folder, f'view_{i:02d}.png'))
        make_grid(views).save(os.path.join(folder, 'grid.png'))

        done += 1
        if done >= MAX_SCHEMAS:
            break

    print(f'\nDone. {done} schematics → {OUT_DIR}')


if __name__ == '__main__':
    main()
