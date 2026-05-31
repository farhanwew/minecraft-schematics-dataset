# Minecraft Schematics — Multi-View Image Generator

Pipeline untuk generate MVCNN-style multi-view images dari Minecraft schematic dataset.
Setiap schematic di-render dari 12 sudut pandang isometrik berbeda.

---

## Requirements

```bash
pip install nbtlib numpy pillow
```

---

## Data yang dibutuhkan

Taruh di folder `data/` (satu level di atas `pyReader/`):

| File | Keterangan |
|------|-----------|
| `schematics_0.tfrecords` | Dataset schematic (binary TFRecords) |
| `schematicsWithFinalUrl.json` | Metadata (judul, tags, URL per schematic) |

---

## Cara pakai

### 1. Generate multi-view images

```bash
cd pyReader
python multiview.py
```

Output tersimpan di `data/multiview_out/`. Setiap schematic punya folder sendiri:

```
data/multiview_out/
├── 01_3x3_f_home_ocean/
│   ├── view_00.png   ← sudut 0°
│   ├── view_01.png   ← sudut 30°
│   ├── ...
│   ├── view_11.png   ← sudut 330°
│   ├── grid.png      ← semua 12 view dalam satu gambar
│   └── blocks.json   ← inventori block (minecraft:name → jumlah)
├── 02_Meeresgrund/
│   └── ...
```

### 2. Konfigurasi (edit bagian Config di `multiview.py`)

```python
OUT_DIR     = 'data/multiview_out'   # folder output
N_VIEWS     = 12                     # jumlah sudut pandang
IMG_SIZE    = 2048                   # resolusi gambar (px)
MAX_SCHEMAS = 1000                   # berapa schematic yang diproses
MAX_VOL     = 500_000                # skip schematic yang terlalu besar
```

---

## Cara kerja render

```
blocks_3d[y,z,x]          array 3D voxel (nilai = block ID)
        ↓
surface mask               hanya render blok yang terlihat dari luar
        ↓
ambient occlusion          blok terkubur → lebih gelap
        ↓
rotasi Y-axis              12 sudut berbeda (0°, 30°, 60°, ...)
        ↓
isometric projection       iso_x = rx - rz
                           iso_y = (rx+rz)*0.5 - ry
        ↓
painter's sort             belakang digambar dulu
        ↓
3 face per blok            top (terang), right (medium), left (gelap)
        ↓
LANCZOS downsample         4096px internal → 2048px output
        ↓
view_XX.png
```

---

## Block ID mapping

Semua legacy numeric ID (format pre-1.13) sudah di-mapping ke nama Minecraft:

```python
from multiview import block_name
block_name(2)   # → "minecraft:grass"
block_name(17)  # → "minecraft:log"
```

Inventori per schematic tersimpan otomatis di `blocks.json`:
```json
{
  "minecraft:quartz_block": 4696,
  "minecraft:prismarine": 4480,
  "minecraft:water": 1476
}
```

---

## File lain

| File | Keterangan |
|------|-----------|
| `multiview.py` | Main script — render CPU (PIL) |
| `multiview_gpu.py` | Versi GPU (moderngl/OpenGL), butuh `pip install moderngl` |
| `feature_extract.py` | MVCNN feature extraction pakai ResNet-18, butuh `pip install torchvision` |
