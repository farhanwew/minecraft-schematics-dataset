# Minecraft Schematic Hybrid Renderer

This project reads Minecraft schematic voxel data from a Parquet file using Python and PyArrow, and sends it to a Node.js `prismarine-viewer` engine to generate 3D isometric screenshots from 12 angles, which are then stitched into a 4x3 tiled grid using Pillow.

## Setup Instructions

### 1. Prerequisites
- **Node.js** (v18+)
- **Python 3**
- **Xvfb** (X virtual framebuffer) - Required if you are running headless on a Linux server or Google Colab, as the Three.js rendering engine needs a display server to run WebGL.

### 2. Installation
Install the Node.js packages:
```bash
npm install
# or if using pnpm: pnpm install
```

Install the Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Running on Google Colab (or Headless Linux)

If you are running this on a Colab notebook to process the whole dataset, you'll need to install `xvfb` first to provide the headless virtual display.

1. Install `xvfb`:
```bash
!apt-get update
!apt-get install -y xvfb
```

2. Run the renderer using `xvfb-run`:
```bash
!xvfb-run -a python3 hybrid_render.py
```

*Note: The `-a` flag automatically finds a free display number.*

## Running Locally (with a display)
If you are running locally on a system with a GUI (like macOS or a Linux desktop), you don't need `xvfb-run`. Just run:
```bash
python3 hybrid_render.py
```

*(If you are running on WSL, you will still want to use `xvfb-run -a python3 hybrid_render.py` or ensure WSLg is working).*

---

## Processing the Full Dataset
Right now, the script is configured to only process the first 5 schematics for testing. To run the full dataset:
1. Open `hybrid_render.py`
2. Update `MAX_SCHEMAS = 5` at the top of the file to a much higher number (e.g. `MAX_SCHEMAS = 10000`), OR modify the script to loop through all `pf.iter_batches()` until the dataset is finished!
3. (Optional) Make sure `PARQUET_FILE` points to your full dataset if it's named something else!

## Outputs
The output renders are saved in the `parquet_render_out/` directory, organized by the schematic title. Each folder will contain the 12 individual angles (`view_00.jpg` to `view_11.jpg`) and a stitched `tiled_views.jpg`!
