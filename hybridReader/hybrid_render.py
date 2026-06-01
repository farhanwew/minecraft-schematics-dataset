import os
import re
import pandas as pd
from javascript import require

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = '/mnt/c/Users/kek/projects/mcschwew'
PARQUET_FILE = os.path.join(DATA_DIR, 'data.parquet')
OUT_DIR     = os.path.join(DATA_DIR, 'parquet_render_out')
MAX_SCHEMAS = 5
# ─────────────────────────────────────────────────────────────────────────────

def safe_name(s):
    return re.sub(r'[^\w\-]', '_', str(s))[:40]

def main():
    print("Loading Node.js rendering engine...")
    render_engine = require('./render_engine.js')
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Loading {PARQUET_FILE}...")
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(PARQUET_FILE)
        batches = pf.iter_batches(batch_size=MAX_SCHEMAS)
        df = next(batches).to_pandas()
    except Exception as e:
        print(f"Could not load parquet file: {e}")
        return

    done = 0
    for idx, row in df.iterrows():
        title = row.get('title', f'schematic_{done}')
        print(f"[{done+1}] Processing: {title}")
        
        folder = os.path.join(OUT_DIR, f'{done+1:02d}_{safe_name(title)}')
        
        voxel_data = row['voxel_data']
        # Ensure it's a list of standard python ints to avoid jspybridge JSON serialization errors
        if hasattr(voxel_data, 'tolist'):
            voxel_data = voxel_data.tolist()
        voxel_data = [int(v) for v in voxel_data]
        
        try:
            render_engine.renderFromRaw(voxel_data, 32, 32, 32, folder, 12)
            
            # Tile the 12 views into a 4x3 grid
            from PIL import Image
            import glob
            images = []
            for i in range(12):
                img_path = os.path.join(folder, f'view_{i:02d}.jpg')
                if os.path.exists(img_path):
                    images.append(Image.open(img_path))
            
            if len(images) == 12:
                w, h = images[0].size
                grid = Image.new('RGB', (w * 4, h * 3))
                for i, img in enumerate(images):
                    grid.paste(img, box=(i % 4 * w, i // 4 * h))
                grid.save(os.path.join(folder, 'tiled_views.jpg'))
        except Exception as e:
            print(f"Error rendering {title}: {e}")
            
        done += 1
        if done >= MAX_SCHEMAS:
            break

if __name__ == '__main__':
    main()
