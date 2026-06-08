import argparse
import multiprocessing as mp
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent
VOXEL_SIZE = 32

_render_engine = None


def safe_name(value):
    return re.sub(r"[^\w\-]", "_", str(value))[:40]


def get_render_engine():
    global _render_engine
    if _render_engine is None:
        # The javascript bridge resolves relative requires from the current cwd.
        os.chdir(SCRIPT_DIR)
        from javascript import require

        _render_engine = require("./render_engine.js")
    return _render_engine


def normalize_voxel_data(voxel_data):
    if hasattr(voxel_data, "tolist"):
        voxel_data = voxel_data.tolist()
    return [int(value) for value in voxel_data]


def make_tiled_image(folder, angles, columns):
    images = []
    for index in range(angles):
        image_path = folder / f"view_{index:02d}.jpg"
        if not image_path.exists():
            return False
        images.append(Image.open(image_path).convert("RGB"))

    if not images:
        return False

    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (width * columns, height * rows))
    for index, image in enumerate(images):
        grid.paste(image, box=(index % columns * width, index // columns * height))
        image.close()

    grid.save(folder / "tiled_views.jpg")
    return True


def render_task(task):
    (
        dataset_index,
        title,
        voxel_data,
        out_dir,
        angles,
        render_width,
        render_height,
        tile_columns,
    ) = task

    folder = Path(out_dir) / f"{dataset_index + 1:06d}_{safe_name(title)}"

    try:
        voxel_data = normalize_voxel_data(voxel_data)
        expected_size = VOXEL_SIZE**3
        if len(voxel_data) != expected_size:
            return dataset_index, False, f"unexpected voxel size {len(voxel_data)}; expected {expected_size}"

        render_engine = get_render_engine()
        render_engine.renderFromRaw(
            voxel_data,
            VOXEL_SIZE,
            VOXEL_SIZE,
            VOXEL_SIZE,
            str(folder),
            angles,
            render_width,
            render_height,
        )
        make_tiled_image(folder, angles, tile_columns)
        return dataset_index, True, str(folder)
    except Exception as exc:
        return dataset_index, False, str(exc)


def iter_tasks(args):
    parquet_file = Path(args.parquet_file)
    out_dir = Path(args.out_dir)
    end_exclusive = args.end_index + 1 if args.end_index is not None else None
    if end_exclusive is None and args.max_schemas > 0:
        end_exclusive = args.start_index + args.max_schemas

    parquet = pq.ParquetFile(parquet_file)
    next_index = 0

    for batch in parquet.iter_batches(batch_size=args.read_batch_size):
        batch_len = batch.num_rows
        batch_start = next_index
        batch_end = next_index + batch_len
        next_index = batch_end

        if batch_end <= args.start_index:
            continue
        if end_exclusive is not None and batch_start >= end_exclusive:
            break

        df = batch.to_pandas()
        for row_offset, row in df.iterrows():
            dataset_index = batch_start + int(row_offset)
            if dataset_index < args.start_index:
                continue
            if end_exclusive is not None and dataset_index >= end_exclusive:
                break

            title = row.get("title", f"schematic_{dataset_index}")
            yield (
                dataset_index,
                title,
                row["voxel_data"],
                str(out_dir),
                args.angles,
                args.render_width,
                args.render_height,
                args.tile_columns,
            )


def count_tasks(args):
    total_rows = pq.ParquetFile(args.parquet_file).metadata.num_rows
    start_index = min(args.start_index, total_rows)
    end_exclusive = args.end_index + 1 if args.end_index is not None else total_rows
    if args.max_schemas > 0 and args.end_index is None:
        end_exclusive = min(end_exclusive, start_index + args.max_schemas)
    end_exclusive = min(max(end_exclusive, start_index), total_rows)
    return end_exclusive - start_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Minecraft schematic Parquet rows with multiple worker processes."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing data.parquet. Defaults to the project parent directory.",
    )
    parser.add_argument(
        "--parquet-file",
        default=None,
        help="Path to the Parquet file. Defaults to <data-dir>/data.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <data-dir>/parquet_render_out_workers.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First dataset row to render.")
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last dataset row to render, inclusive. If set, this overrides --max-schemas.",
    )
    parser.add_argument(
        "--max-schemas",
        type=int,
        default=5,
        help="Number of schematics to render. Use 0 or -1 for all rows after start-index.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of renderer processes.")
    parser.add_argument("--angles", type=int, default=12, help="Number of camera angles per schematic.")
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="Square render size in pixels. Ignored if render width/height are set.",
    )
    parser.add_argument("--render-width", type=int, default=None, help="Render width in pixels.")
    parser.add_argument("--render-height", type=int, default=None, help="Render height in pixels.")
    parser.add_argument("--tile-columns", type=int, default=4, help="Columns in tiled_views.jpg.")
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=128,
        help="Parquet rows loaded by the parent process at a time.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    args.parquet_file = str(Path(args.parquet_file).resolve()) if args.parquet_file else str(data_dir / "data.parquet")
    args.out_dir = str(Path(args.out_dir).resolve()) if args.out_dir else str(data_dir / "parquet_render_out_workers")
    args.render_width = args.render_width or args.render_size
    args.render_height = args.render_height or args.render_size
    args.start_index = max(0, args.start_index)
    if args.end_index is not None:
        args.end_index = max(args.start_index, args.end_index)
    args.workers = max(1, args.workers)
    args.angles = max(1, args.angles)
    args.tile_columns = max(1, args.tile_columns)
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Parquet: {args.parquet_file}")
    print(f"Output:  {args.out_dir}")
    print(f"Workers: {args.workers} | angles: {args.angles} | size: {args.render_width}x{args.render_height}")
    if args.end_index is not None:
        print(f"Range:   index {args.start_index} to {args.end_index} inclusive")
    elif args.max_schemas > 0:
        print(f"Range:   index {args.start_index}, max {args.max_schemas} schematics")
    else:
        print(f"Range:   index {args.start_index} to end")

    tasks = iter_tasks(args)
    total_tasks = count_tasks(args)
    completed = 0
    failed = 0

    if args.workers == 1:
        results = map(render_task, tasks)
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=args.workers)
        results = pool.imap_unordered(render_task, tasks, chunksize=1)

    try:
        for dataset_index, ok, message in tqdm(results, total=total_tasks, unit="schematic"):
            if ok:
                completed += 1
                print(f"[{dataset_index + 1:06d}] saved: {message}")
            else:
                failed += 1
                print(f"[{dataset_index + 1:06d}] error: {message}")
    finally:
        if args.workers != 1:
            pool.close()
            pool.join()

    print(f"Done. Rendered: {completed}, failed: {failed}")


if __name__ == "__main__":
    main()
