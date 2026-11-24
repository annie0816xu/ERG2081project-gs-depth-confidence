"""Fetch published DX.GL benchmark data without executing remote code."""

import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import is_zipfile
import shutil


SOURCES = {
    "chair": ("modern_arm_chair", "KLBJAuie9JaB"),
    "sign": ("wet_floor_sign", "tHdRul1GzzoU"),
}


def valid(path, name):
    if not path.is_file():
        return False
    if name == "reference.zip":
        return is_zipfile(path)
    with path.open("rb") as stream:
        return stream.readline(10).strip() == b"ply"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=SOURCES, default="chair")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or Path("data") / args.object
    root.mkdir(parents=True, exist_ok=True)
    slug, dataset_id = SOURCES[args.object]
    assets = {
        "scene.ply": f"https://dx.gl/splat/{slug}.ply",
        "reference.zip": f"https://dx.gl/api/v/EJbs8npt2RVM/{dataset_id}/dataset",
    }
    for name, url in assets.items():
        destination = root / name
        if valid(destination, name):
            print(f"Reusing {destination}", flush=True)
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {name}", flush=True)
        request = Request(url, headers={"User-Agent": "3DGS-course-project/1.0"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not valid(partial, name):
            raise ValueError(f"Invalid download: {url}")
        partial.replace(destination)
        print(f"{name}: {destination.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
