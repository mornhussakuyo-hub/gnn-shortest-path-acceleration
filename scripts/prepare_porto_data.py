"""通过跨平台的一键流程准备波尔图 OD 与 OSM 路网数据。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/339/"
    "taxi%2Bservice%2Btrajectory%2Bprediction%2Bchallenge%2Becml%2Bpkdd%2B2015.zip"
)
DEFAULT_OSM_URL = "https://download.geofabrik.de/europe/portugal-latest.osm.pbf"

DATA_DIR = ROOT_DIR / "data" / "compressed" / "porto"
UCI_ZIP = DATA_DIR / "uci_porto_taxi.zip"
TRAIN_ZIP = DATA_DIR / "train.csv.zip"
OSM_PBF = DATA_DIR / "portugal-latest.osm.pbf"

OD_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图起终点样本_10万.csv"
NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
REPORT_MD = ROOT_DIR / "data" / "processed" / "porto" / "波尔图起终点吸附质量报告.md"
ROAD_PLOT = ROOT_DIR / "data" / "plots" / "波尔图道路底图起终点热力图_10万.png"


def log(message: str) -> None:
    print(f"[porto-data] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and build the Porto taxi OD + OSM road-network dataset."
    )
    parser.add_argument("--venv", type=Path, default=ROOT_DIR / ".venv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--od-limit", type=int, default=100_000)
    parser.add_argument("--force", action="store_true", help="Redownload and rebuild existing outputs.")
    parser.add_argument("--skip-road-plot", action="store_true", help="Skip the OSM road-background heatmap.")
    parser.add_argument(
        "--skip-dependency-install",
        action="store_true",
        help="Use the existing virtual environment without running pip.",
    )
    parser.add_argument("--uci-url", default=DEFAULT_UCI_URL)
    parser.add_argument("--osm-url", default=DEFAULT_OSM_URL)
    return parser.parse_args()


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def need_file(path: Path, force: bool) -> bool:
    return force or not path.is_file() or path.stat().st_size == 0


def run(command: list[str | os.PathLike[str]], env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    log(printable)
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def ensure_venv(venv_dir: Path, python: str) -> Path:
    python_path = venv_python(venv_dir)
    if not python_path.exists():
        log(f"creating Python virtual environment at {venv_dir}")
        run([python, "-m", "venv", str(venv_dir)])

    log("installing Python dependencies")
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_path), "-m", "pip", "install", "-r", "requirements.txt"])
    return python_path


def download_file(url: str, output: Path, force: bool) -> None:
    if not need_file(output, force):
        log(f"found {output.relative_to(ROOT_DIR)}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    log(f"downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as dst:
        shutil.copyfileobj(response, dst, length=1024 * 1024)
    tmp.replace(output)
    log(f"saved {output.relative_to(ROOT_DIR)}")


def extract_train_zip(force: bool) -> None:
    if not need_file(TRAIN_ZIP, force):
        log(f"found {TRAIN_ZIP.relative_to(ROOT_DIR)}")
        return

    log("extracting train.csv.zip from UCI archive")
    with zipfile.ZipFile(UCI_ZIP) as archive:
        matches = [name for name in archive.namelist() if name.endswith("train.csv.zip")]
        if not matches:
            raise SystemExit("train.csv.zip was not found inside the UCI archive.")
        TRAIN_ZIP.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(matches[0]) as src, TRAIN_ZIP.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def processing_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))
    return env


def run_processing(python_path: Path, od_limit: int, force: bool, skip_road_plot: bool) -> None:
    env = processing_env()

    if need_file(OD_CSV, force):
        log(f"extracting {od_limit:,} Porto OD samples")
        run([str(python_path), "scripts/generate_porto_od_heatmap.py", "--limit", str(od_limit)], env=env)
    else:
        log(f"found {OD_CSV.relative_to(ROOT_DIR)}")

    road_outputs = [NODE_CSV, EDGE_CSV, QUERY_CSV, REPORT_MD]
    if force or any(need_file(path, False) for path in road_outputs):
        log("building Porto road graph and snapping OD queries")
        run([str(python_path), "scripts/build_porto_road_graph_and_snap_od.py"], env=env)
    else:
        log("found processed road graph and snapped query files")

    if not skip_road_plot:
        if need_file(ROAD_PLOT, force):
            log("drawing OD heatmap on local OSM roads")
            run([str(python_path), "scripts/generate_porto_od_heatmap_with_roads.py"], env=env)
        else:
            log(f"found {ROAD_PLOT.relative_to(ROOT_DIR)}")


def main() -> None:
    args = parse_args()
    if args.skip_dependency_install:
        python_path = venv_python(args.venv)
        if not python_path.is_file():
            raise SystemExit(f"existing virtual environment Python not found: {python_path}")
        log(f"using existing Python environment at {python_path}")
    else:
        python_path = ensure_venv(args.venv, args.python)
    download_file(args.uci_url, UCI_ZIP, args.force)
    extract_train_zip(args.force)
    download_file(args.osm_url, OSM_PBF, args.force)
    run_processing(python_path, args.od_limit, args.force, args.skip_road_plot)

    log("done")
    log(f"main query file: {QUERY_CSV.relative_to(ROOT_DIR)}")
    log(f"road nodes: {NODE_CSV.relative_to(ROOT_DIR)}")
    log(f"road edges: {EDGE_CSV.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
