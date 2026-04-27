from pathlib import Path

from models import Graph, Point, Edge

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def _count_lines(path: str | Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _line in f)


def load_small(G:Graph):
    co_path = "data/raw/dimacs/small/USA-road-d.NY.co"
    gr_path = "data/raw/dimacs/small/USA-road-d.NY.gr"

    with open(co_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=_count_lines(co_path), desc="Loading points"):
            if line.startswith("v "):
                _, node, lon, lat = line.split()
                G.add_points(Point(int(lon)/1e6,int(lat)/1e6,int(node)))
                             
    with open(gr_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=_count_lines(gr_path), desc="Loading edges"):
            if line.startswith("a "):
                _, u, v, w = line.split()
                u, v, w = int(u), int(v), int(w)
                G.add_edge(Edge(u,v,w))
    
    G.set_info("USA-road-d.NY")
