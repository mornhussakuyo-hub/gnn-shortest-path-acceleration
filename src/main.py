from load_data import load_small
from models import Graph


def main() -> None:
    graph = Graph()
    load_small(graph)
    print(graph.get_summary())


if __name__ == "__main__":
    main()
