#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr char kMagic[8] = {'A', 'I', 'C', 'C', 'P', 'P', '1', '\0'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint8_t kOutside = 0;
constexpr std::uint8_t kBoundary = 1;
constexpr std::uint8_t kInternal = 2;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr std::uint32_t kNoNode = std::numeric_limits<std::uint32_t>::max();

struct Edge {
    std::uint32_t to{};
    double weight{};
};

struct Graph {
    std::vector<std::int64_t> node_ids;
    std::vector<std::vector<Edge>> out;
    std::vector<std::vector<Edge>> in;
    std::vector<std::uint8_t> active;
    std::uint64_t edge_count{};
};

struct Region {
    std::uint32_t id{};
    std::vector<std::uint32_t> nodes;
    std::vector<std::uint32_t> boundaries;
};

struct MethodSelection {
    std::string name;
    std::vector<std::uint32_t> region_ids;
};

struct Query {
    std::int64_t id{};
    std::uint32_t source{};
    std::uint32_t target{};
};

struct QueryWindow {
    std::string name;
    std::vector<Query> queries;
};

struct BenchmarkInput {
    Graph graph;
    std::unordered_map<std::uint32_t, Region> regions;
    std::vector<MethodSelection> methods;
    std::vector<QueryWindow> windows;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::string& path) : file_(path, std::ios::binary) {
        if (!file_) {
            throw std::runtime_error("cannot open benchmark input: " + path);
        }
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        file_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!file_) {
            throw std::runtime_error("unexpected end of benchmark input");
        }
        return value;
    }

    void read_bytes(char* destination, std::size_t size) {
        file_.read(destination, static_cast<std::streamsize>(size));
        if (!file_) {
            throw std::runtime_error("unexpected end of benchmark input");
        }
    }

    std::string read_string() {
        const auto size = read<std::uint16_t>();
        std::string value(size, '\0');
        read_bytes(value.data(), size);
        return value;
    }

private:
    std::ifstream file_;
};

BenchmarkInput load_input(const std::string& path) {
    BinaryReader reader(path);
    char magic[8]{};
    reader.read_bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("unsupported benchmark input magic");
    }
    const auto version = reader.read<std::uint32_t>();
    if (version != kFormatVersion) {
        throw std::runtime_error("unsupported benchmark input version");
    }

    BenchmarkInput input;
    const auto node_count = reader.read<std::uint32_t>();
    const auto edge_count = reader.read<std::uint64_t>();
    if (node_count == 0 || edge_count == 0) {
        throw std::runtime_error("benchmark graph must be non-empty");
    }
    input.graph.node_ids.resize(node_count);
    input.graph.out.resize(node_count);
    input.graph.in.resize(node_count);
    input.graph.active.assign(node_count, 1);
    for (auto& node_id : input.graph.node_ids) {
        node_id = reader.read<std::int64_t>();
    }
    for (std::uint64_t index = 0; index < edge_count; ++index) {
        const auto source = reader.read<std::uint32_t>();
        const auto target = reader.read<std::uint32_t>();
        const auto weight = reader.read<double>();
        if (source >= node_count || target >= node_count || !(weight > 0.0)) {
            throw std::runtime_error("invalid graph edge in benchmark input");
        }
        input.graph.out[source].push_back({target, weight});
        input.graph.in[target].push_back({source, weight});
    }
    input.graph.edge_count = edge_count;

    const auto region_count = reader.read<std::uint32_t>();
    for (std::uint32_t index = 0; index < region_count; ++index) {
        Region region;
        region.id = reader.read<std::uint32_t>();
        const auto node_size = reader.read<std::uint32_t>();
        const auto boundary_size = reader.read<std::uint32_t>();
        region.nodes.resize(node_size);
        region.boundaries.resize(boundary_size);
        for (auto& node : region.nodes) {
            node = reader.read<std::uint32_t>();
            if (node >= node_count) {
                throw std::runtime_error("region contains invalid node");
            }
        }
        for (auto& node : region.boundaries) {
            node = reader.read<std::uint32_t>();
            if (node >= node_count) {
                throw std::runtime_error("region contains invalid boundary");
            }
        }
        if (!input.regions.emplace(region.id, std::move(region)).second) {
            throw std::runtime_error("duplicate region id in benchmark input");
        }
    }

    const auto method_count = reader.read<std::uint32_t>();
    input.methods.reserve(method_count);
    for (std::uint32_t index = 0; index < method_count; ++index) {
        MethodSelection method;
        method.name = reader.read_string();
        const auto selected_count = reader.read<std::uint32_t>();
        method.region_ids.resize(selected_count);
        for (auto& region_id : method.region_ids) {
            region_id = reader.read<std::uint32_t>();
            if (!input.regions.contains(region_id)) {
                throw std::runtime_error("method references missing region");
            }
        }
        input.methods.push_back(std::move(method));
    }

    const auto window_count = reader.read<std::uint32_t>();
    input.windows.reserve(window_count);
    for (std::uint32_t index = 0; index < window_count; ++index) {
        QueryWindow window;
        window.name = reader.read_string();
        const auto query_count = reader.read<std::uint32_t>();
        window.queries.resize(query_count);
        for (auto& query : window.queries) {
            query.id = reader.read<std::int64_t>();
            query.source = reader.read<std::uint32_t>();
            query.target = reader.read<std::uint32_t>();
            if (query.source >= node_count || query.target >= node_count) {
                throw std::runtime_error("query references invalid node");
            }
        }
        input.windows.push_back(std::move(window));
    }
    return input;
}

struct QueueItem {
    double distance{};
    std::int64_t node_id{};
    std::uint32_t node{};
};

struct QueueGreater {
    bool operator()(const QueueItem& left, const QueueItem& right) const {
        if (left.distance != right.distance) {
            return left.distance > right.distance;
        }
        return left.node_id > right.node_id;
    }
};

class DijkstraWorkspace {
public:
    explicit DijkstraWorkspace(std::size_t node_count)
        : forward_distance_(node_count), backward_distance_(node_count),
          forward_stamp_(node_count), backward_stamp_(node_count),
          forward_settled_(node_count), backward_settled_(node_count) {}

    void begin() {
        ++generation_;
        if (generation_ == 0) {
            std::fill(forward_stamp_.begin(), forward_stamp_.end(), 0);
            std::fill(backward_stamp_.begin(), backward_stamp_.end(), 0);
            std::fill(forward_settled_.begin(), forward_settled_.end(), 0);
            std::fill(backward_settled_.begin(), backward_settled_.end(), 0);
            generation_ = 1;
        }
        forward_heap_.clear();
        backward_heap_.clear();
    }

    bool has_forward(std::uint32_t node) const { return forward_stamp_[node] == generation_; }
    bool has_backward(std::uint32_t node) const { return backward_stamp_[node] == generation_; }
    bool forward_settled(std::uint32_t node) const { return forward_settled_[node] == generation_; }
    bool backward_settled(std::uint32_t node) const { return backward_settled_[node] == generation_; }
    double forward(std::uint32_t node) const { return forward_distance_[node]; }
    double backward(std::uint32_t node) const { return backward_distance_[node]; }

    bool improve_forward(std::uint32_t node, double distance, const Graph& graph) {
        if (has_forward(node) && !(distance < forward_distance_[node])) {
            return false;
        }
        forward_stamp_[node] = generation_;
        forward_distance_[node] = distance;
        forward_heap_.push_back({distance, graph.node_ids[node], node});
        std::push_heap(forward_heap_.begin(), forward_heap_.end(), QueueGreater{});
        return true;
    }

    bool improve_backward(std::uint32_t node, double distance, const Graph& graph) {
        if (has_backward(node) && !(distance < backward_distance_[node])) {
            return false;
        }
        backward_stamp_[node] = generation_;
        backward_distance_[node] = distance;
        backward_heap_.push_back({distance, graph.node_ids[node], node});
        std::push_heap(backward_heap_.begin(), backward_heap_.end(), QueueGreater{});
        return true;
    }

    void settle_forward(std::uint32_t node) { forward_settled_[node] = generation_; }
    void settle_backward(std::uint32_t node) { backward_settled_[node] = generation_; }

    bool clean_forward() {
        while (!forward_heap_.empty()) {
            const auto& item = forward_heap_.front();
            if (has_forward(item.node) && !forward_settled(item.node) &&
                item.distance == forward_distance_[item.node]) {
                return true;
            }
            pop_heap(forward_heap_);
        }
        return false;
    }

    bool clean_backward() {
        while (!backward_heap_.empty()) {
            const auto& item = backward_heap_.front();
            if (has_backward(item.node) && !backward_settled(item.node) &&
                item.distance == backward_distance_[item.node]) {
                return true;
            }
            pop_heap(backward_heap_);
        }
        return false;
    }

    const QueueItem& forward_top() const { return forward_heap_.front(); }
    const QueueItem& backward_top() const { return backward_heap_.front(); }

    QueueItem take_forward() {
        const auto item = forward_heap_.front();
        pop_heap(forward_heap_);
        return item;
    }

    QueueItem take_backward() {
        const auto item = backward_heap_.front();
        pop_heap(backward_heap_);
        return item;
    }

private:
    static void pop_heap(std::vector<QueueItem>& heap) {
        std::pop_heap(heap.begin(), heap.end(), QueueGreater{});
        heap.pop_back();
    }

    std::vector<double> forward_distance_;
    std::vector<double> backward_distance_;
    std::vector<std::uint32_t> forward_stamp_;
    std::vector<std::uint32_t> backward_stamp_;
    std::vector<std::uint32_t> forward_settled_;
    std::vector<std::uint32_t> backward_settled_;
    std::vector<QueueItem> forward_heap_;
    std::vector<QueueItem> backward_heap_;
    std::uint32_t generation_{};
};

struct SearchResult {
    double distance{kInfinity};
    std::uint64_t expanded{};
    std::uint64_t scanned_edges{};
};

SearchResult bidirectional_search(
    const Graph& graph,
    const std::vector<std::pair<std::uint32_t, double>>& forward_frontier,
    const std::vector<std::pair<std::uint32_t, double>>& backward_frontier,
    DijkstraWorkspace& workspace) {
    workspace.begin();
    for (const auto& [node, distance] : forward_frontier) {
        if (graph.active[node] && std::isfinite(distance) && distance >= 0.0) {
            workspace.improve_forward(node, distance, graph);
        }
    }
    for (const auto& [node, distance] : backward_frontier) {
        if (graph.active[node] && std::isfinite(distance) && distance >= 0.0) {
            workspace.improve_backward(node, distance, graph);
        }
    }
    if (!workspace.clean_forward() || !workspace.clean_backward()) {
        return {};
    }

    double best = kInfinity;
    for (const auto& [node, unused] : forward_frontier) {
        static_cast<void>(unused);
        if (workspace.has_forward(node) && workspace.has_backward(node)) {
            best = std::min(best, workspace.forward(node) + workspace.backward(node));
        }
    }
    std::uint64_t forward_expanded = 0;
    std::uint64_t backward_expanded = 0;
    std::uint64_t scanned_edges = 0;

    while (workspace.clean_forward() && workspace.clean_backward()) {
        if (workspace.forward_top().distance + workspace.backward_top().distance >= best) {
            break;
        }
        if (workspace.forward_top().distance <= workspace.backward_top().distance) {
            const auto item = workspace.take_forward();
            workspace.settle_forward(item.node);
            ++forward_expanded;
            if (workspace.backward_settled(item.node)) {
                best = std::min(best, item.distance + workspace.backward(item.node));
            }
            scanned_edges += graph.out[item.node].size();
            for (const auto& edge : graph.out[item.node]) {
                const double candidate = item.distance + edge.weight;
                if (workspace.improve_forward(edge.to, candidate, graph) &&
                    workspace.has_backward(edge.to)) {
                    best = std::min(best, candidate + workspace.backward(edge.to));
                }
            }
        } else {
            const auto item = workspace.take_backward();
            workspace.settle_backward(item.node);
            ++backward_expanded;
            if (workspace.forward_settled(item.node)) {
                best = std::min(best, item.distance + workspace.forward(item.node));
            }
            scanned_edges += graph.in[item.node].size();
            for (const auto& edge : graph.in[item.node]) {
                const double candidate = item.distance + edge.weight;
                if (workspace.improve_backward(edge.to, candidate, graph) &&
                    workspace.has_forward(edge.to)) {
                    best = std::min(best, candidate + workspace.forward(edge.to));
                }
            }
        }
    }
    return {best, forward_expanded + backward_expanded, scanned_edges};
}

struct WorkCount {
    std::uint64_t expanded{};
    std::uint64_t scanned_edges{};
};

WorkCount restricted_search(
    const Graph& graph,
    std::uint32_t source,
    std::int32_t region_index,
    const std::vector<std::int32_t>& node_region,
    bool reverse,
    DijkstraWorkspace& workspace) {
    workspace.begin();
    workspace.improve_forward(source, 0.0, graph);
    std::uint64_t expanded = 0;
    std::uint64_t scanned_edges = 0;
    while (workspace.clean_forward()) {
        const auto item = workspace.take_forward();
        workspace.settle_forward(item.node);
        ++expanded;
        const auto& neighbors = reverse ? graph.in[item.node] : graph.out[item.node];
        scanned_edges += neighbors.size();
        for (const auto& edge : neighbors) {
            if (node_region[edge.to] != region_index || workspace.forward_settled(edge.to)) {
                continue;
            }
            workspace.improve_forward(edge.to, item.distance + edge.weight, graph);
        }
    }
    return {expanded, scanned_edges};
}

struct CompressionIndex {
    Graph graph;
    std::vector<Region> regions;
    std::vector<std::uint8_t> node_state;
    std::vector<std::int32_t> node_region;
    std::uint64_t shortcut_count{};
    std::uint64_t internal_node_count{};
};

void append_minimum_edges(std::vector<std::vector<Edge>>& adjacency) {
    for (auto& edges : adjacency) {
        std::sort(edges.begin(), edges.end(), [](const Edge& left, const Edge& right) {
            if (left.to != right.to) {
                return left.to < right.to;
            }
            return left.weight < right.weight;
        });
        std::size_t write = 0;
        for (std::size_t read = 0; read < edges.size();) {
            const auto target = edges[read].to;
            double best = edges[read].weight;
            ++read;
            while (read < edges.size() && edges[read].to == target) {
                best = std::min(best, edges[read].weight);
                ++read;
            }
            edges[write++] = {target, best};
        }
        edges.resize(write);
    }
}

CompressionIndex build_index(
    const Graph& original,
    const std::unordered_map<std::uint32_t, Region>& region_pool,
    const MethodSelection& selection) {
    CompressionIndex index;
    index.node_state.assign(original.out.size(), kOutside);
    index.node_region.assign(original.out.size(), -1);
    index.regions.reserve(selection.region_ids.size());
    for (const auto region_id : selection.region_ids) {
        index.regions.push_back(region_pool.at(region_id));
    }
    for (std::size_t region_index = 0; region_index < index.regions.size(); ++region_index) {
        const auto& region = index.regions[region_index];
        for (const auto node : region.nodes) {
            if (index.node_region[node] != -1) {
                throw std::runtime_error("selected compression regions overlap");
            }
            index.node_region[node] = static_cast<std::int32_t>(region_index);
            index.node_state[node] = kInternal;
        }
        for (const auto node : region.boundaries) {
            if (index.node_region[node] != static_cast<std::int32_t>(region_index)) {
                throw std::runtime_error("region boundary is not a member");
            }
            index.node_state[node] = kBoundary;
        }
    }
    index.internal_node_count = static_cast<std::uint64_t>(std::count(
        index.node_state.begin(), index.node_state.end(), kInternal));

    std::vector<std::vector<Edge>> materialized(original.out.size());
    for (std::uint32_t source = 0; source < original.out.size(); ++source) {
        if (index.node_state[source] == kInternal) {
            continue;
        }
        for (const auto& edge : original.out[source]) {
            if (index.node_state[edge.to] != kInternal) {
                materialized[source].push_back(edge);
            }
        }
    }

    DijkstraWorkspace workspace(original.out.size());
    for (std::size_t region_index = 0; region_index < index.regions.size(); ++region_index) {
        const auto& region = index.regions[region_index];
        for (const auto source : region.boundaries) {
            restricted_search(
                original,
                source,
                static_cast<std::int32_t>(region_index),
                index.node_region,
                false,
                workspace);
            for (const auto target : region.boundaries) {
                if (target == source || !workspace.has_forward(target)) {
                    continue;
                }
                materialized[source].push_back({target, workspace.forward(target)});
                ++index.shortcut_count;
            }
        }
    }
    append_minimum_edges(materialized);

    index.graph.node_ids = original.node_ids;
    index.graph.out = std::move(materialized);
    index.graph.in.resize(original.in.size());
    index.graph.active.resize(original.out.size());
    for (std::uint32_t source = 0; source < index.graph.out.size(); ++source) {
        index.graph.active[source] = index.node_state[source] == kInternal ? 0 : 1;
        for (const auto& edge : index.graph.out[source]) {
            index.graph.in[edge.to].push_back({source, edge.weight});
            ++index.graph.edge_count;
        }
    }
    return index;
}

struct IndexedResult {
    double distance{kInfinity};
    std::uint64_t expanded{};
    std::uint64_t access_expanded{};
    std::uint64_t graph_expanded{};
    std::uint64_t scanned_edges{};
    std::uint64_t access_scanned_edges{};
    std::uint64_t graph_scanned_edges{};
};

class QueryEngine {
public:
    QueryEngine(const Graph& original, const CompressionIndex& index)
        : original_(original), index_(index), baseline_workspace_(original.out.size()),
          compressed_workspace_(original.out.size()), endpoint_workspace_(original.out.size()) {}

    SearchResult baseline(std::uint32_t source, std::uint32_t target) {
        if (source == target) {
            return {0.0, 1, 0};
        }
        return bidirectional_search(
            original_, {{source, 0.0}}, {{target, 0.0}}, baseline_workspace_);
    }

    IndexedResult indexed(std::uint32_t source, std::uint32_t target) {
        if (source == target) {
            return {0.0, 1, 0, 1, 0, 0, 0};
        }
        if (index_.node_state[source] != kInternal && index_.node_state[target] != kInternal) {
            const auto result = bidirectional_search(
                index_.graph, {{source, 0.0}}, {{target, 0.0}}, compressed_workspace_);
            return {
                result.distance,
                result.expanded,
                0,
                result.expanded,
                result.scanned_edges,
                0,
                result.scanned_edges,
            };
        }

        std::vector<std::pair<std::uint32_t, double>> forward{{source, 0.0}};
        std::vector<std::pair<std::uint32_t, double>> backward{{target, 0.0}};
        std::uint64_t access_expanded = 0;
        std::uint64_t access_scanned_edges = 0;
        double direct_distance = kInfinity;

        if (index_.node_state[source] == kInternal) {
            const auto region_index = index_.node_region[source];
            const auto& region = index_.regions.at(static_cast<std::size_t>(region_index));
            const auto access = restricted_search(
                original_, source, region_index, index_.node_region, false, endpoint_workspace_);
            access_expanded += access.expanded;
            access_scanned_edges += access.scanned_edges;
            forward.clear();
            forward.reserve(region.boundaries.size());
            for (const auto boundary : region.boundaries) {
                if (endpoint_workspace_.has_forward(boundary)) {
                    forward.emplace_back(boundary, endpoint_workspace_.forward(boundary));
                }
            }
            if (index_.node_region[target] == region_index && endpoint_workspace_.has_forward(target)) {
                direct_distance = endpoint_workspace_.forward(target);
            }
        }

        if (index_.node_state[target] == kInternal) {
            const auto region_index = index_.node_region[target];
            const auto& region = index_.regions.at(static_cast<std::size_t>(region_index));
            const auto access = restricted_search(
                original_, target, region_index, index_.node_region, true, endpoint_workspace_);
            access_expanded += access.expanded;
            access_scanned_edges += access.scanned_edges;
            backward.clear();
            backward.reserve(region.boundaries.size());
            for (const auto boundary : region.boundaries) {
                if (endpoint_workspace_.has_forward(boundary)) {
                    backward.emplace_back(boundary, endpoint_workspace_.forward(boundary));
                }
            }
        }

        const auto compressed = bidirectional_search(
            index_.graph, forward, backward, compressed_workspace_);
        const auto distance = std::min(direct_distance, compressed.distance);
        return {
            distance,
            access_expanded + compressed.expanded,
            access_expanded,
            compressed.expanded,
            access_scanned_edges + compressed.scanned_edges,
            access_scanned_edges,
            compressed.scanned_edges,
        };
    }

private:
    const Graph& original_;
    const CompressionIndex& index_;
    DijkstraWorkspace baseline_workspace_;
    DijkstraWorkspace compressed_workspace_;
    DijkstraWorkspace endpoint_workspace_;
};

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

double mean(const std::vector<double>& values) {
    double total = 0.0;
    for (const auto value : values) {
        total += value;
    }
    return values.empty() ? 0.0 : total / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double value) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = (static_cast<double>(values.size()) - 1.0) * value / 100.0;
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    if (lower == upper) {
        return values[lower];
    }
    const double fraction = index - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double change_percent(double newer, double older) {
    return older == 0.0 ? 0.0 : (newer - older) / older * 100.0;
}

double finite_distance(double value) {
    return std::isfinite(value) ? value : 0.0;
}

struct BenchmarkRow {
    std::string method;
    std::string window;
    std::uint64_t query_count{};
    std::uint64_t region_count{};
    std::uint64_t shortcut_count{};
    std::uint64_t internal_node_count{};
    double preprocessing_seconds{};
    std::uint32_t warmup_rounds{};
    std::uint32_t repetitions{};
    double baseline_avg_ms{};
    double indexed_avg_ms{};
    double elapsed_change_pct{};
    double baseline_p50_ms{};
    double indexed_p50_ms{};
    double p50_change_pct{};
    double baseline_p95_ms{};
    double indexed_p95_ms{};
    double p95_change_pct{};
    double baseline_avg_expanded{};
    double indexed_avg_expanded{};
    double indexed_avg_access_expanded{};
    double indexed_avg_graph_expanded{};
    double expanded_change_pct{};
    double baseline_avg_scanned_edges{};
    double indexed_avg_scanned_edges{};
    double indexed_avg_access_scanned_edges{};
    double indexed_avg_graph_scanned_edges{};
    double scanned_edges_change_pct{};
    double faster_query_rate_pct{};
    double correctness_rate{};
    double max_abs_distance_error{};
    double checksum{};
};

BenchmarkRow benchmark_window(
    const Graph& original,
    const CompressionIndex& index,
    const MethodSelection& method,
    const QueryWindow& window,
    double preprocessing_seconds,
    std::uint32_t warmup_rounds,
    std::uint32_t repetitions) {
    QueryEngine engine(original, index);
    double baseline_expanded = 0.0;
    double indexed_expanded = 0.0;
    double access_expanded = 0.0;
    double graph_expanded = 0.0;
    double baseline_scanned_edges = 0.0;
    double indexed_scanned_edges = 0.0;
    double access_scanned_edges = 0.0;
    double graph_scanned_edges = 0.0;
    double max_error = 0.0;
    std::uint64_t correct = 0;
    double checksum = 0.0;

    for (const auto& query : window.queries) {
        const auto baseline = engine.baseline(query.source, query.target);
        const auto indexed = engine.indexed(query.source, query.target);
        const bool same =
            (std::isinf(baseline.distance) && std::isinf(indexed.distance)) ||
            std::abs(baseline.distance - indexed.distance) <= 1e-6;
        if (same) {
            ++correct;
        } else {
            throw std::runtime_error(
                "distance mismatch for method=" + method.name +
                " window=" + window.name + " query=" + std::to_string(query.id));
        }
        if (std::isfinite(baseline.distance) && std::isfinite(indexed.distance)) {
            max_error = std::max(max_error, std::abs(baseline.distance - indexed.distance));
            checksum += baseline.distance + indexed.distance;
        }
        baseline_expanded += static_cast<double>(baseline.expanded);
        indexed_expanded += static_cast<double>(indexed.expanded);
        access_expanded += static_cast<double>(indexed.access_expanded);
        graph_expanded += static_cast<double>(indexed.graph_expanded);
        baseline_scanned_edges += static_cast<double>(baseline.scanned_edges);
        indexed_scanned_edges += static_cast<double>(indexed.scanned_edges);
        access_scanned_edges += static_cast<double>(indexed.access_scanned_edges);
        graph_scanned_edges += static_cast<double>(indexed.graph_scanned_edges);
    }

    for (std::uint32_t round = 0; round < warmup_rounds; ++round) {
        for (const auto& query : window.queries) {
            const auto baseline = engine.baseline(query.source, query.target);
            const auto indexed = engine.indexed(query.source, query.target);
            checksum += finite_distance(baseline.distance) + finite_distance(indexed.distance);
        }
    }

    std::vector<double> baseline_times;
    std::vector<double> indexed_times;
    baseline_times.reserve(window.queries.size() * repetitions);
    indexed_times.reserve(window.queries.size() * repetitions);
    std::uint64_t faster = 0;
    for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        for (const auto& query : window.queries) {
            double baseline_time = 0.0;
            double indexed_time = 0.0;
            SearchResult baseline;
            IndexedResult indexed;
            if ((static_cast<std::uint64_t>(query.id) + repetition) % 2 == 0) {
                auto start = std::chrono::steady_clock::now();
                indexed = engine.indexed(query.source, query.target);
                indexed_time = elapsed_ms(start);
                start = std::chrono::steady_clock::now();
                baseline = engine.baseline(query.source, query.target);
                baseline_time = elapsed_ms(start);
            } else {
                auto start = std::chrono::steady_clock::now();
                baseline = engine.baseline(query.source, query.target);
                baseline_time = elapsed_ms(start);
                start = std::chrono::steady_clock::now();
                indexed = engine.indexed(query.source, query.target);
                indexed_time = elapsed_ms(start);
            }
            baseline_times.push_back(baseline_time);
            indexed_times.push_back(indexed_time);
            faster += indexed_time < baseline_time ? 1 : 0;
            checksum += finite_distance(baseline.distance) + finite_distance(indexed.distance);
        }
    }

    const double query_count = static_cast<double>(window.queries.size());
    BenchmarkRow row;
    row.method = method.name;
    row.window = window.name;
    row.query_count = window.queries.size();
    row.region_count = index.regions.size();
    row.shortcut_count = index.shortcut_count;
    row.internal_node_count = index.internal_node_count;
    row.preprocessing_seconds = preprocessing_seconds;
    row.warmup_rounds = warmup_rounds;
    row.repetitions = repetitions;
    row.baseline_avg_ms = mean(baseline_times);
    row.indexed_avg_ms = mean(indexed_times);
    row.elapsed_change_pct = change_percent(row.indexed_avg_ms, row.baseline_avg_ms);
    row.baseline_p50_ms = percentile(baseline_times, 50.0);
    row.indexed_p50_ms = percentile(indexed_times, 50.0);
    row.p50_change_pct = change_percent(row.indexed_p50_ms, row.baseline_p50_ms);
    row.baseline_p95_ms = percentile(baseline_times, 95.0);
    row.indexed_p95_ms = percentile(indexed_times, 95.0);
    row.p95_change_pct = change_percent(row.indexed_p95_ms, row.baseline_p95_ms);
    row.baseline_avg_expanded = baseline_expanded / query_count;
    row.indexed_avg_expanded = indexed_expanded / query_count;
    row.indexed_avg_access_expanded = access_expanded / query_count;
    row.indexed_avg_graph_expanded = graph_expanded / query_count;
    row.expanded_change_pct = change_percent(row.indexed_avg_expanded, row.baseline_avg_expanded);
    row.baseline_avg_scanned_edges = baseline_scanned_edges / query_count;
    row.indexed_avg_scanned_edges = indexed_scanned_edges / query_count;
    row.indexed_avg_access_scanned_edges = access_scanned_edges / query_count;
    row.indexed_avg_graph_scanned_edges = graph_scanned_edges / query_count;
    row.scanned_edges_change_pct = change_percent(
        row.indexed_avg_scanned_edges, row.baseline_avg_scanned_edges);
    row.faster_query_rate_pct = static_cast<double>(faster) /
        static_cast<double>(baseline_times.size()) * 100.0;
    row.correctness_rate = static_cast<double>(correct) / query_count;
    row.max_abs_distance_error = max_error;
    row.checksum = checksum;
    return row;
}

void write_header(std::ofstream& file) {
    file << "method,window,query_count,region_count,shortcut_count,internal_node_count,"
         << "preprocessing_seconds,warmup_rounds,repetitions,baseline_avg_ms,indexed_avg_ms,"
         << "elapsed_change_pct,baseline_p50_ms,indexed_p50_ms,p50_change_pct,"
         << "baseline_p95_ms,indexed_p95_ms,p95_change_pct,baseline_avg_expanded,"
         << "indexed_avg_expanded,indexed_avg_access_expanded,indexed_avg_graph_expanded,"
         << "expanded_change_pct,baseline_avg_scanned_edges,indexed_avg_scanned_edges,"
         << "indexed_avg_access_scanned_edges,indexed_avg_graph_scanned_edges,"
         << "scanned_edges_change_pct,faster_query_rate_pct,correctness_rate,"
         << "max_abs_distance_error,checksum\n";
}

void write_row(std::ofstream& file, const BenchmarkRow& row) {
    file << row.method << ',' << row.window << ',' << row.query_count << ','
         << row.region_count << ',' << row.shortcut_count << ',' << row.internal_node_count << ','
         << std::setprecision(12) << row.preprocessing_seconds << ',' << row.warmup_rounds << ','
         << row.repetitions << ',' << row.baseline_avg_ms << ',' << row.indexed_avg_ms << ','
         << row.elapsed_change_pct << ',' << row.baseline_p50_ms << ',' << row.indexed_p50_ms << ','
         << row.p50_change_pct << ',' << row.baseline_p95_ms << ',' << row.indexed_p95_ms << ','
         << row.p95_change_pct << ',' << row.baseline_avg_expanded << ','
         << row.indexed_avg_expanded << ',' << row.indexed_avg_access_expanded << ','
         << row.indexed_avg_graph_expanded << ',' << row.expanded_change_pct << ','
         << row.baseline_avg_scanned_edges << ',' << row.indexed_avg_scanned_edges << ','
         << row.indexed_avg_access_scanned_edges << ',' << row.indexed_avg_graph_scanned_edges << ','
         << row.scanned_edges_change_pct << ','
         << row.faster_query_rate_pct << ',' << row.correctness_rate << ','
         << row.max_abs_distance_error << ',' << row.checksum << '\n';
    file.flush();
}

Graph toy_graph() {
    Graph graph;
    graph.node_ids = {0, 1, 2, 3, 4, 5};
    graph.out.resize(6);
    graph.in.resize(6);
    graph.active.assign(6, 1);
    const auto add = [&](std::uint32_t source, std::uint32_t target, double weight) {
        graph.out[source].push_back({target, weight});
        graph.in[target].push_back({source, weight});
        ++graph.edge_count;
    };
    add(0, 1, 1.0); add(1, 0, 1.0);
    add(1, 2, 1.2); add(2, 1, 1.2);
    add(2, 3, 0.8); add(3, 2, 0.8);
    add(3, 4, 1.1); add(4, 3, 1.1);
    add(4, 5, 0.9); add(5, 4, 0.9);
    add(0, 5, 20.0); add(5, 0, 20.0);
    return graph;
}

void self_test() {
    const auto graph = toy_graph();
    Region region{7, {1, 2, 3, 4}, {1, 4}};
    std::unordered_map<std::uint32_t, Region> regions{{7, region}};
    MethodSelection selection{"toy", {7}};
    const auto index = build_index(graph, regions, selection);
    QueryEngine engine(graph, index);
    for (std::uint32_t source = 0; source < graph.out.size(); ++source) {
        for (std::uint32_t target = 0; target < graph.out.size(); ++target) {
            const auto baseline = engine.baseline(source, target);
            const auto indexed = engine.indexed(source, target);
            if (std::abs(baseline.distance - indexed.distance) > 1e-9) {
                throw std::runtime_error("C++ self-test distance mismatch");
            }
        }
    }
    if (index.shortcut_count != 2 || index.internal_node_count != 2) {
        throw std::runtime_error("C++ self-test index structure mismatch");
    }
    std::cout << "self-test passed: all 36 OD pairs exact\n";
}

struct Options {
    std::string input;
    std::string output;
    std::uint32_t warmup_rounds{2};
    std::uint32_t repetitions{10};
    bool self_test{};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[++index];
        };
        if (argument == "--input") {
            options.input = value();
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--warmup") {
            options.warmup_rounds = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--repetitions") {
            options.repetitions = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (!options.self_test && (options.input.empty() || options.output.empty())) {
        throw std::runtime_error("--input and --output are required");
    }
    if (options.repetitions == 0) {
        throw std::runtime_error("--repetitions must be positive");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        const auto options = parse_options(argc, argv);
        if (options.self_test) {
            self_test();
            return 0;
        }
        const auto input = load_input(options.input);
        std::ofstream output(options.output);
        if (!output) {
            throw std::runtime_error("cannot open benchmark output: " + options.output);
        }
        write_header(output);
        std::cout << "loaded nodes=" << input.graph.out.size()
                  << " edges=" << input.graph.edge_count
                  << " methods=" << input.methods.size()
                  << " windows=" << input.windows.size() << '\n';
        for (const auto& method : input.methods) {
            const auto start = std::chrono::steady_clock::now();
            const auto index = build_index(input.graph, input.regions, method);
            const double preprocessing_seconds = elapsed_ms(start) / 1000.0;
            std::cout << "built method=" << method.name
                      << " regions=" << index.regions.size()
                      << " shortcuts=" << index.shortcut_count
                      << " internal=" << index.internal_node_count
                      << " seconds=" << std::fixed << std::setprecision(3)
                      << preprocessing_seconds << '\n';
            for (const auto& window : input.windows) {
                const auto row = benchmark_window(
                    input.graph,
                    index,
                    method,
                    window,
                    preprocessing_seconds,
                    options.warmup_rounds,
                    options.repetitions);
                write_row(output, row);
                std::cout << "complete method=" << method.name
                          << " window=" << window.name
                          << " baseline=" << std::setprecision(4) << row.baseline_avg_ms << "ms"
                          << " indexed=" << row.indexed_avg_ms << "ms"
                          << " time_change=" << std::setprecision(2) << row.elapsed_change_pct << "%"
                          << " expanded_change=" << row.expanded_change_pct << "%"
                          << " correctness=" << std::setprecision(6) << row.correctness_rate << '\n';
            }
        }
        std::cout << "summary=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
