// ahsssp_simple.cpp - Simplified production implementation
// Focuses on correctness and clean code structure
// Compilation: g++ -O3 -march=native -std=c++17 ahsssp_simple.cpp -o ahsssp_simple

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include <random>

using VertexId = uint32_t;
constexpr double INF = std::numeric_limits<double>::infinity();

// Edge structure
struct Edge {
    VertexId target;
    double weight;
    Edge(VertexId t, double w) : target(t), weight(w) {}
};

// Graph class
class Graph {
private:
    size_t n_;
    std::vector<std::vector<Edge>> adj_;
    
public:
    explicit Graph(size_t n) : n_(n), adj_(n) {}
    
    void add_edge(VertexId u, VertexId v, double weight) {
        if (u < n_ && v < n_) {
            adj_[u].emplace_back(v, weight);
        }
    }
    
    const std::vector<Edge>& out_edges(VertexId u) const {
        return adj_[u];
    }
    
    size_t num_vertices() const { return n_; }
    size_t num_edges() const {
        size_t m = 0;
        for (const auto& edges : adj_) m += edges.size();
        return m;
    }
};

// Main SSSP algorithm - Standard Dijkstra with optimizations
class OptimizedDijkstra {
private:
    const Graph& graph_;
    VertexId source_;
    std::vector<double> dist_;
    std::vector<VertexId> parent_;
    
public:
    OptimizedDijkstra(const Graph& graph, VertexId source) 
        : graph_(graph), source_(source), 
          dist_(graph.num_vertices(), INF),
          parent_(graph.num_vertices(), source) {
        dist_[source] = 0.0;
    }
    
    void compute() {
        using PQItem = std::pair<double, VertexId>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        std::vector<bool> visited(graph_.num_vertices(), false);
        
        pq.emplace(0.0, source_);
        
        while (!pq.empty()) {
            auto [d_u, u] = pq.top();
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (const auto& edge : graph_.out_edges(u)) {
                VertexId v = edge.target;
                double d_new = d_u + edge.weight;
                
                if (d_new < dist_[v]) {
                    dist_[v] = d_new;
                    parent_[v] = u;
                    pq.emplace(d_new, v);
                }
            }
        }
    }
    
    double distance(VertexId v) const {
        return v < dist_.size() ? dist_[v] : INF;
    }
    
    std::vector<VertexId> path(VertexId target) const {
        if (target >= dist_.size() || std::isinf(dist_[target])) {
            return {};
        }
        
        std::vector<VertexId> path;
        VertexId v = target;
        path.push_back(v);
        
        while (v != source_) {
            v = parent_[v];
            path.push_back(v);
        }
        
        std::reverse(path.begin(), path.end());
        return path;
    }
};

// Hierarchical SSSP with pivot-based optimization
class AHSSSPSimple {
private:
    const Graph& graph_;
    VertexId source_;
    size_t k_;  // log^(1/3)(n)
    std::vector<double> dist_;
    std::vector<VertexId> parent_;
    
public:
    AHSSSPSimple(const Graph& graph, VertexId source) 
        : graph_(graph), source_(source),
          k_(std::max(size_t(2), static_cast<size_t>(std::pow(std::log2(graph.num_vertices()), 1.0/3.0)))),
          dist_(graph.num_vertices(), INF),
          parent_(graph.num_vertices(), source) {
        dist_[source] = 0.0;
    }
    
    void compute() {
        // Phase 1: Bellman-Ford k sweeps from source
        bellman_ford_k_sweeps();
        
        // Phase 2: Finish with optimized Dijkstra
        finish_dijkstra();
    }
    
    double distance(VertexId v) const {
        return v < dist_.size() ? dist_[v] : INF;
    }
    
    std::vector<VertexId> path(VertexId target) const {
        if (target >= dist_.size() || std::isinf(dist_[target])) {
            return {};
        }
        
        std::vector<VertexId> path;
        VertexId v = target;
        path.push_back(v);
        
        while (v != source_) {
            v = parent_[v];
            path.push_back(v);
        }
        
        std::reverse(path.begin(), path.end());
        return path;
    }
    
private:
    void bellman_ford_k_sweeps() {
        std::vector<VertexId> active = {source_};
        
        for (size_t round = 0; round < k_; round++) {
            std::vector<VertexId> next_active;
            
            for (VertexId u : active) {
                double d_u = dist_[u];
                
                for (const auto& edge : graph_.out_edges(u)) {
                    VertexId v = edge.target;
                    double d_new = d_u + edge.weight;
                    
                    if (d_new < dist_[v]) {
                        dist_[v] = d_new;
                        parent_[v] = u;
                        next_active.push_back(v);
                    }
                }
            }
            
            active = std::move(next_active);
        }
    }
    
    void finish_dijkstra() {
        using PQItem = std::pair<double, VertexId>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        std::vector<bool> visited(graph_.num_vertices(), false);
        
        // Seed with all vertices that have finite distance
        for (VertexId v = 0; v < graph_.num_vertices(); v++) {
            if (!std::isinf(dist_[v])) {
                pq.emplace(dist_[v], v);
            }
        }
        
        while (!pq.empty()) {
            auto [d_u, u] = pq.top();
            pq.pop();
            
            if (visited[u] || d_u > dist_[u]) continue;
            visited[u] = true;
            
            for (const auto& edge : graph_.out_edges(u)) {
                VertexId v = edge.target;
                double d_new = d_u + edge.weight;
                
                if (d_new < dist_[v]) {
                    dist_[v] = d_new;
                    parent_[v] = u;
                    pq.emplace(d_new, v);
                }
            }
        }
    }
};

// ====================================================================================
// Test utilities
// ====================================================================================

std::unique_ptr<Graph> create_grid_graph(size_t rows, size_t cols) {
    size_t n = rows * cols;
    auto graph = std::make_unique<Graph>(n);
    
    auto idx = [cols](size_t r, size_t c) { return r * cols + c; };
    
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            VertexId u = idx(r, c);
            
            if (c + 1 < cols) {
                VertexId v = idx(r, c + 1);
                graph->add_edge(u, v, 1.0);
                graph->add_edge(v, u, 1.0);
            }
            
            if (r + 1 < rows) {
                VertexId v = idx(r + 1, c);
                graph->add_edge(u, v, 1.0);
                graph->add_edge(v, u, 1.0);
            }
        }
    }
    
    return graph;
}

std::unique_ptr<Graph> create_random_graph(size_t n, double edge_prob, uint32_t seed = 42) {
    auto graph = std::make_unique<Graph>(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> weight_dist(1.0, 100.0);
    
    for (VertexId u = 0; u < n; u++) {
        for (VertexId v = 0; v < n; v++) {
            if (u != v && prob_dist(rng) < edge_prob) {
                graph->add_edge(u, v, weight_dist(rng));
            }
        }
    }
    
    return graph;
}

void demo() {
    std::cout << "=== AH-SSSP Simple Demo ===\n\n";
    
    auto graph = std::make_unique<Graph>(6);
    graph->add_edge(0, 1, 4.0);
    graph->add_edge(0, 2, 2.0);
    graph->add_edge(1, 2, 1.0);
    graph->add_edge(1, 3, 5.0);
    graph->add_edge(2, 3, 8.0);
    graph->add_edge(2, 4, 10.0);
    graph->add_edge(3, 4, 2.0);
    graph->add_edge(3, 5, 6.0);
    graph->add_edge(4, 5, 3.0);
    
    AHSSSPSimple engine(*graph, 0);
    engine.compute();
    
    std::cout << "Shortest paths from source 0:\n";
    for (VertexId v = 0; v < graph->num_vertices(); v++) {
        double dist = engine.distance(v);
        auto path = engine.path(v);
        
        std::cout << "  To " << v << ": distance = " << dist << ", path = [";
        for (size_t i = 0; i < path.size(); i++) {
            std::cout << path[i];
            if (i + 1 < path.size()) std::cout << " -> ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void benchmark() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Performance Benchmark\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Test on different graph sizes
    std::vector<std::pair<size_t, size_t>> grid_sizes = {
        {50, 50},
        {100, 100},
        {200, 200},
        {316, 316}  // ~100K vertices
    };
    
    for (const auto& [rows, cols] : grid_sizes) {
        size_t n = rows * cols;
        std::cout << "\n[Grid Graph " << rows << "x" << cols << " (n=" << n << ")]\n";
        
        auto graph = create_grid_graph(rows, cols);
        std::cout << "Graph: " << graph->num_vertices() << " vertices, "
                  << graph->num_edges() << " edges\n";
        
        VertexId source = 0;
        
        // Standard Dijkstra
        {
            auto start = std::chrono::high_resolution_clock::now();
            OptimizedDijkstra dijkstra(*graph, source);
            dijkstra.compute();
            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();
            
            std::cout << "  Dijkstra: " << time << "s\n";
        }
        
        // AH-SSSP Simple
        {
            auto start = std::chrono::high_resolution_clock::now();
            AHSSSPSimple ahsssp(*graph, source);
            ahsssp.compute();
            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();
            
            std::cout << "  AH-SSSP:  " << time << "s\n";
        }
        
        // Verify correctness
        OptimizedDijkstra dijkstra(*graph, source);
        dijkstra.compute();
        
        AHSSSPSimple ahsssp(*graph, source);
        ahsssp.compute();
        
        size_t errors = 0;
        for (VertexId v = 0; v < graph->num_vertices(); v++) {
            if (std::abs(dijkstra.distance(v) - ahsssp.distance(v)) > 1e-6) {
                errors++;
            }
        }
        
        if (errors == 0) {
            std::cout << "  ✓ All distances correct!\n";
        } else {
            std::cout << "  ✗ Errors: " << errors << "/" << graph->num_vertices() << "\n";
        }
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n";
}

int main() {
    std::cout << "AH-SSSP Simplified Implementation\n"
              << "Bellman-Ford + Dijkstra hybrid approach\n\n";
    
    demo();
    benchmark();
    
    return 0;
}
