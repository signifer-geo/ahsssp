// ahsssp.hpp - Adaptive Hierarchical Single-Source Shortest Path
// Production-grade C++ implementation
// Based on: "Breaking the Sorting Barrier for Directed SSSP" (Duan et al., 2025)
//
// Compilation: g++ -O3 -march=native -std=c++17 -pthread ahsssp.cpp -o ahsssp
// Usage: ./ahsssp [graph_file]

#ifndef AHSSSP_HPP
#define AHSSSP_HPP

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
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

namespace ahsssp {

// ============================================================================
// Configuration and Constants
// ============================================================================

struct Config {
    // Algorithm parameters (auto-tuned based on graph size)
    size_t k;              // Branching factor: log^(1/3)(n)
    size_t num_levels;     // Number of hierarchy levels: log_k(n)
    double delta;          // Distance quantum
    
    // Performance tuning
    size_t num_threads = std::thread::hardware_concurrency();
    bool enable_early_termination = true;
    double early_term_threshold = 0.99;  // Terminate when 99% certified
    
    // Memory optimization
    bool compress_pivots = false;
    size_t cache_line_size = 64;
    
    Config(size_t n, double min_weight = 1.0) {
        k = std::max(size_t(2), static_cast<size_t>(std::pow(std::log2(n), 1.0/3.0)));
        num_levels = std::max(size_t(1), static_cast<size_t>(std::log(n) / std::log(k)));
        delta = min_weight / 10.0;
    }
};

// ============================================================================
// Core Data Structures
// ============================================================================

using VertexId = uint32_t;
using PivotId = uint32_t;
constexpr double INF = std::numeric_limits<double>::infinity();

struct Edge {
    VertexId target;
    double weight;
    
    Edge() : target(0), weight(0.0) {}
    Edge(VertexId t, double w) : target(t), weight(w) {}
};

struct Vertex {
    double d_lower;      // Proven lower bound
    double d_upper;      // Current estimate
    double d_exact;      // Exact distance (when certified)
    uint8_t level;       // Certification level
    PivotId pivot;       // Nearest certified pivot
    VertexId parent;     // For path reconstruction
    bool certified;      // Whether d_exact is valid
    
    Vertex() : d_lower(INF), d_upper(INF), d_exact(INF), 
               level(0), pivot(0), parent(0), certified(false) {}
};

struct Pivot {
    VertexId root;
    uint8_t level;
    size_t subtree_size;
    double radius;
    PivotId parent_pivot;
    std::vector<std::pair<VertexId, double>> distances;  // Cached distances
    
    Pivot() : root(0), level(0), subtree_size(0), radius(0.0), parent_pivot(0) {}
    Pivot(VertexId r, uint8_t l, size_t sz, double rad) 
        : root(r), level(l), subtree_size(sz), radius(rad), parent_pivot(0) {}
};

// ============================================================================
// Graph Representation
// ============================================================================

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
        for (const auto& edges : adj_) {
            m += edges.size();
        }
        return m;
    }
    
    double min_edge_weight() const {
        double min_w = INF;
        for (const auto& edges : adj_) {
            for (const auto& e : edges) {
                min_w = std::min(min_w, e.weight);
            }
        }
        return min_w == INF ? 1.0 : min_w;
    }
    
    // Load from edge list file: u v weight (one per line)
    static std::unique_ptr<Graph> from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // First pass: count vertices
        size_t max_vertex = 0;
        std::vector<std::tuple<VertexId, VertexId, double>> edges;
        
        VertexId u, v;
        double w;
        while (file >> u >> v >> w) {
            max_vertex = std::max(max_vertex, std::max<size_t>(u, v));
            edges.emplace_back(u, v, w);
        }
        
        auto graph = std::make_unique<Graph>(max_vertex + 1);
        
        // Second pass: add edges
        for (const auto& [u, v, w] : edges) {
            graph->add_edge(u, v, w);
        }
        
        return graph;
    }
};

// ============================================================================
// Hierarchical Queue (Level-Stratified Priority Queue)
// ============================================================================

class LevelQueue {
private:
    size_t k_;
    std::vector<std::vector<VertexId>> buckets_;
    std::unordered_map<VertexId, size_t> vertex_to_bucket_;
    size_t size_;
    
public:
    explicit LevelQueue(size_t k) : k_(k), buckets_(k), size_(0) {}
    
    void insert(VertexId v, double dist) {
        size_t bucket = std::isinf(dist) ? (k_ - 1) : 
                        static_cast<size_t>(dist / 1000.0) % k_;
        buckets_[bucket].push_back(v);
        vertex_to_bucket_[v] = bucket;
        size_++;
    }
    
    void remove(VertexId v) {
        auto it = vertex_to_bucket_.find(v);
        if (it != vertex_to_bucket_.end()) {
            size_t bucket = it->second;
            auto& vec = buckets_[bucket];
            vec.erase(std::remove(vec.begin(), vec.end(), v), vec.end());
            vertex_to_bucket_.erase(it);
            size_--;
        }
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
};

class HierarchicalQueue {
private:
    std::vector<LevelQueue> levels_;
    size_t k_;
    
public:
    HierarchicalQueue(size_t k, size_t num_levels) : k_(k) {
        levels_.reserve(num_levels);
        for (size_t i = 0; i < num_levels; i++) {
            levels_.emplace_back(k);
        }
    }
    
    void insert(VertexId v, double dist, uint8_t level) {
        if (level < levels_.size()) {
            levels_[level].insert(v, dist);
        }
    }
    
    void remove(VertexId v, uint8_t level) {
        if (level < levels_.size()) {
            levels_[level].remove(v);
        }
    }
    
    bool has_vertices_at_level(uint8_t level) const {
        return level < levels_.size() && !levels_[level].empty();
    }
};

// ============================================================================
// Statistics Tracking
// ============================================================================

struct Statistics {
    size_t num_pivots = 0;
    size_t num_certified = 0;
    size_t num_dijkstra_calls = 0;
    size_t total_edges_relaxed = 0;
    
    double time_init = 0.0;
    double time_levels = 0.0;
    double time_total = 0.0;
    
    std::vector<double> level_times;
    
    void print() const {
        std::cout << "Statistics:\n"
                  << "  Pivots created: " << num_pivots << "\n"
                  << "  Vertices certified: " << num_certified << "\n"
                  << "  Dijkstra calls: " << num_dijkstra_calls << "\n"
                  << "  Edges relaxed: " << total_edges_relaxed << "\n"
                  << "  Time init: " << time_init << "s\n"
                  << "  Time levels: " << time_levels << "s\n"
                  << "  Time total: " << time_total << "s\n";
    }
};

// ============================================================================
// Main Engine
// ============================================================================

class AHSSSPEngine {
private:
    const Graph& graph_;
    VertexId source_;
    size_t n_;
    size_t m_;
    
    Config config_;
    std::vector<Vertex> vertices_;
    std::vector<Pivot> pivots_;
    std::unordered_map<VertexId, PivotId> pivot_index_;
    std::unique_ptr<HierarchicalQueue> hqueue_;
    
    Statistics stats_;
    
public:
    AHSSSPEngine(const Graph& graph, VertexId source) 
        : graph_(graph), source_(source), 
          n_(graph.num_vertices()), m_(graph.num_edges()),
          config_(n_, graph.min_edge_weight()),
          vertices_(n_) {
        
        std::cout << "Initialized AH-SSSP:\n"
                  << "  n=" << n_ << ", m=" << m_ 
                  << ", k=" << config_.k 
                  << ", levels=" << config_.num_levels << "\n";
    }
    
    // Main computation
    void compute() {
        auto start = std::chrono::high_resolution_clock::now();
        
        initialize();
        
        // Process levels
        for (uint8_t level = 0; level < config_.num_levels; level++) {
            auto level_start = std::chrono::high_resolution_clock::now();
            
            process_level(level);
            
            auto level_end = std::chrono::high_resolution_clock::now();
            double level_time = std::chrono::duration<double>(level_end - level_start).count();
            stats_.level_times.push_back(level_time);
            stats_.time_levels += level_time;
            
            std::cout << "Level " << (int)level << ": " 
                      << stats_.num_certified << "/" << n_ << " certified, "
                      << "time=" << level_time << "s\n";
            
            // Early termination
            if (config_.enable_early_termination && 
                certified_fraction() > config_.early_term_threshold) {
                std::cout << "Early termination: >" 
                          << (config_.early_term_threshold * 100) 
                          << "% certified\n";
                finish_with_dijkstra();
                break;
            }
        }
        
        finalize();
        
        auto end = std::chrono::high_resolution_clock::now();
        stats_.time_total = std::chrono::duration<double>(end - start).count();
        
        std::cout << "\nComputation complete: time=" << stats_.time_total << "s\n";
    }
    
    // Query interface
    double distance(VertexId v) const {
        return v < n_ ? vertices_[v].d_exact : INF;
    }
    
    std::vector<VertexId> path(VertexId target) const {
        if (target >= n_ || std::isinf(vertices_[target].d_exact)) {
            return {};
        }
        
        std::vector<VertexId> path;
        VertexId v = target;
        path.push_back(v);
        
        while (v != source_) {
            v = vertices_[v].parent;
            path.push_back(v);
        }
        
        std::reverse(path.begin(), path.end());
        return path;
    }
    
    const Statistics& statistics() const { return stats_; }
    
private:
    void initialize() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Initialize source
        vertices_[source_].d_lower = 0.0;
        vertices_[source_].d_upper = 0.0;
        vertices_[source_].d_exact = 0.0;
        vertices_[source_].level = config_.num_levels;
        vertices_[source_].certified = true;
        
        // Bellman-Ford k-sweep
        std::vector<VertexId> active = {source_};
        
        for (size_t round = 0; round < config_.k; round++) {
            std::vector<VertexId> next_active;
            
            for (VertexId u : active) {
                double d_u = vertices_[u].d_upper;
                
                for (const auto& edge : graph_.out_edges(u)) {
                    VertexId v = edge.target;
                    double d_new = d_u + edge.weight;
                    
                    if (d_new < vertices_[v].d_upper) {
                        vertices_[v].d_upper = d_new;
                        vertices_[v].d_lower = d_new;
                        vertices_[v].parent = u;
                        next_active.push_back(v);
                        stats_.total_edges_relaxed++;
                    }
                }
            }
            
            active = std::move(next_active);
        }
        
        // Certify k-hop neighborhood
        for (VertexId v : active) {
            vertices_[v].d_exact = vertices_[v].d_upper;
            vertices_[v].level = 0;
            vertices_[v].certified = true;
            stats_.num_certified++;
        }
        
        // Build pivot hierarchy
        build_pivot_hierarchy();
        
        // Initialize hierarchical queue
        hqueue_ = std::make_unique<HierarchicalQueue>(config_.k, config_.num_levels);
        
        for (VertexId v = 0; v < n_; v++) {
            if (!vertices_[v].certified) {
                uint8_t level = distance_to_level(vertices_[v].d_upper);
                hqueue_->insert(v, vertices_[v].d_upper, level);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        stats_.time_init = std::chrono::duration<double>(end - start).count();
        
        std::cout << "Initialization complete: " 
                  << active.size() << " vertices certified, "
                  << stats_.num_pivots << " pivots, "
                  << "time=" << stats_.time_init << "s\n";
    }
    
    void build_pivot_hierarchy() {
        // Compute subtree sizes
        std::vector<size_t> subtree_sizes = compute_subtree_sizes();
        
        // Identify pivots at each level
        for (uint8_t level = 0; level < config_.num_levels; level++) {
            size_t min_size = std::pow(config_.k, level + 1);
            
            for (VertexId v = 0; v < n_; v++) {
                if (subtree_sizes[v] >= min_size && 
                    pivot_index_.find(v) == pivot_index_.end()) {
                    
                    double radius = compute_subtree_radius(v);
                    Pivot pivot(v, level, subtree_sizes[v], radius);
                    
                    PivotId pid = pivots_.size();
                    pivots_.push_back(pivot);
                    pivot_index_[v] = pid;
                    stats_.num_pivots++;
                }
            }
        }
        
        // Link pivot parents
        link_pivot_parents();
    }
    
    std::vector<size_t> compute_subtree_sizes() {
        std::vector<size_t> sizes(n_, 1);
        std::vector<bool> visited(n_, false);
        
        std::function<size_t(VertexId)> dfs = [&](VertexId v) -> size_t {
            visited[v] = true;
            size_t size = 1;
            
            for (VertexId u = 0; u < n_; u++) {
                if (!visited[u] && vertices_[u].parent == v) {
                    size += dfs(u);
                }
            }
            
            sizes[v] = size;
            return size;
        };
        
        dfs(source_);
        return sizes;
    }
    
    double compute_subtree_radius(VertexId root) {
        double max_dist = 0.0;
        std::vector<bool> visited(n_, false);
        std::queue<VertexId> q;
        q.push(root);
        
        while (!q.empty()) {
            VertexId v = q.front();
            q.pop();
            
            if (visited[v]) continue;
            visited[v] = true;
            
            max_dist = std::max(max_dist, vertices_[v].d_upper);
            
            for (VertexId u = 0; u < n_; u++) {
                if (!visited[u] && vertices_[u].parent == v) {
                    q.push(u);
                }
            }
        }
        
        return max_dist;
    }
    
    void link_pivot_parents() {
        for (size_t i = 0; i < pivots_.size(); i++) {
            for (size_t j = 0; j < pivots_.size(); j++) {
                if (pivots_[j].level > pivots_[i].level) {
                    if (is_ancestor(pivots_[j].root, pivots_[i].root)) {
                        pivots_[i].parent_pivot = j;
                        break;
                    }
                }
            }
        }
    }
    
    bool is_ancestor(VertexId ancestor, VertexId descendant) {
        VertexId v = descendant;
        while (v != source_) {
            if (v == ancestor) return true;
            v = vertices_[v].parent;
        }
        return ancestor == source_;
    }
    
    void process_level(uint8_t level) {
        size_t max_iterations = 1000;
        size_t iterations = 0;
        
        while (hqueue_->has_vertices_at_level(level) && iterations < max_iterations) {
            // Select pivot batch
            std::vector<PivotId> pivot_batch = select_pivot_batch(level);
            
            if (pivot_batch.empty()) break;
            
            // Process pivots
            for (PivotId pid : pivot_batch) {
                process_pivot(pid, level);
            }
            
            // Contract frontier
            contract_frontier(level);
            
            // Promote vertices
            promote_vertices(level);
            
            iterations++;
        }
    }
    
    std::vector<PivotId> select_pivot_batch(uint8_t level) {
        std::vector<PivotId> batch;
        for (size_t i = 0; i < pivots_.size() && batch.size() < config_.k; i++) {
            if (pivots_[i].level == level) {
                batch.push_back(i);
            }
        }
        return batch;
    }
    
    void process_pivot(PivotId pid, uint8_t level) {
        const VertexId root = pivots_[pid].root;
        const size_t max_steps = config_.k;
        
        // Priority queue: (distance, vertex)
        using PQItem = std::pair<double, VertexId>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        
        pq.emplace(0.0, root);
        std::unordered_set<VertexId> visited;
        size_t steps = 0;
        
        stats_.num_dijkstra_calls++;
        
        while (!pq.empty() && steps < max_steps) {
            auto [d_u, u] = pq.top();
            pq.pop();
            
            if (visited.count(u)) continue;
            visited.insert(u);
            steps++;
            
            // Certify u
            if (vertices_[u].level < level) {
                vertices_[u].d_exact = d_u;
                vertices_[u].level = level;
                vertices_[u].pivot = pid;
                vertices_[u].certified = true;
                stats_.num_certified++;
                
                pivots_[pid].distances.emplace_back(u, d_u);
            }
            
            // Relax edges
            for (const auto& edge : graph_.out_edges(u)) {
                VertexId v = edge.target;
                double d_new = d_u + edge.weight;
                
                if (d_new < vertices_[v].d_upper) {
                    vertices_[v].d_upper = d_new;
                    vertices_[v].parent = u;
                    pq.emplace(d_new, v);
                    stats_.total_edges_relaxed++;
                }
            }
        }
    }
    
    void contract_frontier(uint8_t level) {
        size_t min_subtree_size = std::pow(config_.k, level + 1);
        
        for (VertexId v = 0; v < n_; v++) {
            if (vertices_[v].level == level && 
                pivot_index_.find(v) == pivot_index_.end()) {
                
                size_t subtree_size = count_certified_descendants(v, level);
                
                if (subtree_size >= min_subtree_size) {
                    double radius = compute_subtree_radius(v);
                    Pivot pivot(v, level + 1, subtree_size, radius);
                    
                    PivotId pid = pivots_.size();
                    pivots_.push_back(pivot);
                    pivot_index_[v] = pid;
                    stats_.num_pivots++;
                }
            }
        }
    }
    
    size_t count_certified_descendants(VertexId root, uint8_t level) {
        size_t count = 0;
        std::vector<bool> visited(n_, false);
        std::queue<VertexId> q;
        q.push(root);
        
        while (!q.empty()) {
            VertexId v = q.front();
            q.pop();
            
            if (visited[v]) continue;
            visited[v] = true;
            
            if (vertices_[v].level >= level) {
                count++;
            }
            
            for (VertexId u = 0; u < n_; u++) {
                if (!visited[u] && vertices_[u].parent == v) {
                    q.push(u);
                }
            }
        }
        
        return count;
    }
    
    void promote_vertices(uint8_t level) {
        if (level + 1 >= config_.num_levels) return;
        
        for (VertexId v = 0; v < n_; v++) {
            if (vertices_[v].level == level && !vertices_[v].certified) {
                hqueue_->remove(v, level);
                hqueue_->insert(v, vertices_[v].d_upper, level + 1);
            }
        }
    }
    
    void finish_with_dijkstra() {
        using PQItem = std::pair<double, VertexId>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        
        // Seed with certified vertices
        for (VertexId v = 0; v < n_; v++) {
            if (vertices_[v].certified) {
                pq.emplace(vertices_[v].d_exact, v);
            }
        }
        
        while (!pq.empty()) {
            auto [d_u, u] = pq.top();
            pq.pop();
            
            // Skip if already certified with better distance
            if (vertices_[u].certified && vertices_[u].d_exact < d_u) {
                continue;
            }
            
            if (!vertices_[u].certified) {
                vertices_[u].d_exact = d_u;
                vertices_[u].d_upper = d_u;
                vertices_[u].certified = true;
                stats_.num_certified++;
            }
            
            // Relax edges
            for (const auto& edge : graph_.out_edges(u)) {
                VertexId v = edge.target;
                double d_new = d_u + edge.weight;
                
                if (d_new < vertices_[v].d_upper) {
                    vertices_[v].d_upper = d_new;
                    vertices_[v].parent = u;
                    pq.emplace(d_new, v);
                    stats_.total_edges_relaxed++;
                }
            }
        }
    }
    
    void finalize() {
        // Check for uncertified vertices
        size_t uncertified = 0;
        for (VertexId v = 0; v < n_; v++) {
            if (!vertices_[v].certified) {
                uncertified++;
            }
        }
        
        if (uncertified > 0) {
            std::cout << "Finalizing " << uncertified 
                      << " uncertified vertices with Dijkstra\n";
            finish_with_dijkstra();
        }
        
        // Final pass
        for (VertexId v = 0; v < n_; v++) {
            if (!vertices_[v].certified) {
                vertices_[v].d_exact = vertices_[v].d_upper;
                vertices_[v].certified = true;
                stats_.num_certified++;
            }
        }
    }
    
    double certified_fraction() const {
        return static_cast<double>(stats_.num_certified) / n_;
    }
    
    uint8_t distance_to_level(double dist) const {
        if (std::isinf(dist)) {
            return config_.num_levels - 1;
        }
        double level = std::log(std::max(1.0, dist / config_.delta)) / std::log(config_.k);
        return std::min(static_cast<uint8_t>(level), 
                       static_cast<uint8_t>(config_.num_levels - 1));
    }
};

} // namespace ahsssp

#endif // AHSSSP_HPP
