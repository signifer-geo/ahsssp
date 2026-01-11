#!/usr/bin/env python3
"""
Differential Testing Suite for AH-SSSP
Property-based fuzzing + adversarial test cases
"""



import sys
import random
import math
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass

# Import our implementation
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
from ahsssp import Graph, AHSSSPEngine, dijkstra_baseline

@dataclass
class TestResult:
    passed: bool
    test_name: str
    graph_size: Tuple[int, int]
    error_msg: str = ""
    max_error: float = 0.0
    time_dijkstra: float = 0.0
    time_ahsssp: float = 0.0

class DifferentialTester:
    """Comprehensive differential testing against Dijkstra baseline"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
    
    def verify_distances(self, graph: Graph, source: int, 
                        dist_reference: List[float], 
                        dist_test: List[float],
                        tolerance: float = 1e-6) -> Tuple[bool, float, str]:
        """Compare distances with proper handling of unreachable vertices"""
        max_error = 0.0
        errors = []
        
        for v in range(graph.num_vertices()):
            d_ref = dist_reference[v]
            d_test = dist_test[v]
            
            # Both should agree on reachability
            if math.isinf(d_ref) and math.isinf(d_test):
                continue
            
            if math.isinf(d_ref) != math.isinf(d_test):
                errors.append(f"Reachability mismatch at {v}: ref={d_ref}, test={d_test}")
                continue
            
            error = abs(d_ref - d_test)
            max_error = max(max_error, error)
            
            if error > tolerance:
                errors.append(f"Distance error at {v}: ref={d_ref:.6f}, test={d_test:.6f}, error={error:.6f}")
        
        if errors:
            return False, max_error, "; ".join(errors[:5])  # First 5 errors
        return True, max_error, ""
    
    def run_test(self, test_name: str, graph: Graph, source: int) -> TestResult:
        """Run single differential test"""
        self.total_tests += 1
        n, m = graph.num_vertices(), graph.num_edges()
        
        self.log(f"{test_name}: n={n}, m={m}, source={source}")
        
        try:
            # Run Dijkstra (baseline)
            start = time.time()
            dist_dijkstra, _ = dijkstra_baseline(graph, source)
            time_dijkstra = time.time() - start
            
            # Run AH-SSSP
            start = time.time()
            engine = AHSSSPEngine(graph, source)
            engine.compute()
            time_ahsssp = time.time() - start
            
            # Get distances from AH-SSSP
            dist_ahsssp = [engine.distance(v) for v in range(n)]
            
            # Verify
            passed, max_error, error_msg = self.verify_distances(
                graph, source, dist_dijkstra, dist_ahsssp
            )
            
            if passed:
                self.passed_tests += 1
                self.log(f"✓ PASS (max_error={max_error:.2e})")
            else:
                self.log(f"✗ FAIL: {error_msg}")
            
            return TestResult(
                passed=passed,
                test_name=test_name,
                graph_size=(n, m),
                error_msg=error_msg,
                max_error=max_error,
                time_dijkstra=time_dijkstra,
                time_ahsssp=time_ahsssp
            )
            
        except Exception as e:
            self.log(f"✗ EXCEPTION: {str(e)}")
            return TestResult(
                passed=False,
                test_name=test_name,
                graph_size=(n, m),
                error_msg=f"Exception: {str(e)}"
            )
    
    # =========================================================================
    # Graph Generators - Adversarial Cases
    # =========================================================================
    
    def gen_equal_lengths(self, n: int, num_paths: int) -> Graph:
        """Many equal-length paths (tests tie-breaking)"""
        graph = Graph(n)
        source = 0
        target = n - 1
        
        # Create num_paths parallel paths of equal length
        nodes_per_path = (n - 2) // num_paths
        
        for path_id in range(num_paths):
            prev = source
            for i in range(nodes_per_path):
                curr = 1 + path_id * nodes_per_path + i
                if curr < target:
                    graph.add_edge(prev, curr, 1.0)
                    prev = curr
            graph.add_edge(prev, target, 1.0)
        
        return graph
    
    def gen_zero_weights(self, n: int) -> Graph:
        """Graph with many zero-weight edges (plateau creation)"""
        graph = Graph(n)
        
        # Create clusters connected by zero weights
        cluster_size = max(2, n // 10)
        
        for i in range(n - 1):
            # Connect to next
            weight = 0.0 if i % cluster_size != cluster_size - 1 else 1.0
            graph.add_edge(i, i + 1, weight)
            
            # Add some back edges with zero weight
            if i > 0 and random.random() < 0.1:
                graph.add_edge(i, i - 1, 0.0)
        
        return graph
    
    def gen_star_graph(self, n: int) -> Graph:
        """Star: center connected to all leaves"""
        graph = Graph(n)
        center = 0
        
        for v in range(1, n):
            graph.add_edge(center, v, float(v))
            graph.add_edge(v, center, float(v))
        
        return graph
    
    def gen_long_chain(self, n: int) -> Graph:
        """Long chain (worst case for k-hop certification)"""
        graph = Graph(n)
        
        for i in range(n - 1):
            graph.add_edge(i, i + 1, 1.0 + random.random() * 0.1)
        
        return graph
    
    def gen_almost_equal_distances(self, n: int) -> Graph:
        """Many nodes with very similar distances"""
        graph = Graph(n)
        
        # Binary tree with small weight variations
        for i in range(n // 2):
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n:
                graph.add_edge(i, left, 1.0 + random.random() * 0.001)
            if right < n:
                graph.add_edge(i, right, 1.0 + random.random() * 0.001)
        
        return graph
    
    def gen_dense_complete(self, n: int) -> Graph:
        """Dense complete graph"""
        graph = Graph(n)
        
        for u in range(n):
            for v in range(n):
                if u != v:
                    graph.add_edge(u, v, random.random() * 10.0 + 1.0)
        
        return graph
    
    def gen_sparse_random(self, n: int, avg_degree: int = 4) -> Graph:
        """Sparse random graph"""
        graph = Graph(n)
        m = n * avg_degree
        
        for _ in range(m):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                graph.add_edge(u, v, random.random() * 100.0 + 1.0)
        
        return graph
    
    def gen_grid(self, rows: int, cols: int) -> Graph:
        """2D grid graph"""
        n = rows * cols
        graph = Graph(n)
        
        idx = lambda r, c: r * cols + c
        
        for r in range(rows):
            for c in range(cols):
                u = idx(r, c)
                
                if c + 1 < cols:
                    v = idx(r, c + 1)
                    graph.add_edge(u, v, 1.0)
                    graph.add_edge(v, u, 1.0)
                
                if r + 1 < rows:
                    v = idx(r + 1, c)
                    graph.add_edge(u, v, 1.0)
                    graph.add_edge(v, u, 1.0)
        
        return graph
    
    def gen_disconnected(self, n: int, num_components: int) -> Graph:
        """Graph with disconnected components"""
        graph = Graph(n)
        component_size = n // num_components
        
        for comp in range(num_components):
            start = comp * component_size
            end = min(start + component_size, n)
            
            # Make each component a chain
            for i in range(start, end - 1):
                graph.add_edge(i, i + 1, 1.0)
        
        return graph
    
    # =========================================================================
    # Test Suites
    # =========================================================================
    
    def run_adversarial_suite(self):
        """Run all adversarial test cases"""
        print("\n" + "="*70)
        print("ADVERSARIAL TEST SUITE")
        print("="*70)
        
        tests = [
            ("Equal-length paths (n=50)", lambda: self.gen_equal_lengths(50, 5)),
            ("Zero weights (n=100)", lambda: self.gen_zero_weights(100)),
            ("Star graph (n=100)", lambda: self.gen_star_graph(100)),
            ("Long chain (n=200)", lambda: self.gen_long_chain(200)),
            ("Almost equal distances (n=127)", lambda: self.gen_almost_equal_distances(127)),
            ("Dense complete (n=50)", lambda: self.gen_dense_complete(50)),
            ("Sparse random (n=500)", lambda: self.gen_sparse_random(500, 4)),
            ("Grid 20x20", lambda: self.gen_grid(20, 20)),
            ("Disconnected (n=200)", lambda: self.gen_disconnected(200, 4)),
        ]
        
        for test_name, gen_func in tests:
            graph = gen_func()
            source = 0
            result = self.run_test(test_name, graph, source)
            self.results.append(result)
    
    def run_random_fuzzing(self, num_tests: int = 100, max_n: int = 200):
        """Random fuzzing with various graph types"""
        print("\n" + "="*70)
        print(f"RANDOM FUZZING SUITE ({num_tests} tests)")
        print("="*70)
        
        for i in range(num_tests):
            n = random.randint(10, max_n)
            graph_type = random.choice([
                'sparse', 'dense', 'grid', 'chain', 'star'
            ])
            
            if graph_type == 'sparse':
                avg_degree = random.randint(2, 8)
                graph = self.gen_sparse_random(n, avg_degree)
            elif graph_type == 'dense':
                graph = self.gen_dense_complete(min(n, 50))
                n = min(n, 50)
            elif graph_type == 'grid':
                side = int(math.sqrt(n))
                graph = self.gen_grid(side, side)
                n = side * side
            elif graph_type == 'chain':
                graph = self.gen_long_chain(n)
            else:  # star
                graph = self.gen_star_graph(n)
            
            source = random.randint(0, n - 1)
            
            test_name = f"Random-{i+1} ({graph_type}, n={n})"
            if not self.verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            result = self.run_test(test_name, graph, source)
            self.results.append(result)
    
    def run_scaling_tests(self):
        """Test at different scales"""
        print("\n" + "="*70)
        print("SCALING TEST SUITE")
        print("="*70)
        
        sizes = [100, 500, 1000, 2000]
        
        for n in sizes:
            # Grid
            side = int(math.sqrt(n))
            graph = self.gen_grid(side, side)
            result = self.run_test(f"Grid-{side}x{side}", graph, 0)
            self.results.append(result)
            
            # Sparse random
            graph = self.gen_sparse_random(n, 4)
            result = self.run_test(f"Sparse-random-{n}", graph, 0)
            self.results.append(result)
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Pass rate: {100 * self.passed_tests / max(1, self.total_tests):.1f}%")
        
        # Failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for r in failed[:10]:  # Show first 10
                print(f"  ✗ {r.test_name}: {r.error_msg[:100]}")
        
        # Performance summary
        print("\nPerformance summary:")
        perf_results = [r for r in self.results if r.passed and r.time_dijkstra > 0]
        
        if perf_results:
            avg_speedup = sum(r.time_dijkstra / r.time_ahsssp 
                            for r in perf_results) / len(perf_results)
            print(f"  Average speedup vs Dijkstra: {avg_speedup:.2f}x")
            
            # Best/worst cases
            best = max(perf_results, key=lambda r: r.time_dijkstra / r.time_ahsssp)
            worst = min(perf_results, key=lambda r: r.time_dijkstra / r.time_ahsssp)
            
            print(f"  Best case: {best.test_name} ({best.time_dijkstra/best.time_ahsssp:.2f}x)")
            print(f"  Worst case: {worst.test_name} ({worst.time_dijkstra/worst.time_ahsssp:.2f}x)")

def main():
    print("AH-SSSP Differential Testing Suite")
    print("Testing against Dijkstra baseline\n")
    
    random.seed(42)  # Reproducible tests
    
    tester = DifferentialTester(verbose=True)
    
    # Run test suites
    tester.run_adversarial_suite()
    tester.run_random_fuzzing(num_tests=50, max_n=100)
    tester.run_scaling_tests()
    
    # Print summary
    tester.print_summary()
    
    # Exit code based on results
    sys.exit(0 if tester.passed_tests == tester.total_tests else 1)

if __name__ == "__main__":
    main()
