#include <cassert>
#include <cmath>
#include <iostream>

// Include full engine header
#include "ahsssp.hpp"

int main() {
    using namespace ahsssp;

    // --- Build a small chain graph: 0->1->2->...->9 weight 1 ---
    const unsigned n = 10;
    Graph g(n);                // <-- adjust if your Graph type differs
    for (unsigned i = 0; i + 1 < n; ++i) {
        g.add_edge(i, i + 1, 1.0);  // <-- adjust if API differs
    }

    AHSSSPEngine engine(g, 0);
    engine.compute();         // <-- adjust if API differs

    double d9 = engine.distance(9);
    if (std::isinf(d9) || std::fabs(d9 - 9.0) > 1e-9) {
        std::cerr << "Full engine smoke failed: dist[9]=" << d9 << " expected 9\n";
        return 1;
    }

    std::cout << "Full engine smoke OK: dist[9]=" << d9 << "\n";
    return 0;
}

