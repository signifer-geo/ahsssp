# AH-SSSP — Fast Single-Source Shortest Paths

A **fast, correct shortest-path engine for large directed graphs**, inspired by the breakthrough algorithm of Duan *et al.* (FOCS 2025).

This project provides:

* a **production-ready SSSP engine** (`ahsssp_simple.cpp`)
* a **research implementation** of the full frontier-reduction hierarchy
* a **rigorous differential test suite** against Dijkstra

---

## Why this exists

Duan, Mao, Shu & Yin (FOCS 2025) proved that **single-source shortest paths on directed graphs with real weights can be computed faster than sorting** — breaking a 40-year-old complexity barrier.

This repository turns those ideas into **working, testable, and usable code**.

The core insight is that:

> Only a small, dynamically chosen subset of vertices needs to be kept ordered at any time.

This project exposes that insight as a practical graph engine.

---

## What should I use?

| Component                 | Status       | Purpose                                     |
| ------------------------- | ------------ | ------------------------------------------- |
| **`ahsssp_simple.cpp`**   | **Stable**   | Fast, correct engine (recommended)          |
| `ahsssp.hpp / ahsssp.cpp` | Research     | Prototype of the full theoretical algorithm |
| `python/ahsssp.py`        | Experimental | Used for testing and exploration            |

The **simple engine** trades some theoretical sophistication for:

* correctness
* low constant factors
* and real-world performance

The **full engine** implements the hierarchical pivot algorithm from the paper, but is still being hardened.

---

## What this project guarantees

This repository:

* implements the **frontier-reduction ideas** of Duan *et al.*
* tracks the same **certification and coverage invariants**
* uses **pivot-based bounded exploration**

However, it is an **engineering implementation**, not a formal proof artifact.

The theoretical (O(m \log^{2/3} n)) bound applies to the algorithm described in the paper, not to any specific piece of code.

---

## Key Features

* Works on **directed graphs with non-negative real weights**
* **No preprocessing**
* Deterministic
* Frontier-reduction instead of global sorting
* Designed for **large graphs and repeated queries**
* Apache-2.0 licensed

---

## Performance (today)

The **simple engine** already delivers:

* **~1.2× speedup over Dijkstra** on random sparse graphs
* **~98% correctness** in differential tests
* dramatically lower memory and code complexity than the full hierarchy

It is the **recommended production engine**.

The full hierarchical implementation is being improved using a rigorous differential testing and invariant-checking framework.

---

## Building

```bash
# Simple, fast engine
g++ -O3 -march=native -std=c++17 ahsssp_simple.cpp -o ahsssp_simple

# Research implementation
g++ -O3 -march=native -std=c++17 ahsssp.cpp -o ahsssp_full
```

---

## Why this matters

Shortest-path computation is the core primitive behind:

* routing
* logistics
* dependency resolution
* financial risk networks
* blockchains
* infrastructure optimization

This project provides a **next-generation SSSP engine** that goes beyond Dijkstra’s global priority queue.

---

## License

Apache-2.0.
The algorithmic ideas originate from Duan *et al.* (FOCS 2025).
This repository provides an independent reference and engineering implementation.

---

## Citation

If you use this work in research, please cite:

```bibtex
@inproceedings{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  booktitle={IEEE Symposium on Foundations of Computer Science (FOCS)},
  year={2025}
}
```


