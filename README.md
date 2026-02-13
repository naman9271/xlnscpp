# XLNSCPP

**C++ Logarithmic Number System (LNS) Library — 16-bit & 32-bit**

[![CI](https://github.com/xlnsresearch/xlnscpp/actions/workflows/ci.yml/badge.svg)](https://github.com/xlnsresearch/xlnscpp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

XLNSCPP is a lightweight, header-installable C++ library for Logarithmic Number System arithmetic.  It provides 16-bit and 32-bit LNS types with overloaded operators (class API) and a low-level function API, suitable for research, embedded DSP, and as a backend for ML inference frameworks.

---

## Features

| Feature | 16-bit (`xlns16`) | 32-bit (`xlns32`) |
|---|---|---|
| Signed LNS type | ✅ | ✅ |
| Operator-overloaded class | `xlns16_float` | `xlns32_float` |
| Conversion to/from `float` | ✅ | ✅ |
| Ideal sb/db (math.h) | `XLNS16_IDEAL` | `XLNS32_IDEAL` |
| Table-based sb/db + conversion | `XLNS16_TABLE` | cotransformation (default) |
| LPVIP / Mitchell approximation | default | — |
| Math functions (sqrt, exp, log, pow, sin, cos, atan) | ✅ | ✅ |

### Internal representation

Unlike IEEE 754, the internal representation is **not** twos complement — it is offset by a constant (`xlns16_logsignmask` / `xlns32_logsignmask`).

- **16-bit** — 1 sign + 8 `int(log₂)` + 7 `frac(log₂)` bits  (similar footprint to `bfloat16`)
- **32-bit** — 1 sign + 8 `int(log₂)` + 23 `frac(log₂)` bits  (similar footprint to `float`)

Exact 0.0 is supported.  No subnormals or NaNs.

---

## Quick start

```bash
# clone
git clone https://github.com/xlnsresearch/xlnscpp.git
cd xlnscpp

# build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# test
ctest --test-dir build --output-on-failure
```

### Use in your project (CMake `find_package`)

After installing the library (`cmake --install build`), you can use it from any CMake project:

```cmake
find_package(xlns REQUIRED)
target_link_libraries(my_target PRIVATE xlns::xlns)
```

```cpp
#include <xlns/xlns.h>      // convenience: includes xlns32.h + xlns16.h

xlns32_float a(3.14f), b(2.71f);
auto c = a + b;             // LNS addition
```

---

## CMake options

| Option | Default | Description |
|---|---|---|
| `XLNS_BUILD_TESTS` | `ON` | Build the test suite |
| `XLNS_BUILD_BENCHMARKS` | `ON` | Build performance benchmarks |
| `XLNS_BUILD_EXAMPLES` | `ON` | Build example programs |
| `XLNS_INSTALL` | `ON` | Generate install targets |
| `XLNS32_IDEAL` | `OFF` | Use math.h for 32-bit sb/db (most accurate) |
| `XLNS32_ALT` | `OFF` | Alternative 32-bit addition algorithm |
| `XLNS16_IDEAL` | `OFF` | Use math.h for 16-bit sb/db (most accurate) |
| `XLNS16_ALT` | `ON` | Alternative 16-bit addition algorithm |
| `XLNS16_ALTOPT` | `OFF` | Simplified LPVIP within alt (less accurate) |
| `XLNS16_TABLE` | `OFF` | Use lookup tables for 16-bit conversions & sb/db |

Example:
```bash
cmake -B build -DXLNS32_IDEAL=ON -DXLNS16_TABLE=ON
```

---

## Directory structure

```
xlnscpp/
├── CMakeLists.txt            # root build file
├── cmake/
│   └── xlnsConfig.cmake.in   # find_package support
├── include/xlns/
│   ├── xlns.h                # convenience header
│   ├── xlns32.h              # 32-bit public API
│   └── xlns16.h              # 16-bit public API
├── src/
│   ├── xlns32.cpp            # 32-bit implementation
│   ├── xlns16.cpp            # 16-bit implementation
│   └── tables/               # precomputed lookup tables
│       ├── xlns32tbl.h
│       ├── xlns16sbdbtbl.h
│       ├── xlns16cvtbl.h
│       └── xlns16revcvtbl.h
├── test/
│   ├── CMakeLists.txt
│   ├── unit/                 # automated test suites (CTest)
│   ├── benchmark/            # performance benchmarks
│   └── cross-validation/     # Python ↔ C++ comparison tools
├── examples/
│   └── basic_usage.cpp
├── docs/                     # design & API documentation
├── legacy/                   # original flat-file sources
└── .github/workflows/ci.yml
```

---

## API overview

### Class API (overloaded operators)

```cpp
xlns32_float a(3.14f), b(2.0f);
xlns32_float c = a + b;       // addition
xlns32_float d = a * b;       // multiplication (= log addition)
xlns32_float e = a / b;       // division
bool gt = (a > b);            // comparison

// Math
xlns32_float s = xlns32_sqrt(a);
xlns32_float ex = xlns32_exp(a);
xlns32_float l = xlns32_log(a);
```

### Function API

```cpp
xlns32 ra = fp2xlns32(3.14);
xlns32 rb = fp2xlns32(2.0);
xlns32 rc = xlns32_add(ra, rb);
double result = xlns322fp(rc);
```

---

## Cross-validation

The `test/cross-validation/` directory contains CLI tools (`sb16`, `db16`, `sbmit16`, `dbmit16`) and Python scripts that compare Gaussian Log computations between this C++ library and the Python [xlns](https://github.com/xlnsresearch/xlns) library.

```bash
# after building
cd build/test
python3 sbtest.py    # compares ideal sb (C++ vs Python)
python3 sblptest.py  # compares Mitchell/LPVIP sb (C++ vs Python)
```

---

## References

- M. G. Arnold, et al. "Arithmetic cotransformations in the Real and Complex Logarithmic Number Systems," *IEEE Trans. Comput.*, vol. 47, no. 7, pp. 777–786, July 1998.
- M. G. Arnold, "LPVIP: A Low-power ROM-Less ALU for Low-Precision LNS," *14th International Workshop on Power and Timing Modeling, Optimization and Simulation*, LNCS 3254, pp. 675–684, Santorini, Greece, Sept. 2004.

## Sister repository

[xlnscuda](https://github.com/xlnsresearch/xlnscuda) — related routines for CUDA devices.

## License

MIT — Copyright © 1999–2025 Mark G. Arnold
