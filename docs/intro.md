# XLNSCPP — Logarithmic Number System Library for C++

## Introduction

**XLNSCPP** is an open-source C++ library for **Logarithmic Number System (LNS)** arithmetic, providing both 16-bit and 32-bit implementations. It was originally written by Mark G. Arnold (copyright 1999–2025) and is released under the MIT License.

In a Logarithmic Number System, every non-zero real number is stored as the **logarithm (base 2) of its absolute value** plus a **sign bit**. This representation makes **multiplication and division trivially cheap** (they become integer addition and subtraction of the logarithms), at the expense of making **addition and subtraction more complex** (they require computing the Gaussian logarithms `sb` and `db`).

XLNSCPP is the C++ sibling of the [Python xlns](https://github.com/xlnsresearch/xlns) package and shares the same mathematical foundation. However, the internal storage format differs: XLNSCPP uses an **offset** (biased) representation rather than two's complement for the integer part of the logarithm.

---

## Why LNS?

| Property | Floating Point (FP) | Logarithmic Number System (LNS) |
|---|---|---|
| Multiplication | Hardware multiplier needed | Integer addition (**very cheap**) |
| Division | Hardware divider needed | Integer subtraction (**very cheap**) |
| Addition | Simple hardware adder | Gaussian log computation (**expensive**) |
| Storage | Sign + Exponent + Mantissa | Sign + log₂(magnitude) |
| Power consumption | Higher for mul-heavy workloads | Lower for mul-heavy workloads |

LNS is especially attractive for workloads dominated by multiplication, such as deep neural network inference, where power efficiency is critical and approximate results are acceptable.

---

## Data Format

Both the 16-bit and 32-bit variants use the same logical layout:

```
+------+----------------------------+
| sign |  int(log₂) . frac(log₂)   |
+------+----------------------------+
  1 bit     8 bits       F bits
```

| Variant | Total Bits | `int(log₂)` Bits | `frac(log₂)` Bits (F) | FP Analogy | Scale Constant |
|---------|-----------|-------------------|------------------------|------------|----------------|
| `xlns16` | 16 | 8 | 7 | `bfloat16` | 128 (2⁷) |
| `xlns32` | 32 | 8 | 23 | IEEE `float32` | 8388608 (2²³) |

**Key properties:**
- The `int(log₂)` part is **not** two's complement; it is **offset** by `logsignmask` (0x4000 for 16-bit, 0x40000000 for 32-bit).
- There is an **exact representation of 0.0** (the all-zeros pattern).
- There are **no subnormals or NaNs**.

---

## Two Usage Modes

### 1. Low-Level Function API (Faster)

Operate directly on the raw integer representation (`xlns16` / `xlns32` typedefs):

```cpp
#include "xlns32.cpp"

xlns32 a = fp2xlns32(3.14f);   // float → LNS32
xlns32 b = fp2xlns32(2.71f);

xlns32 product = xlns32_mul(a, b);   // Cheap: integer add
xlns32 quotient = xlns32_div(a, b);  // Cheap: integer sub
xlns32 sum = xlns32_add(a, b);       // Expensive: Gaussian log

float result = xlns322fp(sum);       // LNS32 → float
printf("3.14 + 2.71 ≈ %f\n", result);
```

### 2. Operator-Overloaded Class API (Easier)

Use the `xlns16_float` or `xlns32_float` classes with natural C++ syntax:

```cpp
#include "xlns32.cpp"

xlns32_float a, b, c;
a = 3.14f;
b = 2.71f;

c = a + b;          // Uses xlns32_add internally
c = a * b;          // Uses xlns32_mul internally
c = a / b;          // Uses xlns32_div internally

// Mixed-mode with float
c = a + 1.0f;       // Auto-converts 1.0f to LNS
c = 2.0f * b;       // Auto-converts 2.0f to LNS

// Comparisons
if (a > b) { /* ... */ }

// Math functions
c = sin(a);          // Computed via: convert to FP → sin → convert back
c = sqrt(a);         // Efficient: halve the log magnitude

// Stream output
std::cout << "c = " << c << std::endl;
```

---

## Compile-Time Options

The library's behavior is controlled by preprocessor macros defined **before** including `xlns16.cpp` or `xlns32.cpp`:

| Macro | Applies To | Default | Effect |
|-------|-----------|---------|--------|
| `xlns16_ideal` | xlns16 | Off | Use `<math.h>` `log`/`pow` for Gaussian log (`sb`/`db`). Most accurate but slowest. |
| `xlns32_ideal` | xlns32 | Off | Same for 32-bit. |
| `xlns16_alt` | xlns16 | Off | Alternative addition algorithm with less branching. Better on modern CPUs. |
| `xlns32_alt` | xlns32 | Off | Same for 32-bit. |
| `xlns16_altopt` | xlns16 | Off | Even more streamlined (but slightly less accurate) LPVIP within `xlns16_alt`. |
| `xlns16_table` | xlns16 | Off | Use precomputed lookup tables for conversion and (with `xlns16_alt`) for `sb`/`db`. Fastest option. Costs < 1 MB RAM. |
| `xlns32_arch16` | xlns32 | Off | Use `unsigned long` / `signed long` for 16-bit architectures (Turbo C/C++). |

### Recommended Configurations

| Use Case | Macros | Speed | Accuracy | Memory |
|----------|--------|-------|----------|--------|
| Maximum accuracy | `xlns16_ideal` or `xlns32_ideal` | Slowest | Best | Minimal |
| Balanced (no tables) | `xlns16_alt` | Medium | Good (LPVIP) | Minimal |
| Maximum speed (16-bit) | `xlns16_alt` + `xlns16_table` | **Fastest** | Good (table-based) | ~1 MB |
| Speed with slight accuracy loss | `xlns16_alt` + `xlns16_altopt` | Fast | Slightly less | Minimal |

---

## Performance Characteristics

Based on benchmarks from `time16test.cpp` (1 billion operations on a typical x86-64):

| Operation | Time (ns/op) | Notes |
|-----------|-------------|-------|
| FP addition | ~3 | Baseline hardware FP |
| LNS multiplication | ~1–2 | Integer add — very cheap |
| LNS addition (all tables) | ~14 | `xlns16_alt` + `xlns16_table` |
| LNS addition (LPVIP, no tables) | ~19 | `xlns16_alt` without `xlns16_table` |
| LNS addition (ideal) | ~45 | `xlns16_ideal`, uses `log`/`pow` |
| FP→LNS conversion (table) | ~3 | `xlns16_table` |
| FP→LNS conversion (computed) | ~14 | Without tables, uses `log` |
| LNS→FP conversion (table) | ~4 | `xlns16_table` |
| LNS→FP conversion (computed) | ~25 | Without tables, uses `pow` |
| Full pipeline: FP→LNS→sum→FP (tables) | ~23 | 6 + 14 + 3 ≈ **~8× slower** than FP |
| Full pipeline: FP→LNS→sum→FP (no tables) | ~86 | 27 + 45 + 14 ≈ **~30× slower** than FP |

**Key takeaway:** With lookup tables, the overhead is manageable for proof-of-concept integrations like LLM inference engines.

---

## Conversion Cache

Both `xlns16_float` and `xlns32_float` include a **direct-mapped cache** for float→LNS conversion to avoid redundant `log` computations:

| Property | `xlns16_float` | `xlns32_float` |
|----------|---------------|----------------|
| Cache size | 1024 entries | 1024 entries |
| Default state | **Off** (`xlns16_cacheon 0`) | **On** (`xlns32_cacheon 1`) |
| Reason | Table lookup is faster | Computed conversion is expensive |

The cache is indexed by bytes of the float bit pattern and stores the corresponding LNS value.

---

## Sister Projects

- **[xlns (Python)](https://github.com/xlnsresearch/xlns)** — Python LNS library sharing the same mathematical foundation.
- **[xlnscuda](https://github.com/xlnsresearch/xlnscuda)** — CUDA GPU implementations of LNS routines.

---

## References

1. M. G. Arnold, et al. "Arithmetic cotransformations in the Real and Complex Logarithmic Number Systems," *IEEE Trans. Comput.*, vol. 47, no. 7, pp. 777–786, July 1998.

2. M. G. Arnold, "LPVIP: A Low-power ROM-Less ALU for Low-Precision LNS," *14th International Workshop on Power and Timing Modeling, Optimization and Simulation*, LNCS 3254, pp. 675–684, Santorini, Greece, Sept. 2004.

---

## License

MIT License — Copyright (c) 1999–2025 Mark G. Arnold
