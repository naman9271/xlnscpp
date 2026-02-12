# Project Plan: Support for Logarithmic Number Systems in Large Language Models

## Integrating xlnscpp into ggml / llama.cpp

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Map — Which Repo Does What](#2-repository-map--which-repo-does-what)
3. [Architecture Overview](#3-architecture-overview)
4. [Prerequisites & Environment Setup](#4-prerequisites--environment-setup)
5. [Phase 1 — Understand the Codebase](#5-phase-1--understand-the-codebase)
6. [Phase 2 — Build xlnscpp as a Standalone Library](#6-phase-2--build-xlnscpp-as-a-standalone-library)
7. [Phase 3 — Create the ggml-lns Backend](#7-phase-3--create-the-ggml-lns-backend)
8. [Phase 4 — Implement Tensor Operations in LNS](#8-phase-4--implement-tensor-operations-in-lns)
9. [Phase 5 — Register the Backend & Build Integration](#9-phase-5--register-the-backend--build-integration)
10. [Phase 6 — End-to-End Testing with llama.cpp](#10-phase-6--end-to-end-testing-with-llamacpp)
11. [Phase 7 — Performance Optimization](#11-phase-7--performance-optimization)
12. [Phase 8 — Validation with a Real LLM (DeepSeek)](#12-phase-8--validation-with-a-real-llm-deepseek)
13. [File-by-File Change Map](#13-file-by-file-change-map)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Risk Register](#15-risk-register)
16. [Milestones & Timeline](#16-milestones--timeline)

---

## 1. Project Overview

### Goal
Create a "virtual" LNS (Logarithmic Number System) backend for **ggml** (the tensor library inside llama.cpp) that uses **xlnscpp** to perform all arithmetic in logarithmic representation instead of floating-point. This backend should appear to llama.cpp as just another hardware platform (like a GPU), so that an LLM (like DeepSeek) can run inference using LNS arithmetic and produce valid output.

### What This Is NOT
- We are **not** building real hardware — this is a software emulation.
- We are **not** expecting competitive speed — the goal is a **proof of concept** that LNS produces valid LLM output.
- We are **not** changing the weight storage format — weights stay in their existing quantized formats (Q4_0, Q8_0, etc.) and are **converted to LNS at computation time**.

### Core Idea
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  llama.cpp   │────>│   ggml       │────>│  ggml-lns       │
│  (unchanged) │     │  scheduler   │     │  backend        │
│              │     │  routes ops  │     │  (NEW - our     │
│              │     │  to backend  │     │   code)         │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │    xlnscpp       │
                                          │  (xlns16/xlns32) │
                                          │  LNS arithmetic  │
                                          └─────────────────┘
```

---

## 2. Repository Map — Which Repo Does What

There are **three** repositories involved. Here's exactly where your work goes:

### Repository 1: `xlnscpp` (this repo)
**URL:** `https://github.com/xlnsresearch/xlnscpp`  
**Role:** Provides the LNS arithmetic primitives  
**What you do here:**
- Potentially create a proper header (`xlns.h`) that cleanly exposes the API
- Build a wrapper/adapter layer that ggml can call
- Add any missing operations needed for LLM inference
- **This is where the LNS math lives**

### Repository 2: `llama.cpp` (fork it)
**URL:** `https://github.com/ggml-org/llama.cpp`  
**Role:** The LLM inference engine. ggml is a **subdirectory** inside this repo (`llama.cpp/ggml/`)  
**What you do here:**
- Add the new `ggml-lns` backend inside `llama.cpp/ggml/src/ggml-lns/`
- Register the backend in the build system and backend registry
- This is where 90% of the integration code lives
- **You do NOT need to modify llama.cpp's own code** — only the ggml subdirectory

### Repository 3: `ggml` (standalone — DO NOT modify directly)
**URL:** `https://github.com/ggml-org/ggml`  
**Role:** Standalone mirror of the ggml library  
**What you do here:** **NOTHING** directly. The ggml repo is synced from llama.cpp. All your changes go into the `llama.cpp/ggml/` subdirectory. They will eventually be synced to the standalone ggml repo by the maintainers.

### Summary: Where to Contribute

| Task | Repository | Directory |
|------|-----------|-----------|
| LNS arithmetic primitives | `xlnscpp` | Root |
| ggml backend implementation | `llama.cpp` (fork) | `ggml/src/ggml-lns/` |
| Backend header | `llama.cpp` (fork) | `ggml/include/ggml-lns.h` |
| CMake integration | `llama.cpp` (fork) | `ggml/CMakeLists.txt` and `ggml/src/ggml-lns/CMakeLists.txt` |
| Backend registration | `llama.cpp` (fork) | `ggml/src/ggml-backend-reg.cpp` |
| Testing | Both | `xlnscpp/tests/` and `llama.cpp/tests/` |

---

## 3. Architecture Overview

### How ggml Backends Work

ggml uses a **pluggable backend architecture**. Every backend (CPU, CUDA, Metal, Vulkan, etc.) implements the same set of C interfaces:

```
ggml_backend_reg_i        — Registry: "I exist, here are my devices"
  ├── get_name()
  ├── get_device_count()
  ├── get_device()
  └── get_proc_address()

ggml_backend_device_i     — Device: "Here's what I can do"
  ├── get_name()
  ├── get_description()
  ├── get_type()            → GGML_BACKEND_DEVICE_TYPE_ACCEL
  ├── get_props()
  ├── init_backend()
  ├── get_buffer_type()
  ├── supports_op()         → "Can I handle MUL_MAT? ADD? SOFTMAX?"
  └── supports_buft()

ggml_backend_i            — Backend instance: "I compute graphs"
  ├── get_name()
  ├── free()
  ├── graph_compute()       → THE MAIN ENTRY POINT
  ├── set_tensor_async()
  └── get_tensor_async()

ggml_backend_buffer_type_i — Buffer type: "How I allocate memory"
  ├── alloc_buffer()
  ├── get_alignment()
  └── get_max_size()

ggml_backend_buffer_i     — Buffer: "I hold tensor data"
  ├── get_base()
  ├── init_tensor()
  ├── set_tensor()
  ├── get_tensor()
  └── clear()
```

### The LNS Backend Strategy

Our LNS backend will be an **accelerator-type** backend (`GGML_BACKEND_DEVICE_TYPE_ACCEL`), similar to BLAS or ZenDNN. This means:

1. **It uses host (CPU) memory** — no separate memory space
2. **It selectively handles operations** — `supports_op()` returns true only for ops we implement in LNS
3. **The ggml scheduler automatically falls back to CPU** for any ops we don't support
4. **Tensors are stored as float in memory** — conversion to/from LNS happens at computation time

This is the simplest viable approach and matches how BLAS/ZenDNN work.

### Data Flow for a Single Operation (e.g., MUL_MAT)

```
1. llama.cpp builds a compute graph with ggml_mul_mat() nodes
2. ggml scheduler asks our backend: supports_op(MUL_MAT)? → YES
3. ggml scheduler routes the MUL_MAT node to our backend
4. Our graph_compute() is called:
   a. Read input tensors (float data from CPU buffers)
   b. Convert float → xlns32 using fp2xlns32()
   c. Perform matrix multiply using xlns32_mul() and xlns32_add()
   d. Convert result xlns32 → float using xlns322fp()
   e. Write result back to output tensor (float)
5. ggml scheduler continues with next operation
```

### Optimized Data Flow (Minimize Conversions)

For better performance, we can keep data in LNS format between operations:

```
1. First time a tensor enters our backend:
   - Convert float → LNS, cache the LNS representation in tensor->extra
2. Subsequent operations on the same tensor:
   - Reuse cached LNS data (no re-conversion)
3. When the tensor leaves our backend (needed by CPU):
   - Convert LNS → float, write back to tensor buffer
```

---

## 4. Prerequisites & Environment Setup

### Step 4.1: Install Build Tools

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-pip

# Verify
g++ --version    # Need C++17 support (GCC 7+ or Clang 5+)
cmake --version  # Need 3.14+
```

### Step 4.2: Clone Repositories

```bash
# Create a workspace
mkdir ~/lns-llm-project && cd ~/lns-llm-project

# Fork llama.cpp on GitHub first, then clone YOUR fork
git clone https://github.com/<YOUR_USERNAME>/llama.cpp.git
cd llama.cpp

# Create a feature branch
git checkout -b feature/ggml-lns-backend

# Go back to workspace
cd ..

# Clone xlnscpp (you already have this)
# git clone https://github.com/xlnsresearch/xlnscpp.git
```

### Step 4.3: Verify llama.cpp Builds Without Modifications

```bash
cd ~/lns-llm-project/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Test with a small model
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 20
```

### Step 4.4: Verify xlnscpp Works

```bash
cd ~/lns-llm-project/xlnscpp   # (or /home/naman/Desktop/xlnscpp)
g++ -O2 -o xlns32test xlns32test.cpp
echo "" | ./xlns32test
# Should see test output with matching FP and LNS results
```

---

## 5. Phase 1 — Understand the Codebase

**Time estimate:** 1–2 weeks  
**Goal:** Understand how ggml backends work by studying existing ones

### Step 5.1: Study a Simple Backend (BLAS)

The BLAS backend is the closest model for what we're building. Study these files:

```
llama.cpp/ggml/src/ggml-blas/ggml-blas.cpp     (~520 lines)
llama.cpp/ggml/include/ggml-blas.h              (~20 lines)
llama.cpp/ggml/src/ggml-blas/CMakeLists.txt
```

Key things to understand:
- How `ggml_backend_blas_reg()` registers the backend
- How `ggml_backend_blas_graph_compute()` iterates over the graph and handles each op
- How `ggml_backend_blas_supports_op()` declares supported operations
- How the buffer type uses CPU memory (`ggml_backend_cpu_buffer_type()`)

### Step 5.2: Study ZenDNN Backend

Even simpler than BLAS:

```
llama.cpp/ggml/src/ggml-zendnn/ggml-zendnn.cpp  (~470 lines)
```

Key pattern to copy:
```cpp
static ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                // Handle this op
                break;
            default:
                GGML_ABORT("unsupported op");
        }
    }
    return GGML_STATUS_SUCCESS;
}
```

### Step 5.3: Study the Backend Implementation Interface

Read this file carefully:

```
llama.cpp/ggml/src/ggml-backend-impl.h
```

Key structs to understand:
- `ggml_backend_i` (lines ~87–120): The backend instance interface
- `ggml_backend_device_i` (lines ~140–180): The device interface
- `ggml_backend_reg_i` (lines ~194–210): The registry interface
- `ggml_backend_buffer_i` (lines ~40–65): The buffer interface
- `ggml_backend_buffer_type_i` (lines ~15–35): The buffer type interface

### Step 5.4: Study ggml Operations

Understand which operations LLMs actually use:

```
llama.cpp/docs/ops.md         — List of all ggml operations
llama.cpp/ggml/include/ggml.h — ggml_op enum
```

The critical operations for LLM inference are:
1. `GGML_OP_MUL_MAT` — Matrix multiplication (most compute-intensive)
2. `GGML_OP_ADD` — Tensor addition (residual connections)
3. `GGML_OP_MUL` — Element-wise multiplication (RMSNorm scaling)
4. `GGML_OP_SOFT_MAX` — Softmax (attention)
5. `GGML_OP_ROPE` — Rotary position embedding
6. `GGML_OP_NORM` / `GGML_OP_RMS_NORM` — Normalization
7. `GGML_OP_SILU` / `GGML_OP_GELU` — Activation functions

### Step 5.5: Study xlnscpp API

Read and understand:
- `xlns32.cpp` — The 32-bit implementation (lines 1–644)
- `xlns16.cpp` — The 16-bit implementation (lines 1–609)
- `docs/api-reference.md` — Function signatures
- `docs/architecture.md` — How LNS math works

Key functions you'll need:
```cpp
// Conversion
xlns32 fp2xlns32(float x);       // float → LNS
float xlns322fp(xlns32 x);       // LNS → float

// Arithmetic
xlns32 xlns32_mul(xlns32 a, xlns32 b);   // Multiplication (CHEAP in LNS)
xlns32 xlns32_div(xlns32 a, xlns32 b);   // Division (CHEAP in LNS)
xlns32 xlns32_add(xlns32 a, xlns32 b);   // Addition (EXPENSIVE in LNS)

// Utility
xlns32 xlns32_neg(xlns32 x);     // Negation
xlns32 xlns32_abs(xlns32 x);     // Absolute value
xlns32 xlns32_sqrt(xlns32 x);    // Square root (CHEAP: halve the log)
```

---

## 6. Phase 2 — Build xlnscpp as a Standalone Library

**Time estimate:** 1 week  
**Repo:** `xlnscpp`

### Step 6.1: Create a Clean Header for ggml Integration

Currently xlnscpp uses a header-include pattern (`#include "xlns32.cpp"`). For ggml integration, we need a proper header/source separation.

Create `xlnscpp/xlns_ggml_adapter.h`:

```cpp
// xlns_ggml_adapter.h — Clean C-linkage API for ggml integration
#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque type for LNS values
typedef uint32_t xlns32_t;
typedef uint16_t xlns16_t;

// === Scalar Conversion ===
xlns32_t xlns_from_float_32(float x);
float    xlns_to_float_32(xlns32_t x);
xlns16_t xlns_from_float_16(float x);
float    xlns_to_float_16(xlns16_t x);

// === Scalar Arithmetic (32-bit) ===
xlns32_t xlns32_multiply(xlns32_t a, xlns32_t b);
xlns32_t xlns32_divide(xlns32_t a, xlns32_t b);
xlns32_t xlns32_addition(xlns32_t a, xlns32_t b);
xlns32_t xlns32_negate(xlns32_t x);
xlns32_t xlns32_absolute(xlns32_t x);
xlns32_t xlns32_square_root(xlns32_t x);

// === Batch Operations (for ggml tensors) ===

// Convert an array of floats to LNS32 in-place (writes to dst)
void xlns_batch_float_to_lns32(const float * src, xlns32_t * dst, size_t n);
void xlns_batch_lns32_to_float(const xlns32_t * src, float * dst, size_t n);

// Vector dot product in LNS: sum(a[i] * b[i]) for i in [0, n)
float xlns_vec_dot_lns32(const float * a, const float * b, size_t n);

// Matrix multiply: C[m×n] = A[m×k] × B[k×n]  (all float, internal LNS)
void xlns_mat_mul_lns32(
    const float * A, const float * B, float * C,
    int m, int n, int k,
    size_t lda, size_t ldb, size_t ldc
);

// Element-wise operations on float arrays (internal LNS)
void xlns_vec_add(const float * a, const float * b, float * c, size_t n);
void xlns_vec_mul(const float * a, const float * b, float * c, size_t n);
void xlns_vec_scale(const float * a, float s, float * c, size_t n);

#ifdef __cplusplus
}
#endif
```

### Step 6.2: Implement the Adapter

Create `xlnscpp/xlns_ggml_adapter.cpp`:

```cpp
// xlns_ggml_adapter.cpp — Implementation of the adapter layer

// Include the xlnscpp implementation
#define xlns32_alt
#include "xlns32.cpp"

#include "xlns_ggml_adapter.h"

extern "C" {

// === Scalar Conversion ===
xlns32_t xlns_from_float_32(float x)        { return fp2xlns32(x); }
float    xlns_to_float_32(xlns32_t x)       { return xlns322fp(x); }

// === Scalar Arithmetic ===
xlns32_t xlns32_multiply(xlns32_t a, xlns32_t b)  { return xlns32_mul(a, b); }
xlns32_t xlns32_divide(xlns32_t a, xlns32_t b)    { return xlns32_div(a, b); }
xlns32_t xlns32_addition(xlns32_t a, xlns32_t b)  { return xlns32_add(a, b); }
xlns32_t xlns32_negate(xlns32_t x)                 { return xlns32_neg(x); }
xlns32_t xlns32_absolute(xlns32_t x)               { return xlns32_abs(x); }
xlns32_t xlns32_square_root(xlns32_t x)            { return xlns32_sqrt(x); }

// === Batch Conversion ===
void xlns_batch_float_to_lns32(const float * src, xlns32_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp2xlns32(src[i]);
    }
}

void xlns_batch_lns32_to_float(const xlns32_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = xlns322fp(src[i]);
    }
}

// === Vector Dot Product ===
float xlns_vec_dot_lns32(const float * a, const float * b, size_t n) {
    xlns32 sum = xlns32_zero;
    for (size_t i = 0; i < n; i++) {
        xlns32 la = fp2xlns32(a[i]);
        xlns32 lb = fp2xlns32(b[i]);
        xlns32 prod = xlns32_mul(la, lb);
        sum = xlns32_add(sum, prod);
    }
    return xlns322fp(sum);
}

// === Matrix Multiply ===
void xlns_mat_mul_lns32(
    const float * A, const float * B, float * C,
    int m, int n, int k,
    size_t lda, size_t ldb, size_t ldc
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            xlns32 sum = xlns32_zero;
            for (int p = 0; p < k; p++) {
                xlns32 la = fp2xlns32(A[i * lda + p]);
                xlns32 lb = fp2xlns32(B[p * ldb + j]);
                sum = xlns32_add(sum, xlns32_mul(la, lb));
            }
            C[i * ldc + j] = xlns322fp(sum);
        }
    }
}

// === Element-wise Operations ===
void xlns_vec_add(const float * a, const float * b, float * c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = xlns322fp(xlns32_add(fp2xlns32(a[i]), fp2xlns32(b[i])));
    }
}

void xlns_vec_mul(const float * a, const float * b, float * c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = xlns322fp(xlns32_mul(fp2xlns32(a[i]), fp2xlns32(b[i])));
    }
}

void xlns_vec_scale(const float * a, float s, float * c, size_t n) {
    xlns32 ls = fp2xlns32(s);
    for (size_t i = 0; i < n; i++) {
        c[i] = xlns322fp(xlns32_mul(fp2xlns32(a[i]), ls));
    }
}

} // extern "C"
```

### Step 6.3: Test the Adapter

Create `xlnscpp/test_adapter.cpp`:

```cpp
#include "xlns_ggml_adapter.h"
#include <cstdio>
#include <cmath>

int main() {
    // Test scalar operations
    float a = 3.14f, b = 2.71f;
    xlns32_t la = xlns_from_float_32(a);
    xlns32_t lb = xlns_from_float_32(b);

    printf("a=%.4f, b=%.4f\n", a, b);
    printf("a*b: expected=%.4f, got=%.4f\n", a*b, xlns_to_float_32(xlns32_multiply(la, lb)));
    printf("a+b: expected=%.4f, got=%.4f\n", a+b, xlns_to_float_32(xlns32_addition(la, lb)));
    printf("a/b: expected=%.4f, got=%.4f\n", a/b, xlns_to_float_32(xlns32_divide(la, lb)));

    // Test vector dot product
    float va[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vb[] = {4.0f, 3.0f, 2.0f, 1.0f};
    float expected_dot = 1*4 + 2*3 + 3*2 + 4*1;  // = 20
    float lns_dot = xlns_vec_dot_lns32(va, vb, 4);
    printf("dot product: expected=%.4f, got=%.4f\n", expected_dot, lns_dot);

    // Test matrix multiply
    float A[] = {1, 2, 3, 4, 5, 6};  // 2x3
    float B[] = {7, 8, 9, 10, 11, 12}; // 3x2
    float C[4] = {0};  // 2x2
    xlns_mat_mul_lns32(A, B, C, 2, 2, 3, 3, 2, 2);
    printf("matmul [0,0]: expected=%.1f, got=%.4f\n", 1*7+2*9+3*11.0f, C[0]);
    printf("matmul [0,1]: expected=%.1f, got=%.4f\n", 1*8+2*10+3*12.0f, C[1]);
    printf("matmul [1,0]: expected=%.1f, got=%.4f\n", 4*7+5*9+6*11.0f, C[2]);
    printf("matmul [1,1]: expected=%.1f, got=%.4f\n", 4*8+5*10+6*12.0f, C[3]);

    return 0;
}
```

Build and test:
```bash
cd ~/lns-llm-project/xlnscpp
g++ -O2 -o test_adapter test_adapter.cpp xlns_ggml_adapter.cpp
./test_adapter
```

---

## 7. Phase 3 — Create the ggml-lns Backend

**Time estimate:** 2–3 weeks  
**Repo:** `llama.cpp` (your fork)

### Step 7.1: Create the Directory Structure

```bash
cd ~/lns-llm-project/llama.cpp
mkdir -p ggml/src/ggml-lns
```

Copy xlnscpp files into the backend:
```bash
cp ~/lns-llm-project/xlnscpp/xlns32.cpp      ggml/src/ggml-lns/
cp ~/lns-llm-project/xlnscpp/xlns32tbl.h      ggml/src/ggml-lns/
cp ~/lns-llm-project/xlnscpp/xlns16.cpp        ggml/src/ggml-lns/
cp ~/lns-llm-project/xlnscpp/xlns16sbdbtbl.h   ggml/src/ggml-lns/
cp ~/lns-llm-project/xlnscpp/xlns16cvtbl.h     ggml/src/ggml-lns/
cp ~/lns-llm-project/xlnscpp/xlns16revcvtbl.h  ggml/src/ggml-lns/
```

Create the following new files:

```
ggml/
├── include/
│   └── ggml-lns.h              ← Public API header (NEW)
└── src/
    └── ggml-lns/
        ├── CMakeLists.txt       ← Build configuration (NEW)
        ├── ggml-lns.cpp         ← Backend implementation (NEW, ~800-1200 lines)
        ├── xlns32.cpp           ← Copy from xlnscpp
        ├── xlns32tbl.h          ← Copy from xlnscpp
        ├── xlns16.cpp           ← Copy from xlnscpp (optional for phase 1)
        └── xlns16*.h            ← Copy from xlnscpp (optional for phase 1)
```

### Step 7.2: Create the Public Header

Create `ggml/include/ggml-lns.h`:

```cpp
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_LNS_NAME "LNS"

// Backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_lns_init(void);

GGML_BACKEND_API bool ggml_backend_is_lns(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_lns_reg(void);

#ifdef __cplusplus
}
#endif
```

### Step 7.3: Implement the Backend

Create `ggml/src/ggml-lns/ggml-lns.cpp`. This is the **core of the project**. Here is the skeleton with detailed guidance:

```cpp
// ggml-lns.cpp — LNS (Logarithmic Number System) backend for ggml
// Uses xlnscpp for arithmetic

#include "ggml-lns.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <mutex>

// Include xlnscpp — it uses the header-include pattern
#define xlns32_alt
#include "xlns32.cpp"

// ============================================================
// Forward declarations
// ============================================================
static const char * ggml_backend_lns_get_name(ggml_backend_t backend);
static void ggml_backend_lns_free(ggml_backend_t backend);
static enum ggml_status ggml_backend_lns_graph_compute(
    ggml_backend_t backend, struct ggml_cgraph * cgraph);

// ============================================================
// Backend context (holds per-instance state)
// ============================================================
struct ggml_backend_lns_context {
    int n_threads;  // For potential future parallelization
};

// ============================================================
// LNS Tensor Operations
// ============================================================

// Helper: get float data pointer from a tensor
static const float * get_tensor_float_data(const struct ggml_tensor * t) {
    return (const float *)t->data;
}

static float * get_tensor_float_data_mut(struct ggml_tensor * t) {
    return (float *)t->data;
}

// --- GGML_OP_MUL_MAT ---
// C = A × B^T  (ggml convention: src0=weights, src1=input, dst=output)
// In ggml, MUL_MAT computes: dst[i,j] = dot(src0_row_j, src1_row_i)
static void ggml_lns_op_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // weights [ne00 × ne01]
    const struct ggml_tensor * src1 = dst->src[1];  // input   [ne10 × ne11]

    const int64_t ne00 = src0->ne[0];  // K (inner dimension)
    const int64_t ne01 = src0->ne[1];  // N (output features / rows of weights)
    const int64_t ne10 = src1->ne[0];  // K (should equal ne00)
    const int64_t ne11 = src1->ne[1];  // M (batch / sequence length)

    GGML_ASSERT(ne00 == ne10);

    const float * src0_data = get_tensor_float_data(src0);
    const float * src1_data = get_tensor_float_data(src1);
    float * dst_data = get_tensor_float_data_mut(dst);

    // For each batch element and output row, compute the dot product
    for (int64_t i1 = 0; i1 < ne11; i1++) {         // M dimension
        for (int64_t i0 = 0; i0 < ne01; i0++) {     // N dimension
            xlns32 sum = xlns32_zero;
            for (int64_t k = 0; k < ne00; k++) {    // K dimension
                float a_val = src0_data[i0 * ne00 + k];
                float b_val = src1_data[i1 * ne10 + k];
                xlns32 la = fp2xlns32(a_val);
                xlns32 lb = fp2xlns32(b_val);
                xlns32 prod = xlns32_mul(la, lb);
                sum = xlns32_add(sum, prod);
            }
            dst_data[i1 * ne01 + i0] = xlns322fp(sum);
        }
    }
}

// --- GGML_OP_ADD ---
static void ggml_lns_op_add(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t n = ggml_nelements(dst);

    const float * a = get_tensor_float_data(src0);
    const float * b = get_tensor_float_data(src1);
    float * c = get_tensor_float_data_mut(dst);

    // Handle broadcasting (src1 may be smaller)
    const int64_t n0 = ggml_nelements(src0);
    const int64_t n1 = ggml_nelements(src1);

    for (int64_t i = 0; i < n; i++) {
        xlns32 la = fp2xlns32(a[i % n0]);
        xlns32 lb = fp2xlns32(b[i % n1]);
        c[i] = xlns322fp(xlns32_add(la, lb));
    }
}

// --- GGML_OP_MUL (element-wise) ---
static void ggml_lns_op_mul(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t n = ggml_nelements(dst);
    const int64_t n0 = ggml_nelements(src0);
    const int64_t n1 = ggml_nelements(src1);

    const float * a = get_tensor_float_data(src0);
    const float * b = get_tensor_float_data(src1);
    float * c = get_tensor_float_data_mut(dst);

    for (int64_t i = 0; i < n; i++) {
        xlns32 la = fp2xlns32(a[i % n0]);
        xlns32 lb = fp2xlns32(b[i % n1]);
        c[i] = xlns322fp(xlns32_mul(la, lb));
    }
}

// Additional ops to implement:
// GGML_OP_SOFT_MAX, GGML_OP_RMS_NORM, GGML_OP_SILU, etc.
// (See Phase 4 for details)

// ============================================================
// graph_compute — THE MAIN DISPATCH FUNCTION
// ============================================================
static enum ggml_status ggml_backend_lns_graph_compute(
    ggml_backend_t backend, struct ggml_cgraph * cgraph)
{
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_lns_op_mul_mat(node);
                break;
            case GGML_OP_ADD:
                ggml_lns_op_add(node);
                break;
            case GGML_OP_MUL:
                ggml_lns_op_mul(node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These are no-ops (metadata only)
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

// ============================================================
// Backend Interface
// ============================================================
static const char * ggml_backend_lns_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_LNS_NAME;
}

static void ggml_backend_lns_free(ggml_backend_t backend) {
    ggml_backend_lns_context * ctx =
        (ggml_backend_lns_context *)backend->context;
    delete ctx;
    delete backend;
}

static struct ggml_backend_i ggml_backend_lns_i = {
    /* .get_name                = */ ggml_backend_lns_get_name,
    /* .free                    = */ ggml_backend_lns_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_lns_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ============================================================
// GUID
// ============================================================
static ggml_guid_t ggml_backend_lns_guid(void) {
    static ggml_guid guid = {
        0x4c, 0x4e, 0x53, 0x42, 0x41, 0x43, 0x4b, 0x45,
        0x4e, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    };  // "LNSBACKEND\0\0\0\0\0\1"
    return &guid;
}

// ============================================================
// Backend Init
// ============================================================
ggml_backend_t ggml_backend_lns_init(void) {
    ggml_backend_lns_context * ctx = new ggml_backend_lns_context;
    ctx->n_threads = 1;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_lns_guid(),
        /* .interface = */ ggml_backend_lns_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_lns_reg(), 0),
        /* .context   = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_lns(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(
        ggml_backend_guid(backend), ggml_backend_lns_guid());
}

// ============================================================
// Device Interface
// ============================================================
static const char * ggml_backend_lns_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "LNS";
}

static const char * ggml_backend_lns_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Logarithmic Number System (xlnscpp)";
}

static void ggml_backend_lns_device_get_memory(
    ggml_backend_dev_t dev, size_t * free, size_t * total)
{
    // We use CPU memory
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_lns_device_get_type(
    ggml_backend_dev_t dev)
{
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_lns_device_get_props(
    ggml_backend_dev_t dev, struct ggml_backend_dev_props * props)
{
    props->name        = ggml_backend_lns_device_get_name(dev);
    props->description = ggml_backend_lns_device_get_description(dev);
    props->type        = ggml_backend_lns_device_get_type(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->caps = {
        /* .async            = */ false,
        /* .host_buffer      = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events           = */ false,
    };
}

static ggml_backend_t ggml_backend_lns_device_init(
    ggml_backend_dev_t dev, const char * params)
{
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_lns_init();
}

static ggml_backend_buffer_type_t ggml_backend_lns_device_get_buffer_type(
    ggml_backend_dev_t dev)
{
    GGML_UNUSED(dev);
    // We use CPU buffers — data stays in host memory
    return ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_lns_device_supports_op(
    ggml_backend_dev_t dev, const struct ggml_tensor * op)
{
    GGML_UNUSED(dev);

    // Only claim support for ops we've implemented
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // Only support float32 inputs for now
            return op->src[0]->type == GGML_TYPE_F32
                && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return op->src[0]->type == GGML_TYPE_F32;
        default:
            return false;
    }
}

static bool ggml_backend_lns_device_supports_buft(
    ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
{
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_lns_device_i = {
    /* .get_name             = */ ggml_backend_lns_device_get_name,
    /* .get_description      = */ ggml_backend_lns_device_get_description,
    /* .get_memory           = */ ggml_backend_lns_device_get_memory,
    /* .get_type             = */ ggml_backend_lns_device_get_type,
    /* .get_props            = */ ggml_backend_lns_device_get_props,
    /* .init_backend         = */ ggml_backend_lns_device_init,
    /* .get_buffer_type      = */ ggml_backend_lns_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_lns_device_supports_op,
    /* .supports_buft        = */ ggml_backend_lns_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ============================================================
// Registry Interface
// ============================================================
static const char * ggml_backend_lns_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_LNS_NAME;
}

static size_t ggml_backend_lns_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;  // We have exactly one "LNS device"
}

static ggml_backend_dev_t ggml_backend_lns_reg_get_device(
    ggml_backend_reg_t reg, size_t index)
{
    GGML_ASSERT(index == 0);
    GGML_UNUSED(reg);

    static ggml_backend_device ggml_backend_lns_device = {
        /* .iface   = */ ggml_backend_lns_device_i,
        /* .reg     = */ ggml_backend_lns_reg(),
        /* .context = */ NULL,
    };

    return &ggml_backend_lns_device;
}

static void * ggml_backend_lns_get_proc_address(
    ggml_backend_reg_t reg, const char * name)
{
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_lns_reg_i = {
    /* .get_name         = */ ggml_backend_lns_reg_get_name,
    /* .get_device_count = */ ggml_backend_lns_reg_get_device_count,
    /* .get_device       = */ ggml_backend_lns_reg_get_device,
    /* .get_proc_address = */ ggml_backend_lns_get_proc_address,
};

ggml_backend_reg_t ggml_backend_lns_reg(void) {
    static struct ggml_backend_reg ggml_backend_lns_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_lns_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_lns_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_lns_reg)
```

---

## 8. Phase 4 — Implement Tensor Operations in LNS

**Time estimate:** 3–4 weeks  
**Repo:** `llama.cpp` (in `ggml/src/ggml-lns/ggml-lns.cpp`)

### Operations Needed for a Minimal LLM (Priority Order)

#### Tier 1 — Essential (blocks inference without these)

| Op | Description | LNS Strategy |
|----|-------------|--------------|
| `GGML_OP_MUL_MAT` | Matrix multiply | Convert to LNS, multiply (cheap), accumulate (expensive) |
| `GGML_OP_ADD` | Tensor addition | Convert to LNS, add |
| `GGML_OP_MUL` | Element-wise multiply | Convert to LNS, multiply (cheap) |
| `GGML_OP_RMS_NORM` | RMS normalization | Convert to LNS, compute norm |
| `GGML_OP_SOFT_MAX` | Softmax | Find max (comparison), subtract, exp, sum, divide |
| `GGML_OP_ROPE` | Rotary position embedding | Trigonometric ops, then multiply and add |
| `GGML_OP_SILU` | SiLU activation | `x * sigmoid(x)` |
| `GGML_OP_CPY` | Tensor copy | Direct memcpy (no LNS needed) |
| `GGML_OP_CONT` | Make contiguous | Layout operation (no LNS needed) |
| `GGML_OP_SCALE` | Multiply by scalar | Convert scalar to LNS, multiply (cheap) |

#### Tier 2 — Needed for Full Models

| Op | Description | LNS Strategy |
|----|-------------|--------------|
| `GGML_OP_NORM` | Layer normalization | Similar to RMS_NORM |
| `GGML_OP_GELU` | GELU activation | Approximation or convert-compute-convert |
| `GGML_OP_DIAG_MASK_INF` | Causal attention mask | Set values to -inf (special LNS value) |
| `GGML_OP_GET_ROWS` | Embedding lookup | No arithmetic, just data movement |

#### Tier 3 — Can Fall Back to CPU

| Op | Description | Why CPU is OK |
|----|-------------|---------------|
| `GGML_OP_RESHAPE` / `VIEW` / `PERMUTE` / `TRANSPOSE` | Layout ops | No computation, metadata only |

### Implementation Guide for Key Operations

#### GGML_OP_RMS_NORM

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

In LNS:
1. x² → multiply each element by itself (cheap: double the log)
2. mean → sum all (expensive) then divide by n (cheap)
3. sqrt → halve the log (very cheap)
4. x / rms → subtract logs (cheap)
5. * weight → add logs (cheap)
```

#### GGML_OP_SOFT_MAX

```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

In LNS:
1. Find max(x) → comparison (done on float or LNS)
2. x - max(x) → subtraction in LNS
3. exp() → convert to float, compute exp, convert back
   OR compute in LNS: exp(x) = 2^(x / ln2)
4. sum → expensive LNS addition
5. divide → cheap log subtraction
```

#### GGML_OP_SILU

```
SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))

In LNS:
- exp(-x): compute in LNS
- 1 + exp(-x): LNS addition (expensive)
- 1 / (1 + exp(-x)): LNS division (cheap)
- x * sigmoid: LNS multiplication (cheap)
```

### Strategy: Hybrid Approach

For operations that require transcendental functions (exp, sin, cos for ROPE), use a **hybrid approach**:

```cpp
// Example: exp() via convert-compute-convert
static void ggml_lns_op_exp(struct ggml_tensor * dst) {
    const float * src = get_tensor_float_data(dst->src[0]);
    float * out = get_tensor_float_data_mut(dst);
    const int64_t n = ggml_nelements(dst);

    for (int64_t i = 0; i < n; i++) {
        // For exp: convert to float, compute, convert back
        // This is acceptable because exp() is not the bottleneck
        out[i] = expf(src[i]);
    }
}
```

This is fine because:
1. Transcendental functions are called once per attention head, not per matrix element
2. The main benefit of LNS is in matrix multiply (which dominates compute)
3. Even hardware LNS implementations would use lookup tables for exp/sin/cos

---

## 9. Phase 5 — Register the Backend & Build Integration

**Time estimate:** 1 week  
**Repo:** `llama.cpp` (your fork)

### Step 9.1: Create CMakeLists.txt for the Backend

Create `ggml/src/ggml-lns/CMakeLists.txt`:

```cmake
message(STATUS "Using LNS backend")

ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
)

# The xlns32.cpp is included directly in ggml-lns.cpp, so no need to add it here
# But we need the include path for the table headers
target_include_directories(ggml-lns PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
```

### Step 9.2: Add to Main CMakeLists.txt

Edit `ggml/CMakeLists.txt` — add the LNS option alongside other backends:

```cmake
# Add near the other backend options (search for "GGML_BLAS" or similar)
option(GGML_LNS "ggml: use LNS backend (Logarithmic Number System)" OFF)
```

And in the backend subdirectory section:

```cmake
# Add near "add_subdirectory(src/ggml-blas)" or similar
if (GGML_LNS)
    add_subdirectory(src/ggml-lns)
endif()
```

### Step 9.3: Register in Backend Registry

Edit `ggml/src/ggml-backend-reg.cpp`. Add the LNS backend registration:

1. Add the include (near the top, with other backend includes):
```cpp
#ifdef GGML_USE_LNS
#include "ggml-lns.h"
#endif
```

2. Add registration in the `ggml_backend_registry()` constructor (near BLAS):
```cpp
#ifdef GGML_USE_LNS
    register_backend(ggml_backend_lns_reg());
#endif
```

### Step 9.4: Build and Test

```bash
cd ~/lns-llm-project/llama.cpp
cmake -B build -DGGML_LNS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify the backend is loaded
./build/bin/llama-cli --list-devices
# Should show "LNS" in the device list
```

---

## 10. Phase 6 — End-to-End Testing with llama.cpp

**Time estimate:** 2–3 weeks  
**Repo:** `llama.cpp` (your fork)

### Step 10.1: Test with Backend Operations Test

```bash
# Run the ggml backend operation tests against LNS
./build/bin/test-backend-ops -b LNS
```

This will test each supported operation individually and report pass/fail.

### Step 10.2: Test with a Tiny Model

Download a very small model (e.g., TinyLlama 1.1B or a Q4_0 quantized version):

```bash
# Download a small model
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 10

# Force use of LNS backend (mechanism depends on implementation)
# You may need to add a command-line flag or environment variable
GGML_LNS=1 ./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 10
```

### Step 10.3: Numerical Validation

Write a test that compares LNS output with CPU output:

```bash
# Run same prompt with CPU and record output
./build/bin/llama-cli -m model.gguf -p "Hello world" -n 50 --seed 42 > cpu_output.txt

# Run same prompt with LNS backend
GGML_LNS=1 ./build/bin/llama-cli -m model.gguf -p "Hello world" -n 50 --seed 42 > lns_output.txt

# Compare
diff cpu_output.txt lns_output.txt
```

The outputs may differ slightly (LNS introduces quantization noise), but should produce coherent text.

### Step 10.4: Perplexity Test

```bash
# Measure perplexity with CPU (baseline)
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw

# Measure perplexity with LNS
GGML_LNS=1 ./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw
```

**Expected:** LNS perplexity should be slightly worse than CPU but still reasonable (within ~5-20% for 32-bit LNS).

---

## 11. Phase 7 — Performance Optimization

**Time estimate:** 2–3 weeks  
**Repo:** Both `xlnscpp` and `llama.cpp`

### Optimization 1: LNS Caching

Instead of converting float↔LNS for every operation, cache the LNS representation:

```cpp
// Use tensor->extra to cache LNS data
struct lns_tensor_extra {
    xlns32 * lns_data;  // Cached LNS representation
    size_t n_elements;
    bool is_valid;      // Whether the cache matches the float data
};
```

### Optimization 2: Pre-convert Weights

Weights are constant during inference. Convert them to LNS once at load time:

```cpp
// In buffer init_tensor callback:
if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
    // Convert weight tensor to LNS and cache
    convert_and_cache_lns(tensor);
}
```

### Optimization 3: Use 16-bit LNS Where Appropriate

For attention scores and intermediate activations, 16-bit LNS may be sufficient:

```cpp
// Use xlns16 for attention computation (less precision needed)
#define xlns16_alt
#define xlns16_table
#include "xlns16.cpp"
```

### Optimization 4: Batch Conversions with SIMD

For the conversion bottleneck, consider vectorizing:

```cpp
// Use compiler auto-vectorization hints
void xlns_batch_convert(const float * src, xlns32 * dst, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp2xlns32(src[i]);
    }
}
```

### Optimization 5: Avoid Unnecessary Conversions

For multiplication-heavy operations (which is LNS's strength), keep data in LNS as long as possible:

```
MUL_MAT → result in LNS
  → ADD (residual) → still in LNS
    → RMS_NORM → still in LNS
      → SILU → needs float temporarily → back to LNS
        → MUL_MAT → result in LNS
```

---

## 12. Phase 8 — Validation with a Real LLM (DeepSeek)

**Time estimate:** 1–2 weeks  
**Repo:** `llama.cpp` (your fork)

### Step 12.1: Get DeepSeek Model

```bash
# Download DeepSeek model in GGUF format
# Use a small quantization to fit in memory
./build/bin/llama-cli -hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF -p "What is 2+2?" -n 50
```

### Step 12.2: Run Full Inference

```bash
# CPU baseline
./build/bin/llama-cli -m deepseek-1.5b-q4_0.gguf \
    -p "Explain quantum computing in simple terms" \
    -n 200 --seed 42 > deepseek_cpu.txt

# LNS backend
GGML_LNS=1 ./build/bin/llama-cli -m deepseek-1.5b-q4_0.gguf \
    -p "Explain quantum computing in simple terms" \
    -n 200 --seed 42 > deepseek_lns.txt
```

### Step 12.3: Measure Quality

```bash
# Perplexity comparison
./build/bin/llama-perplexity -m deepseek-1.5b-q4_0.gguf -f test_data.txt
GGML_LNS=1 ./build/bin/llama-perplexity -m deepseek-1.5b-q4_0.gguf -f test_data.txt
```

### Step 12.4: Measure Performance

```bash
./build/bin/llama-bench -m deepseek-1.5b-q4_0.gguf
GGML_LNS=1 ./build/bin/llama-bench -m deepseek-1.5b-q4_0.gguf
```

### Expected Results

| Metric | CPU (baseline) | LNS Backend | Notes |
|--------|---------------|-------------|-------|
| Perplexity | ~X | ~X + 5-20% | LNS quantization adds noise |
| Speed (tokens/sec) | ~Y | ~Y / 5-15x | Software emulation is slow |
| Output quality | Coherent text | Coherent text | Main goal: valid output |

---

## 13. File-by-File Change Map

### New Files (created by you)

| File | Repo | Purpose |
|------|------|---------|
| `ggml/include/ggml-lns.h` | llama.cpp | Public API header |
| `ggml/src/ggml-lns/ggml-lns.cpp` | llama.cpp | Backend implementation (~800-1500 lines) |
| `ggml/src/ggml-lns/CMakeLists.txt` | llama.cpp | Build configuration |
| `ggml/src/ggml-lns/xlns32.cpp` | llama.cpp | Copy of xlnscpp 32-bit implementation |
| `ggml/src/ggml-lns/xlns32tbl.h` | llama.cpp | Copy of xlnscpp lookup tables |
| `xlns_ggml_adapter.h` | xlnscpp | Clean C API (optional) |
| `xlns_ggml_adapter.cpp` | xlnscpp | Adapter implementation (optional) |
| `test_adapter.cpp` | xlnscpp | Adapter tests |

### Modified Files

| File | Repo | Change |
|------|------|--------|
| `ggml/CMakeLists.txt` | llama.cpp | Add `GGML_LNS` option and `add_subdirectory` |
| `ggml/src/ggml-backend-reg.cpp` | llama.cpp | Register LNS backend |

### NOT Modified

| File | Why |
|------|-----|
| `src/*.cpp` (llama.cpp source) | Backend is transparent to llama.cpp |
| `ggml/src/ggml.c` | Core ggml doesn't change |
| `ggml/include/ggml.h` | No new ops needed |
| `ggml/src/ggml-cpu/` | CPU backend stays as-is |

---

## 14. Key Design Decisions

### Decision 1: 32-bit vs 16-bit LNS

**Recommendation:** Start with **32-bit LNS** (`xlns32`)

| Factor | xlns32 | xlns16 |
|--------|--------|--------|
| Precision | 23 fractional bits (like float32) | 7 fractional bits (like bfloat16) |
| Accuracy | Very close to float32 | Noticeable quantization errors |
| Speed | Slower (cotransformation tables) | Faster (direct table lookup) |
| Proof of concept | Easier to validate | Harder to validate |

Start with xlns32 to prove correctness, then optionally add xlns16 as a faster mode.

### Decision 2: Which Operations to Implement in LNS

**Recommendation:** Start with `MUL_MAT` only, let everything else fall back to CPU.

Rationale:
- `MUL_MAT` is 90%+ of compute in LLM inference
- The ggml scheduler automatically routes unsupported ops to CPU
- You can incrementally add more ops later

### Decision 3: Accelerator vs GPU Backend Type

**Recommendation:** Use `GGML_BACKEND_DEVICE_TYPE_ACCEL`

This means:
- Uses host (CPU) memory — no memory management complexity
- Automatically participates in the scheduler alongside CPU
- No need to implement buffer allocation or memory transfer

### Decision 4: Static vs Dynamic Linking

**Recommendation:** Start with **static** (compile-time) integration (`#ifdef GGML_USE_LNS`)

The `GGML_BACKEND_DL_IMPL` macro at the end of the backend enables dynamic loading too, so you get both options.

---

## 15. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LNS addition noise accumulates, causing garbage output | High | Use 32-bit LNS; validate incrementally; compare with CPU at each layer |
| ggml backend interface changes frequently | Medium | Pin to a specific llama.cpp release; check API version |
| Some ops are too complex to implement in LNS | Low | Fall back to CPU for those ops (hybrid approach) |
| Performance too slow for meaningful testing | Medium | Optimize hot path (MUL_MAT); use caching; test with small models |
| Weight dequantization adds extra conversion overhead | Medium | Dequantize to float first (standard path), then convert float→LNS |

---

## 16. Milestones & Timeline

| Week | Milestone | Deliverable |
|------|-----------|------------|
| 1-2 | **M1: Understanding** | Study complete; can explain ggml backend architecture |
| 3 | **M2: xlnscpp adapter** | `xlns_ggml_adapter` working with tests passing |
| 4-5 | **M3: Backend skeleton** | ggml-lns compiles, registers, shows in device list |
| 6-7 | **M4: MUL_MAT works** | Matrix multiply produces correct results in LNS |
| 8-9 | **M5: Minimal ops** | ADD, MUL, RMS_NORM, SOFTMAX, SILU working |
| 10-11 | **M6: Small model runs** | Tiny model produces coherent output with LNS |
| 12-13 | **M7: Optimization** | Caching, pre-conversion of weights |
| 14-15 | **M8: DeepSeek validation** | DeepSeek model produces valid output |
| 16 | **M9: Documentation** | Final report, benchmarks, code cleanup |

---

## Appendix A: Quick Reference — ggml Backend Interface Structs

```cpp
// ggml-backend-impl.h — The interfaces you must implement

// 1. Buffer Type Interface (how memory is allocated)
struct ggml_backend_buffer_type_i {
    const char *  (*get_name)      (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    size_t        (*get_alignment) (ggml_backend_buffer_type_t buft);
    size_t        (*get_max_size)  (ggml_backend_buffer_type_t buft);
    size_t        (*get_alloc_size)(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
    bool          (*is_host)       (ggml_backend_buffer_type_t buft);
};

// 2. Buffer Interface (actual memory management)
struct ggml_backend_buffer_i {
    const char * (*get_name)   (ggml_backend_buffer_t buffer);
    void         (*free_buffer)(ggml_backend_buffer_t buffer);
    void *       (*get_base)   (ggml_backend_buffer_t buffer);
    void         (*init_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
    void         (*set_tensor) (ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void         (*get_tensor) (ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool         (*cpy_tensor) (ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst);
    void         (*clear)      (ggml_backend_buffer_t buffer, uint8_t value);
};

// 3. Backend Interface (computation)
struct ggml_backend_i {
    const char *   (*get_name)          (ggml_backend_t backend);
    void           (*free)              (ggml_backend_t backend);
    void           (*set_tensor_async)  (ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void           (*get_tensor_async)  (ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool           (*cpy_tensor_async)  (ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst);
    void           (*synchronize)       (ggml_backend_t backend);
    ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const ggml_cgraph * cgraph);
    void           (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    void           (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const ggml_cgraph * cgraph);
    ggml_status    (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    ggml_status    (*graph_compute)     (ggml_backend_t backend, ggml_cgraph * cgraph);
    void           (*event_record)      (ggml_backend_t backend, ggml_backend_event_t event);
    void           (*event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
};

// 4. Device Interface (device capabilities)
struct ggml_backend_device_i {
    const char *          (*get_name)        (ggml_backend_dev_t dev);
    const char *          (*get_description) (ggml_backend_dev_t dev);
    void                  (*get_memory)      (ggml_backend_dev_t dev, size_t * free, size_t * total);
    ggml_backend_dev_type (*get_type)        (ggml_backend_dev_t dev);
    void                  (*get_props)       (ggml_backend_dev_t dev, ggml_backend_dev_props * props);
    ggml_backend_t        (*init_backend)    (ggml_backend_dev_t dev, const char * params);
    ggml_backend_buffer_type_t (*get_buffer_type)     (ggml_backend_dev_t dev);
    ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);
    ggml_backend_buffer_t      (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);
    bool                  (*supports_op)     (ggml_backend_dev_t dev, const ggml_tensor * op);
    bool                  (*supports_buft)   (ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
    bool                  (*offload_op)      (ggml_backend_dev_t dev, const ggml_tensor * op);
};

// 5. Registry Interface (backend discovery)
struct ggml_backend_reg_i {
    const char *       (*get_name)        (ggml_backend_reg_t reg);
    size_t             (*get_device_count)(ggml_backend_reg_t reg);
    ggml_backend_dev_t (*get_device)      (ggml_backend_reg_t reg, size_t index);
    void *             (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};
```

## Appendix B: Useful ggml Commands for Development

```bash
# Build with LNS enabled
cmake -B build -DGGML_LNS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Run backend operation tests
./build/bin/test-backend-ops -b LNS

# Run with specific backend
./build/bin/llama-cli -m model.gguf -p "Hello" -n 20

# List available backends/devices
./build/bin/llama-cli --list-devices

# Benchmark
./build/bin/llama-bench -m model.gguf
```

## Appendix C: References

1. M. G. Arnold et al., "Arithmetic cotransformations in the Real and Complex Logarithmic Number Systems," IEEE Trans. Comput., vol. 47, no. 7, 1998.
2. M. G. Arnold, "LPVIP: A Low-power ROM-Less ALU for Low-Precision LNS," PATMOS 2004.
3. ggml documentation: https://github.com/ggml-org/ggml
4. llama.cpp documentation: https://github.com/ggml-org/llama.cpp
5. xlnscpp: https://github.com/xlnsresearch/xlnscpp
6. xlns (Python): https://github.com/xlnsresearch/xlns
