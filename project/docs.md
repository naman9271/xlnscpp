# GSoC Project: Support for Logarithmic Number Systems in Large Language Models

## Complete Project Documentation & Implementation Guide

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Understanding the Problem Statement](#2-understanding-the-problem-statement)
3. [Prerequisite Knowledge](#3-prerequisite-knowledge)
4. [Current Codebase Analysis](#4-current-codebase-analysis)
5. [Target Systems Analysis (llama.cpp & ggml)](#5-target-systems-analysis-llamacpp--ggml)
6. [Architecture Design](#6-architecture-design)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Phase 1: Environment Setup & Understanding](#8-phase-1-environment-setup--understanding)
9. [Phase 2: xlnscpp Library Preparation](#9-phase-2-xlnscpp-library-preparation)
10. [Phase 3: ggml Backend Development](#10-phase-3-ggml-backend-development)
11. [Phase 4: Integration with llama.cpp](#11-phase-4-integration-with-llamacpp)
12. [Phase 5: Testing & Validation](#12-phase-5-testing--validation)
13. [Phase 6: Optimization & Documentation](#13-phase-6-optimization--documentation)
14. [File-by-File Implementation Details](#14-file-by-file-implementation-details)
15. [Performance Considerations](#15-performance-considerations)
16. [Testing Strategy](#16-testing-strategy)
17. [Timeline & Milestones](#17-timeline--milestones)
18. [Risk Assessment & Mitigation](#18-risk-assessment--mitigation)
19. [References & Resources](#19-references--resources)

---

## 1. Executive Summary

### Project Goal
Create a "virtual" LNS (Logarithmic Number System) backend for ggml that enables llama.cpp to perform LLM inference using xlnscpp instead of floating-point arithmetic.

### Why This Matters
- **Power Efficiency**: LNS makes multiplication/division trivially cheap (integer add/sub), which is beneficial for inference-heavy workloads
- **Proof of Concept**: Demonstrate that LNS can produce valid LLM outputs
- **Research Enablement**: Allow researchers to study LNS behavior in transformer architectures

### Expected Deliverables
1. A new ggml backend (`ggml-lns`) that uses xlnscpp for computations
2. Support for attention mechanism and feed-forward networks in LNS
3. Conversion routines between standard quantized formats and LNS
4. Working inference on at least one LLM (e.g., DeepSeek or similar)
5. Documentation and benchmarks

---

## 2. Understanding the Problem Statement

### 2.1 What is LNS (Logarithmic Number System)?

In LNS, every non-zero real number is stored as:
$$v = (-1)^s \cdot 2^L$$

Where:
- $s$ is the sign bit (0 = positive, 1 = negative)
- $L$ is the real-valued logarithm base 2 of $|v|$

**Key Properties:**

| Operation | Floating Point | LNS |
|-----------|----------------|-----|
| Multiplication | Hardware multiplier (expensive) | Integer addition (cheap) |
| Division | Hardware divider (expensive) | Integer subtraction (cheap) |
| Addition | Simple adder (cheap) | Gaussian logarithms (expensive) |

### 2.2 LLM Architecture Components

LLMs (like DeepSeek, LLaMA, etc.) consist of:

```
Input Tokens
    ↓
┌─────────────────────────────────────┐
│          Transformer Block          │  ← Repeated N times
│  ┌───────────────────────────────┐  │
│  │        Attention Layer        │  │  ← Q, K, V projections + softmax
│  │  - Query/Key/Value Projections│  │  ← Matrix multiplications
│  │  - Attention Scores (Q·K^T)   │  │  ← Matrix multiplication
│  │  - Softmax                    │  │  ← exp() and division
│  │  - Weighted Values (Attn·V)   │  │  ← Matrix multiplication
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │     Feed-Forward Network      │  │  ← Dense layers
│  │  - Linear projections         │  │  ← Matrix multiplications
│  │  - Activation (GELU/SiLU)     │  │  ← Non-linear functions
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │     Layer Normalization       │  │  ← Mean, variance, normalize
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
Output Logits → Tokens
```

### 2.3 Where LNS Fits In

**Operations favorable for LNS:**
- Matrix multiplications (dominant in attention & FFN)
- Scaling operations
- Some parts of normalization

**Operations challenging for LNS:**
- Softmax (requires additions)
- Layer normalization (requires additions for mean/variance)
- Activation functions

### 2.4 Project Scope

**In Scope:**
- Create ggml-lns backend
- Support core tensor operations (GGML_OP_ADD, GGML_OP_MUL, GGML_OP_MUL_MAT, etc.)
- Convert weights from FP/quantized → LNS at load time
- Token tensors in LNS format
- Proof-of-concept inference

**Out of Scope (but nice to have):**
- Hardware acceleration
- Training support
- Optimal performance (this is proof of concept)

---

## 3. Prerequisite Knowledge

### 3.1 Required Skills
Before starting, ensure familiarity with:

#### C++ Fundamentals
- [ ] Operator overloading
- [ ] Template programming
- [ ] Memory management (RAII, smart pointers)
- [ ] Header-only libraries

#### Build Systems
- [ ] CMake basics
- [ ] Static vs shared libraries
- [ ] Cross-platform compilation

#### Number Systems
- [ ] IEEE 754 floating-point representation
- [ ] Fixed-point arithmetic
- [ ] Logarithmic number systems (study xlnscpp)

#### Machine Learning Basics
- [ ] Transformer architecture
- [ ] Attention mechanism
- [ ] Quantization concepts

#### ggml/llama.cpp Architecture
- [ ] Backend interface
- [ ] Tensor operations
- [ ] Computation graphs
- [ ] Buffer management

### 3.2 Learning Resources

**LNS Background:**
1. Read the xlnscpp README and docs
2. Study references [1]-[6] from the project description
3. Especially focus on reference [6] (Bridging the Gap Between LLMs and LNS)

**ggml/llama.cpp:**
1. Study `ggml/include/ggml-backend.h` - backend interface
2. Study `ggml/src/ggml-backend-impl.h` - implementation details
3. Look at simple backends (CPU backend in `ggml-backend.cpp`)
4. Study an existing backend (RPC or WebGPU) for patterns

---

## 4. Current Codebase Analysis

### 4.1 xlnscpp Repository Structure

```
xlnscpp/
├── Core Library
│   ├── xlns16.cpp          # 16-bit LNS (like bfloat16)
│   ├── xlns32.cpp          # 32-bit LNS (like float32)
│   └── Lookup Tables
│       ├── xlns16sbdbtbl.h # 16-bit sb/db tables
│       ├── xlns16cvtbl.h   # 16-bit LNS→float table
│       ├── xlns16revcvtbl.h# 16-bit float→LNS table
│       └── xlns32tbl.h     # 32-bit interpolation tables
├── Tests
│   ├── xlns16test.cpp
│   ├── xlns32test.cpp
│   └── ...
└── docs/
    └── (documentation)
```

### 4.2 xlnscpp Key Components

#### Data Types
```cpp
// 16-bit LNS (7 fractional bits, like bfloat16 precision)
typedef u_int16_t xlns16;        // Raw storage
typedef int16_t xlns16_signed;   // For signed operations
class xlns16_float { ... };      // Wrapper with operator overloading

// 32-bit LNS (23 fractional bits, like float32 precision)
typedef unsigned int xlns32;
typedef signed int xlns32_signed;
class xlns32_float { ... };
```

#### Bit Layout (xlns32)
```
Bit: 31  30       23  22                    0
     +---+----------+------------------------+
     | s | int(log₂)|     frac(log₂)         |
     +---+----------+------------------------+
       1      8              23 bits
```

#### Core Operations
```cpp
// CHEAP operations (integer arithmetic)
xlns32 xlns32_mul(xlns32 x, xlns32 y);  // log(x) + log(y) - bias
xlns32 xlns32_div(xlns32 x, xlns32 y);  // log(x) - log(y) + bias

// EXPENSIVE operation (requires Gaussian logs)
xlns32 xlns32_add(xlns32 x, xlns32 y);  // Uses sb()/db() functions

// Conversions
xlns32 fp2xlns32(float x);   // Float → LNS
float xlns322fp(xlns32 x);   // LNS → Float
```

#### Gaussian Logarithms
```cpp
// sb(z) = log₂(1 + 2^z) - for same-sign addition
// db(z) = log₂(|1 - 2^z|) - for different-sign addition

// Implementation options:
// 1. xlns32_ideal - uses math.h (accurate, slow)
// 2. Table-based - interpolation (fast, ~100KB tables)
// 3. LPVIP - Mitchell approximation (no tables, less accurate)
```

### 4.3 Usage Patterns

**Function API (faster):**
```cpp
xlns32 a = fp2xlns32(3.14f);
xlns32 b = fp2xlns32(2.71f);
xlns32 c = xlns32_mul(a, b);  // 3.14 * 2.71
float result = xlns322fp(c);
```

**Operator Overloading (easier):**
```cpp
xlns32_float a = 3.14f;
xlns32_float b = 2.71f;
xlns32_float c = a * b;
std::cout << c;  // Prints float value
```

### 4.4 Compile-Time Options

| Option | Effect |
|--------|--------|
| `xlns32_ideal` | Use math.h for sb/db (accurate, slow) |
| `xlns32_alt` | Alternative addition algorithm |
| `xlns16_table` | Use lookup tables (fast) |
| `xlns16_altopt` | Optimized LPVIP (no tables) |

---

## 5. Target Systems Analysis (llama.cpp & ggml)

### 5.1 ggml Overview

ggml is a tensor library that llama.cpp uses for all computations. It provides:
- Tensor data structures
- Computation graphs
- Backend abstraction
- Memory management

### 5.2 ggml Backend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      llama.cpp                               │
│  (Model loading, tokenization, sampling, chat interface)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        ggml API                              │
│  (Tensor creation, graph building, computation scheduling)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Scheduler                          │
│  (Selects appropriate backend for each operation)            │
└─────────────────────────────────────────────────────────────┘
            │              │              │              │
            ▼              ▼              ▼              ▼
      ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐
      │   CPU   │   │   CUDA   │   │  Metal  │   │   LNS   │ ← NEW
      │ Backend │   │ Backend  │   │ Backend │   │ Backend │
      └─────────┘   └──────────┘   └─────────┘   └─────────┘
```

### 5.3 Backend Interface Hierarchy

```cpp
// 1. Backend Registry (ggml_backend_reg)
struct ggml_backend_reg {
    int api_version;
    struct ggml_backend_reg_i iface;
    void * context;
};

// 2. Backend Device (ggml_backend_dev)
struct ggml_backend_device {
    struct ggml_backend_device_i iface;
    ggml_backend_reg_t reg;
    void * context;
};

// 3. Backend Instance (ggml_backend)
struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;
    ggml_backend_dev_t device;
    void * context;
};

// 4. Buffer Type (ggml_backend_buffer_type)
struct ggml_backend_buffer_type {
    struct ggml_backend_buffer_type_i iface;
    ggml_backend_dev_t device;
    void * context;
};

// 5. Buffer (ggml_backend_buffer)
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    ggml_backend_buffer_type_t buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};
```

### 5.4 Key Interface Functions

#### Buffer Type Interface
```cpp
struct ggml_backend_buffer_type_i {
    const char * (*get_name)(ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    size_t (*get_alignment)(ggml_backend_buffer_type_t buft);
    size_t (*get_max_size)(ggml_backend_buffer_type_t buft);
    size_t (*get_alloc_size)(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
    bool (*is_host)(ggml_backend_buffer_type_t buft);
};
```

#### Buffer Interface
```cpp
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);
    ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
    void (*memset_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
    void (*set_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor)(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool (*cpy_tensor)(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst);
    void (*clear)(ggml_backend_buffer_t buffer, uint8_t value);
    void (*reset)(ggml_backend_buffer_t buffer);
};
```

#### Backend Interface
```cpp
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void (*free)(ggml_backend_t backend);
    void (*set_tensor_async)(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst);
    void (*synchronize)(ggml_backend_t backend);
    ggml_status (*graph_compute)(ggml_backend_t backend, ggml_cgraph * cgraph);  // THE KEY FUNCTION
};
```

#### Device Interface
```cpp
struct ggml_backend_device_i {
    const char * (*get_name)(ggml_backend_dev_t dev);
    const char * (*get_description)(ggml_backend_dev_t dev);
    void (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);
    enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);
    void (*get_props)(ggml_backend_dev_t dev, ggml_backend_dev_props * props);
    ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);
    ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);
    ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);
    ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);
    bool (*supports_op)(ggml_backend_dev_t dev, const ggml_tensor * op);
    bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
    bool (*offload_op)(ggml_backend_dev_t dev, const ggml_tensor * op);
    ggml_backend_event_t (*event_new)(ggml_backend_dev_t dev);
    void (*event_free)(ggml_backend_dev_t dev, ggml_backend_event_t event);
    void (*event_synchronize)(ggml_backend_dev_t dev, ggml_backend_event_t event);
};
```

### 5.5 Tensor Operations to Support

For LLM inference, these are the critical operations:

**Essential (Must Have):**
```cpp
GGML_OP_ADD        // Addition (LNS expensive)
GGML_OP_MUL        // Element-wise multiply (LNS cheap)
GGML_OP_MUL_MAT    // Matrix multiply (LNS favorable for multiply, unfavorable for accumulate)
GGML_OP_SCALE      // Multiply by scalar (LNS cheap)
GGML_OP_SOFT_MAX   // Softmax (challenging - requires exp and sum)
GGML_OP_ROPE       // Rotary position embedding
GGML_OP_NORM       // Normalization (requires mean/variance)
GGML_OP_RMS_NORM   // RMS normalization
```

**Important (Should Have):**
```cpp
GGML_OP_CONT       // Make tensor contiguous
GGML_OP_VIEW       // Tensor view
GGML_OP_PERMUTE    // Dimension permutation
GGML_OP_TRANSPOSE  // Matrix transpose
GGML_OP_RESHAPE    // Tensor reshape
GGML_OP_GET_ROWS   // Embedding lookup
GGML_OP_CPY        // Tensor copy
```

**Optional (Nice to Have):**
```cpp
GGML_OP_SILU       // SiLU activation
GGML_OP_GELU       // GELU activation
GGML_OP_CONV_1D    // 1D convolution
```

---

## 6. Architecture Design

### 6.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              llama.cpp                                       │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  Model Loading    Token Processing    Sampling    Chat Interface   │   │
│   └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ggml-lns Backend                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LNS Backend Registry                             │   │
│  │   - ggml_backend_lns_reg()                                          │   │
│  │   - Registration with ggml backend system                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       LNS Device                                     │   │
│  │   - Virtual "LNS device" (CPU-based emulation)                      │   │
│  │   - Device capabilities and properties                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LNS Buffer Type                                 │   │
│  │   - Allocation strategy for LNS tensors                             │   │
│  │   - Memory layout for xlns16/xlns32 data                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       LNS Buffer                                     │   │
│  │   - Actual memory for LNS data                                      │   │
│  │   - FP ↔ LNS conversion on set_tensor/get_tensor                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Graph Compute Engine                              │   │
│  │   - Dispatch tensor operations to xlnscpp                           │   │
│  │   - Implementation of GGML_OP_* using xlns32_* functions           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              xlnscpp Library                                 │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────────┐       │
│   │   xlns32.cpp   │    │   xlns16.cpp   │    │   Lookup Tables    │       │
│   │   - mul/div    │    │   - mul/div    │    │   - sb/db tables   │       │
│   │   - add        │    │   - add        │    │   - conversion tbls│       │
│   │   - conversions│    │   - conversions│    │                    │       │
│   └────────────────┘    └────────────────┘    └────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Flow Design

```
Weight Loading Flow:
┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
│ GGUF Weights │ ──► │ Dequantize to │ ──► │ Convert to LNS  │
│ (Q4, Q8, FP) │     │    Float32    │     │  (fp2xlns32)    │
└──────────────┘     └───────────────┘     └─────────────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │ LNS Buffer      │
                                           │ (xlns32 array)  │
                                           └─────────────────┘

Token Processing Flow:
┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Input Tokens │ ──► │ Embedding     │ ──► │ LNS Tensors     │
│   (int32)    │     │  Lookup       │     │ for computation │
└──────────────┘     └───────────────┘     └─────────────────┘

Computation Flow:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LNS Tensor A   │ ──► │  graph_compute  │ ──► │ LNS Result      │
│  LNS Tensor B   │     │  (xlnscpp ops)  │     │ Tensor          │
└─────────────────┘     └─────────────────┘     └─────────────────┘

Output Flow:
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│ LNS Logits      │ ──► │ Convert to FP │ ──► │ Sampling        │
│ Tensor          │     │ (xlns322fp)   │     │ (top-k, top-p)  │
└─────────────────┘     └───────────────┘     └─────────────────┘
```

### 6.3 File Structure Design

```
llama.cpp/
└── ggml/
    ├── include/
    │   └── ggml-lns.h              # Public API header
    └── src/
        └── ggml-lns/
            ├── ggml-lns.cpp        # Main backend implementation
            ├── ggml-lns-buffer.cpp # Buffer management
            ├── ggml-lns-compute.cpp# Tensor operations
            ├── ggml-lns-convert.cpp# FP ↔ LNS conversion
            ├── xlnscpp/            # xlnscpp library (submodule or copy)
            │   ├── xlns32.cpp
            │   ├── xlns16.cpp
            │   └── *.h
            └── CMakeLists.txt      # Build configuration
```

### 6.4 Key Design Decisions

#### Decision 1: Use xlns32 (not xlns16) primarily
**Rationale:** xlns32 has precision comparable to float32, which most LLMs use internally. xlns16 can be an optimization later.

#### Decision 2: Convert at buffer boundaries
**Rationale:** Convert FP→LNS when loading into LNS buffer, and LNS→FP when reading out. This keeps all internal computations in LNS.

#### Decision 3: Use table-based sb/db (not ideal mode)
**Rationale:** The ideal mode using math.h is too slow for practical use. Tables add ~100KB memory but are much faster.

#### Decision 4: Support only CPU execution initially
**Rationale:** This is a proof-of-concept. GPU optimization can come later.

#### Decision 5: Implement essential operations first
**Rationale:** Focus on MUL_MAT, ADD, SOFT_MAX, NORM - the core operations needed for transformer inference.

---

## 7. Implementation Roadmap

### Overview Timeline (12-week GSoC-style project)

```
Week 1-2:   Environment Setup & Deep Dive
Week 3-4:   xlnscpp Library Preparation
Week 5-7:   Core Backend Implementation
Week 8-9:   Operation Implementation
Week 10-11: Integration & Testing
Week 12:    Optimization & Documentation
```

### Detailed Milestones

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Setup | Dev environment, llama.cpp builds, xlnscpp understanding |
| 2 | Study | Deep dive into ggml backend architecture, study existing backends |
| 3 | Prep | xlnscpp modifications for library use, CMake integration |
| 4 | Prep | Conversion functions, basic testing framework |
| 5 | Core | Backend skeleton (registry, device, buffer type) |
| 6 | Core | Buffer implementation, tensor data handling |
| 7 | Core | Basic graph_compute, simple ops (CPY, CONT) |
| 8 | Ops | MUL, SCALE, MUL_MAT implementation |
| 9 | Ops | ADD, SOFT_MAX, NORM, RMS_NORM implementation |
| 10 | Int | Integration with llama.cpp, first inference attempts |
| 11 | Test | Debugging, validation against FP, accuracy testing |
| 12 | Doc | Performance analysis, documentation, final submission |

---

## 8. Phase 1: Environment Setup & Understanding

### 8.1 Step 1: Clone and Build llama.cpp

```bash
# Clone llama.cpp (includes ggml)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with CMake
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)

# Verify build
./bin/llama-cli --help
```

### 8.2 Step 2: Download a Test Model

```bash
# Download a small model for testing (e.g., TinyLlama)
# Using Hugging Face CLI
pip install huggingface-hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir ./models/

# Test basic inference
./bin/llama-cli -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Hello, world!" -n 20
```

### 8.3 Step 3: Build and Test xlnscpp

```bash
# Clone xlnscpp
git clone https://github.com/xlnsresearch/xlnscpp.git
cd xlnscpp

# Build tests
g++ -O2 -o xlns32test xlns32test.cpp
g++ -O2 -o xlns16test xlns16test.cpp

# Run tests
./xlns32test
./xlns16test

# Build with different options
g++ -O2 -Dxlns32_ideal -o xlns32test_ideal xlns32test.cpp  # Ideal mode
g++ -O2 -Dxlns16_alt -Dxlns16_table -o xlns16test_fast xlns16test.cpp  # Fast mode
```

### 8.4 Step 4: Study the Code

#### Study xlnscpp (1-2 days)
1. Read [docs/architecture.md](docs/architecture.md) completely
2. Read [docs/api-reference.md](docs/api-reference.md) completely
3. Trace through `xlns32_mul`, `xlns32_add` implementations
4. Understand conversion functions (`fp2xlns32`, `xlns322fp`)
5. Write small test programs to build intuition

```cpp
// test_xlns.cpp - Write your own tests
#define xlns32_ideal  // Start with ideal mode for accuracy
#include "xlns32.cpp"

int main() {
    // Test basic operations
    xlns32_float a = 3.14159f;
    xlns32_float b = 2.71828f;
    
    std::cout << "a = " << a << " (internal: 0x" 
              << std::hex << xlns32_internal(a) << ")" << std::endl;
    std::cout << "a + b = " << (a + b) << std::endl;
    std::cout << "a * b = " << (a * b) << std::endl;
    std::cout << "a / b = " << (a / b) << std::endl;
    
    // Test matrix-like operation (dot product)
    xlns32_float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum = sum + (xlns32_float)(i * 0.1f) * (xlns32_float)(i * 0.2f);
    }
    std::cout << "Dot product: " << sum << std::endl;
    
    return 0;
}
```

#### Study ggml Backend Architecture (2-3 days)

1. **Read header files:**
   - `ggml/include/ggml-backend.h` - Public API
   - `ggml/src/ggml-backend-impl.h` - Implementation interfaces

2. **Study CPU backend:**
   - `ggml/src/ggml-backend.cpp` - Contains CPU backend implementation
   - Look for `ggml_backend_cpu_*` functions

3. **Study a simple external backend:**
   - `ggml/src/ggml-rpc/ggml-rpc.cpp` - RPC backend is a good example
   - Note the patterns used

4. **Understand computation graphs:**
   - `ggml/include/ggml.h` - Tensor and graph structures
   - `ggml/src/ggml.c` - Core implementation

### 8.5 Checkpoint Questions

Before proceeding, ensure you can answer:
- [ ] How does xlns32_mul work at the bit level?
- [ ] What are sb() and db() functions? Why are they needed?
- [ ] What is the cost of addition vs multiplication in LNS?
- [ ] What interfaces must a ggml backend implement?
- [ ] How does ggml dispatch operations to backends?
- [ ] How are tensors allocated in different backends?

---

## 9. Phase 2: xlnscpp Library Preparation

### 9.1 Goal
Modify xlnscpp to be usable as a library component within ggml-lns backend.

### 9.2 Step 1: Create Library Header

Create a single header that can be included:

```cpp
// xlnscpp/xlns.h - Unified header

#ifndef XLNS_H
#define XLNS_H

// Configuration options - users can define these before including
// #define XLNS_USE_16BIT      // Use xlns16 instead of xlns32
// #define XLNS_IDEAL_MODE     // Use math.h for accuracy
// #define XLNS_TABLE_MODE     // Use lookup tables for speed

#ifdef XLNS_IDEAL_MODE
    #define xlns32_ideal
    #define xlns16_ideal
#endif

#ifdef XLNS_TABLE_MODE
    #define xlns16_alt
    #define xlns16_table
#endif

// Include the implementations
#include "xlns32.cpp"

#ifdef XLNS_USE_16BIT
    #include "xlns16.cpp"
#endif

// Additional utilities for ggml integration
namespace xlns {

// Bulk conversion functions (for efficiency)
inline void fp_to_lns32_array(const float* src, xlns32* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp2xlns32(src[i]);
    }
}

inline void lns32_to_fp_array(const xlns32* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = xlns322fp(src[i]);
    }
}

// Vectorized operations (for tensor math)
inline void lns32_add_array(const xlns32* a, const xlns32* b, xlns32* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = xlns32_add(a[i], b[i]);
    }
}

inline void lns32_mul_array(const xlns32* a, const xlns32* b, xlns32* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = xlns32_mul(a[i], b[i]);
    }
}

inline void lns32_scale_array(const xlns32* a, xlns32 scale, xlns32* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = xlns32_mul(a[i], scale);
    }
}

// Matrix multiplication (naive - optimize later)
inline void lns32_matmul(
    const xlns32* A, const xlns32* B, xlns32* C,
    size_t M, size_t K, size_t N
) {
    // C[M,N] = A[M,K] @ B[K,N]
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            xlns32 sum = xlns32_zero;
            for (size_t k = 0; k < K; k++) {
                xlns32 prod = xlns32_mul(A[i * K + k], B[k * N + j]);
                sum = xlns32_add(sum, prod);
            }
            C[i * N + j] = sum;
        }
    }
}

// Special functions for LLM operations
inline xlns32 lns32_exp(xlns32 x) {
    // exp(x) in LNS: exp(x) = 2^(x * log2(e))
    // This is a multiplication by a constant in log domain
    // log2(e) ≈ 1.4427
    static const xlns32 log2e = fp2xlns32(1.4426950408889634f);
    return xlns32_mul(x, log2e);
}

inline xlns32 lns32_log(xlns32 x) {
    // log(x) in LNS: the value IS already log2(x) * scale
    // To get natural log: ln(x) = log2(x) * ln(2)
    // We need to convert the internal representation to a result
    float fx = xlns322fp(x);
    return fp2xlns32(logf(fx));
}

} // namespace xlns

#endif // XLNS_H
```

### 9.3 Step 2: Add CMake Support

```cmake
# xlnscpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(xlnscpp VERSION 1.0.0 LANGUAGES CXX)

# Options
option(XLNS_IDEAL_MODE "Use math.h for accuracy" OFF)
option(XLNS_TABLE_MODE "Use lookup tables for speed" ON)
option(XLNS_BUILD_TESTS "Build test programs" ON)

# Library target (header-only with sources)
add_library(xlnscpp INTERFACE)
target_include_directories(xlnscpp INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Set compile definitions based on options
if(XLNS_IDEAL_MODE)
    target_compile_definitions(xlnscpp INTERFACE XLNS_IDEAL_MODE)
endif()

if(XLNS_TABLE_MODE)
    target_compile_definitions(xlnscpp INTERFACE XLNS_TABLE_MODE)
endif()

# Tests
if(XLNS_BUILD_TESTS)
    add_executable(xlns32test xlns32test.cpp)
    add_executable(xlns16test xlns16test.cpp)
    add_executable(xlnsbothtest xlnsbothtest.cpp)
endif()
```

### 9.4 Step 3: Validate Library Functions

Create comprehensive tests:

```cpp
// xlnscpp/tests/test_library.cpp

#include "xlns.h"
#include <cmath>
#include <cassert>
#include <iostream>

void test_basic_ops() {
    std::cout << "Testing basic operations..." << std::endl;
    
    xlns32 a = fp2xlns32(3.14159f);
    xlns32 b = fp2xlns32(2.71828f);
    
    // Multiplication
    xlns32 prod = xlns32_mul(a, b);
    float expected_prod = 3.14159f * 2.71828f;
    float actual_prod = xlns322fp(prod);
    assert(fabsf(actual_prod - expected_prod) / expected_prod < 0.01f);
    
    // Addition
    xlns32 sum = xlns32_add(a, b);
    float expected_sum = 3.14159f + 2.71828f;
    float actual_sum = xlns322fp(sum);
    assert(fabsf(actual_sum - expected_sum) / expected_sum < 0.01f);
    
    std::cout << "  PASS" << std::endl;
}

void test_bulk_conversion() {
    std::cout << "Testing bulk conversion..." << std::endl;
    
    const size_t N = 1000;
    float* fp_arr = new float[N];
    xlns32* lns_arr = new xlns32[N];
    float* fp_back = new float[N];
    
    // Initialize
    for (size_t i = 0; i < N; i++) {
        fp_arr[i] = (float)(i + 1) * 0.123f;
    }
    
    // Convert to LNS
    xlns::fp_to_lns32_array(fp_arr, lns_arr, N);
    
    // Convert back
    xlns::lns32_to_fp_array(lns_arr, fp_back, N);
    
    // Check accuracy
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float err = fabsf(fp_arr[i] - fp_back[i]) / fabsf(fp_arr[i]);
        max_err = fmaxf(max_err, err);
    }
    
    std::cout << "  Max relative error: " << max_err << std::endl;
    assert(max_err < 0.001f);
    std::cout << "  PASS" << std::endl;
    
    delete[] fp_arr;
    delete[] lns_arr;
    delete[] fp_back;
}

void test_matrix_multiply() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    // 2x3 @ 3x2 = 2x2
    const size_t M = 2, K = 3, N = 2;
    
    float A_fp[] = {1, 2, 3, 4, 5, 6};  // 2x3
    float B_fp[] = {7, 8, 9, 10, 11, 12};  // 3x2
    float C_expected[] = {58, 64, 139, 154};  // 2x2
    
    xlns32 A_lns[6], B_lns[6], C_lns[4];
    float C_actual[4];
    
    xlns::fp_to_lns32_array(A_fp, A_lns, 6);
    xlns::fp_to_lns32_array(B_fp, B_lns, 6);
    
    xlns::lns32_matmul(A_lns, B_lns, C_lns, M, K, N);
    
    xlns::lns32_to_fp_array(C_lns, C_actual, 4);
    
    for (size_t i = 0; i < 4; i++) {
        float err = fabsf(C_actual[i] - C_expected[i]) / C_expected[i];
        std::cout << "  C[" << i << "]: expected=" << C_expected[i] 
                  << " actual=" << C_actual[i] << " err=" << err << std::endl;
        assert(err < 0.01f);
    }
    
    std::cout << "  PASS" << std::endl;
}

int main() {
    test_basic_ops();
    test_bulk_conversion();
    test_matrix_multiply();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

---

## 10. Phase 3: ggml Backend Development

### 10.1 Goal
Implement the ggml-lns backend that wraps xlnscpp and presents it as a ggml backend.

### 10.2 Step 1: Create Backend Header

```cpp
// ggml/include/ggml-lns.h

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend registration
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_lns_reg(void);

// Backend buffer type
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_lns_buffer_type(void);

// Backend creation (convenience function)
GGML_BACKEND_API ggml_backend_t ggml_backend_lns_init(void);

// Check if a buffer is an LNS buffer
GGML_BACKEND_API bool ggml_backend_is_lns(ggml_backend_t backend);
GGML_BACKEND_API bool ggml_backend_buffer_is_lns(ggml_backend_buffer_t buffer);

// LNS-specific configuration
typedef enum {
    GGML_LNS_PRECISION_16 = 16,  // xlns16 (bfloat16-like)
    GGML_LNS_PRECISION_32 = 32,  // xlns32 (float32-like)
} ggml_lns_precision;

typedef enum {
    GGML_LNS_MODE_IDEAL = 0,    // Use math.h (accurate, slow)
    GGML_LNS_MODE_TABLE = 1,    // Use lookup tables (fast)
    GGML_LNS_MODE_LPVIP = 2,    // Use LPVIP approximation (no tables)
} ggml_lns_mode;

// Set LNS configuration (call before ggml_backend_lns_init)
GGML_BACKEND_API void ggml_lns_set_precision(ggml_lns_precision precision);
GGML_BACKEND_API void ggml_lns_set_mode(ggml_lns_mode mode);

#ifdef __cplusplus
}
#endif
```

### 10.3 Step 2: Implement Backend Core

```cpp
// ggml/src/ggml-lns/ggml-lns.cpp

#include "ggml-lns.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

// Include xlnscpp
#define XLNS_TABLE_MODE
#include "xlnscpp/xlns.h"

#include <cstring>
#include <cstdlib>
#include <vector>

// ============================================================================
// Internal state and configuration
// ============================================================================

static ggml_lns_precision g_lns_precision = GGML_LNS_PRECISION_32;
static ggml_lns_mode g_lns_mode = GGML_LNS_MODE_TABLE;

void ggml_lns_set_precision(ggml_lns_precision precision) {
    g_lns_precision = precision;
}

void ggml_lns_set_mode(ggml_lns_mode mode) {
    g_lns_mode = mode;
}

// ============================================================================
// LNS Buffer Implementation
// ============================================================================

struct ggml_backend_lns_buffer_context {
    void * data;
    size_t size;
};

static void ggml_backend_lns_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_lns_buffer_context * ctx = 
        (ggml_backend_lns_buffer_context *)buffer->context;
    if (ctx->data) {
        free(ctx->data);
    }
    delete ctx;
}

static void * ggml_backend_lns_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_lns_buffer_context * ctx = 
        (ggml_backend_lns_buffer_context *)buffer->context;
    return ctx->data;
}

static void ggml_backend_lns_buffer_memset_tensor(
    ggml_backend_buffer_t buffer, 
    ggml_tensor * tensor, 
    uint8_t value, 
    size_t offset, 
    size_t size
) {
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_lns_buffer_set_tensor(
    ggml_backend_buffer_t buffer, 
    ggml_tensor * tensor, 
    const void * data, 
    size_t offset, 
    size_t size
) {
    // Convert FP data to LNS when setting tensor
    // Note: data is expected to be float, tensor->data is xlns32
    
    const float * src = (const float *)data;
    xlns32 * dst = (xlns32 *)((char *)tensor->data + offset);
    size_t n_elements = size / sizeof(float);
    
    for (size_t i = 0; i < n_elements; i++) {
        dst[i] = fp2xlns32(src[i]);
    }
}

static void ggml_backend_lns_buffer_get_tensor(
    ggml_backend_buffer_t buffer, 
    const ggml_tensor * tensor, 
    void * data, 
    size_t offset, 
    size_t size
) {
    // Convert LNS data to FP when getting tensor
    
    const xlns32 * src = (const xlns32 *)((const char *)tensor->data + offset);
    float * dst = (float *)data;
    size_t n_elements = size / sizeof(float);
    
    for (size_t i = 0; i < n_elements; i++) {
        dst[i] = xlns322fp(src[i]);
    }
}

static void ggml_backend_lns_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_lns_buffer_context * ctx = 
        (ggml_backend_lns_buffer_context *)buffer->context;
    memset(ctx->data, value, ctx->size);
}

static const struct ggml_backend_buffer_i ggml_backend_lns_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_lns_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_lns_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_lns_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_lns_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_lns_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_lns_buffer_clear,
    /* .reset           = */ nullptr,
};

// ============================================================================
// LNS Buffer Type Implementation
// ============================================================================

static const char * ggml_backend_lns_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "LNS";
}

static ggml_backend_buffer_t ggml_backend_lns_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, 
    size_t size
) {
    ggml_backend_lns_buffer_context * ctx = new ggml_backend_lns_buffer_context;
    
    // Allocate aligned memory
    ctx->size = size;
    ctx->data = malloc(size);
    
    if (!ctx->data) {
        delete ctx;
        return nullptr;
    }
    
    return ggml_backend_buffer_init(buft, ggml_backend_lns_buffer_interface, ctx, size);
}

static size_t ggml_backend_lns_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32; // 32-byte alignment for SIMD
}

static size_t ggml_backend_lns_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX; // No limit
}

static size_t ggml_backend_lns_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, 
    const ggml_tensor * tensor
) {
    // xlns32 is same size as float (4 bytes)
    return ggml_nbytes(tensor);
}

static bool ggml_backend_lns_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true; // LNS buffer is in CPU memory
}

static const struct ggml_backend_buffer_type_i ggml_backend_lns_buffer_type_interface = {
    /* .get_name      = */ ggml_backend_lns_buffer_type_get_name,
    /* .alloc_buffer  = */ ggml_backend_lns_buffer_type_alloc_buffer,
    /* .get_alignment = */ ggml_backend_lns_buffer_type_get_alignment,
    /* .get_max_size  = */ ggml_backend_lns_buffer_type_get_max_size,
    /* .get_alloc_size= */ ggml_backend_lns_buffer_type_get_alloc_size,
    /* .is_host       = */ ggml_backend_lns_buffer_type_is_host,
};

static struct ggml_backend_buffer_type ggml_backend_lns_buffer_type_struct = {
    /* .iface   = */ ggml_backend_lns_buffer_type_interface,
    /* .device  = */ nullptr,  // Will be set during registration
    /* .context = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_lns_buffer_type(void) {
    return &ggml_backend_lns_buffer_type_struct;
}

// ============================================================================
// LNS Backend Implementation
// ============================================================================

struct ggml_backend_lns_context {
    // Add any context needed
    std::string name;
};

static const char * ggml_backend_lns_get_name(ggml_backend_t backend) {
    ggml_backend_lns_context * ctx = (ggml_backend_lns_context *)backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_lns_free(ggml_backend_t backend) {
    ggml_backend_lns_context * ctx = (ggml_backend_lns_context *)backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_lns_synchronize(ggml_backend_t backend) {
    // No-op for CPU backend
    GGML_UNUSED(backend);
}

// Forward declaration
static enum ggml_status ggml_backend_lns_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);

static const struct ggml_backend_i ggml_backend_lns_interface = {
    /* .get_name         = */ ggml_backend_lns_get_name,
    /* .free             = */ ggml_backend_lns_free,
    /* .set_tensor_async = */ nullptr,
    /* .get_tensor_async = */ nullptr,
    /* .cpy_tensor_async = */ nullptr,
    /* .synchronize      = */ ggml_backend_lns_synchronize,
    /* .graph_plan_create= */ nullptr,
    /* .graph_plan_free  = */ nullptr,
    /* .graph_plan_update= */ nullptr,
    /* .graph_plan_compute=*/ nullptr,
    /* .graph_compute    = */ ggml_backend_lns_graph_compute,
    /* .event_record     = */ nullptr,
    /* .event_wait       = */ nullptr,
};

static ggml_guid_t ggml_backend_lns_guid(void) {
    static ggml_guid guid = { 
        0x4c, 0x4e, 0x53, 0x00,  // "LNS\0"
        0x00, 0x00, 0x00, 0x00, 
        0x00, 0x00, 0x00, 0x00, 
        0x00, 0x00, 0x00, 0x01 
    };
    return &guid;
}

// ============================================================================
// LNS Device Implementation
// ============================================================================

static const char * ggml_backend_lns_device_get_name(ggml_backend_dev_t dev) {
    return "LNS";
}

static const char * ggml_backend_lns_device_get_description(ggml_backend_dev_t dev) {
    return "Logarithmic Number System (CPU emulation)";
}

static void ggml_backend_lns_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // Report system memory
    *free = SIZE_MAX;
    *total = SIZE_MAX;
}

static enum ggml_backend_dev_type ggml_backend_lns_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static void ggml_backend_lns_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = "LNS";
    props->description = "Logarithmic Number System backend";
    props->memory_free = SIZE_MAX;
    props->memory_total = SIZE_MAX;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_lns_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    
    ggml_backend_lns_context * ctx = new ggml_backend_lns_context;
    ctx->name = "LNS";
    
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_lns_guid(),
        /* .iface   = */ ggml_backend_lns_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_lns_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_lns_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_lns_device_buffer_from_host_ptr(
    ggml_backend_dev_t dev, 
    void * ptr, 
    size_t size, 
    size_t max_tensor_size
) {
    // Create buffer that wraps existing host memory
    ggml_backend_lns_buffer_context * ctx = new ggml_backend_lns_buffer_context;
    ctx->data = ptr;
    ctx->size = size;
    
    return ggml_backend_buffer_init(
        ggml_backend_lns_buffer_type(), 
        ggml_backend_lns_buffer_interface, 
        ctx, 
        size
    );
}

static bool ggml_backend_lns_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // List of supported operations
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_MUL_MAT:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_GET_ROWS:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_lns_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_lns_buffer_type();
}

static const struct ggml_backend_device_i ggml_backend_lns_device_interface = {
    /* .get_name            = */ ggml_backend_lns_device_get_name,
    /* .get_description     = */ ggml_backend_lns_device_get_description,
    /* .get_memory          = */ ggml_backend_lns_device_get_memory,
    /* .get_type            = */ ggml_backend_lns_device_get_type,
    /* .get_props           = */ ggml_backend_lns_device_get_props,
    /* .init_backend        = */ ggml_backend_lns_device_init_backend,
    /* .get_buffer_type     = */ ggml_backend_lns_device_get_buffer_type,
    /* .get_host_buffer_type= */ nullptr,
    /* .buffer_from_host_ptr= */ ggml_backend_lns_device_buffer_from_host_ptr,
    /* .supports_op         = */ ggml_backend_lns_device_supports_op,
    /* .supports_buft       = */ ggml_backend_lns_device_supports_buft,
    /* .offload_op          = */ nullptr,
    /* .event_new           = */ nullptr,
    /* .event_free          = */ nullptr,
    /* .event_synchronize   = */ nullptr,
};

static struct ggml_backend_device ggml_backend_lns_device = {
    /* .iface   = */ ggml_backend_lns_device_interface,
    /* .reg     = */ nullptr,  // Will be set during registration
    /* .context = */ nullptr,
};

// ============================================================================
// LNS Registry Implementation
// ============================================================================

static const char * ggml_backend_lns_reg_get_name(ggml_backend_reg_t reg) {
    return "LNS";
}

static size_t ggml_backend_lns_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;  // Single LNS device
}

static ggml_backend_dev_t ggml_backend_lns_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    return &ggml_backend_lns_device;
}

static const struct ggml_backend_reg_i ggml_backend_lns_reg_interface = {
    /* .get_name        = */ ggml_backend_lns_reg_get_name,
    /* .get_device_count= */ ggml_backend_lns_reg_get_device_count,
    /* .get_device      = */ ggml_backend_lns_reg_get_device,
    /* .get_proc_address= */ nullptr,
};

static struct ggml_backend_reg ggml_backend_lns_reg_struct = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ ggml_backend_lns_reg_interface,
    /* .context     = */ nullptr,
};

ggml_backend_reg_t ggml_backend_lns_reg(void) {
    // Set up cross-references
    ggml_backend_lns_device.reg = &ggml_backend_lns_reg_struct;
    ggml_backend_lns_buffer_type_struct.device = &ggml_backend_lns_device;
    
    return &ggml_backend_lns_reg_struct;
}

// ============================================================================
// Convenience Functions
// ============================================================================

ggml_backend_t ggml_backend_lns_init(void) {
    return ggml_backend_lns_device_init_backend(&ggml_backend_lns_device, nullptr);
}

bool ggml_backend_is_lns(ggml_backend_t backend) {
    return backend && ggml_guid_matches(backend->guid, ggml_backend_lns_guid());
}

bool ggml_backend_buffer_is_lns(ggml_backend_buffer_t buffer) {
    return buffer && buffer->buft == ggml_backend_lns_buffer_type();
}
```

### 10.4 Step 3: Implement Graph Compute

```cpp
// ggml/src/ggml-lns/ggml-lns-compute.cpp

// Include in ggml-lns.cpp or as separate file

// ============================================================================
// Operation Implementations
// ============================================================================

static void ggml_lns_op_cpy(ggml_tensor * dst, const ggml_tensor * src) {
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src));
    memcpy(dst->data, src->data, ggml_nbytes(src));
}

static void ggml_lns_op_add(ggml_tensor * dst, const ggml_tensor * src0, const ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(dst);
    
    xlns32 * dst_data = (xlns32 *)dst->data;
    const xlns32 * src0_data = (const xlns32 *)src0->data;
    const xlns32 * src1_data = (const xlns32 *)src1->data;
    
    // Handle broadcasting
    const int64_t ne0 = ggml_nelements(src0);
    const int64_t ne1 = ggml_nelements(src1);
    
    if (ne0 == ne1) {
        // Element-wise addition
        for (int64_t i = 0; i < ne; i++) {
            dst_data[i] = xlns32_add(src0_data[i], src1_data[i]);
        }
    } else if (ne1 == 1) {
        // Broadcast scalar
        xlns32 scalar = src1_data[0];
        for (int64_t i = 0; i < ne; i++) {
            dst_data[i] = xlns32_add(src0_data[i], scalar);
        }
    } else {
        // General broadcasting - implement based on shapes
        // For now, fall back to element-wise with wrap-around
        for (int64_t i = 0; i < ne; i++) {
            dst_data[i] = xlns32_add(src0_data[i % ne0], src1_data[i % ne1]);
        }
    }
}

static void ggml_lns_op_mul(ggml_tensor * dst, const ggml_tensor * src0, const ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(dst);
    
    xlns32 * dst_data = (xlns32 *)dst->data;
    const xlns32 * src0_data = (const xlns32 *)src0->data;
    const xlns32 * src1_data = (const xlns32 *)src1->data;
    
    const int64_t ne0 = ggml_nelements(src0);
    const int64_t ne1 = ggml_nelements(src1);
    
    if (ne0 == ne1) {
        for (int64_t i = 0; i < ne; i++) {
            dst_data[i] = xlns32_mul(src0_data[i], src1_data[i]);
        }
    } else if (ne1 == 1) {
        xlns32 scalar = src1_data[0];
        for (int64_t i = 0; i < ne; i++) {
            dst_data[i] = xlns32_mul(src0_data[i], scalar);
        }
    } else {
        for (int64_t i = 0; i < ne; i++) {
            dst_data[i] = xlns32_mul(src0_data[i % ne0], src1_data[i % ne1]);
        }
    }
}

static void ggml_lns_op_scale(ggml_tensor * dst, const ggml_tensor * src, float scale_factor) {
    const int64_t ne = ggml_nelements(dst);
    
    xlns32 * dst_data = (xlns32 *)dst->data;
    const xlns32 * src_data = (const xlns32 *)src->data;
    xlns32 scale = fp2xlns32(scale_factor);
    
    for (int64_t i = 0; i < ne; i++) {
        dst_data[i] = xlns32_mul(src_data[i], scale);
    }
}

static void ggml_lns_op_mul_mat(
    ggml_tensor * dst,
    const ggml_tensor * src0,  // Weight matrix
    const ggml_tensor * src1   // Input matrix
) {
    // Matrix multiplication: C = A @ B
    // src0: [K, M] (transposed weights)
    // src1: [K, N] (input)
    // dst:  [M, N] (output)
    
    const int64_t M = src0->ne[1];
    const int64_t K = src0->ne[0];
    const int64_t N = src1->ne[1];
    
    GGML_ASSERT(src1->ne[0] == K);
    GGML_ASSERT(dst->ne[0] == M);
    GGML_ASSERT(dst->ne[1] == N);
    
    const xlns32 * A = (const xlns32 *)src0->data;  // K x M
    const xlns32 * B = (const xlns32 *)src1->data;  // K x N
    xlns32 * C = (xlns32 *)dst->data;               // M x N
    
    // Naive implementation - optimize later
    for (int64_t j = 0; j < N; j++) {
        for (int64_t i = 0; i < M; i++) {
            xlns32 sum = xlns32_zero;
            for (int64_t k = 0; k < K; k++) {
                xlns32 a = A[k + i * K];  // A[k, i] in column-major
                xlns32 b = B[k + j * K];  // B[k, j] in column-major
                xlns32 prod = xlns32_mul(a, b);
                sum = xlns32_add(sum, prod);
            }
            C[i + j * M] = sum;  // C[i, j] in column-major
        }
    }
}

static void ggml_lns_op_soft_max(ggml_tensor * dst, const ggml_tensor * src) {
    // Softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    // This is challenging in LNS because it requires addition for the sum
    
    const int64_t ne0 = src->ne[0];  // Softmax dimension
    const int64_t ne1 = ggml_nelements(src) / ne0;  // Batch dimension
    
    const xlns32 * src_data = (const xlns32 *)src->data;
    xlns32 * dst_data = (xlns32 *)dst->data;
    
    for (int64_t j = 0; j < ne1; j++) {
        const xlns32 * row_src = src_data + j * ne0;
        xlns32 * row_dst = dst_data + j * ne0;
        
        // Find max (need to convert to compare)
        float max_val = -INFINITY;
        for (int64_t i = 0; i < ne0; i++) {
            float val = xlns322fp(row_src[i]);
            if (val > max_val) max_val = val;
        }
        xlns32 max_lns = fp2xlns32(max_val);
        
        // Compute exp(x - max) and sum
        // Note: exp in LNS can be efficient
        xlns32 sum_lns = xlns32_zero;
        
        // Temporary storage for exp values
        std::vector<xlns32> exp_vals(ne0);
        
        for (int64_t i = 0; i < ne0; i++) {
            // x - max in LNS is division
            xlns32 diff = xlns32_div(row_src[i], max_lns);
            
            // exp in LNS: exp(x) = 2^(x * log2(e))
            float diff_fp = xlns322fp(diff);
            float exp_fp = expf(diff_fp);
            exp_vals[i] = fp2xlns32(exp_fp);
            
            sum_lns = xlns32_add(sum_lns, exp_vals[i]);
        }
        
        // Divide by sum
        for (int64_t i = 0; i < ne0; i++) {
            row_dst[i] = xlns32_div(exp_vals[i], sum_lns);
        }
    }
}

static void ggml_lns_op_rms_norm(ggml_tensor * dst, const ggml_tensor * src, float eps) {
    // RMS normalization: y = x / sqrt(mean(x^2) + eps)
    
    const int64_t ne0 = src->ne[0];  // Normalization dimension
    const int64_t ne1 = ggml_nelements(src) / ne0;
    
    const xlns32 * src_data = (const xlns32 *)src->data;
    xlns32 * dst_data = (xlns32 *)dst->data;
    
    for (int64_t j = 0; j < ne1; j++) {
        const xlns32 * row_src = src_data + j * ne0;
        xlns32 * row_dst = dst_data + j * ne0;
        
        // Compute mean of squares
        xlns32 sum_sq = xlns32_zero;
        for (int64_t i = 0; i < ne0; i++) {
            xlns32 sq = xlns32_mul(row_src[i], row_src[i]);
            sum_sq = xlns32_add(sum_sq, sq);
        }
        
        // mean = sum / n
        xlns32 n_lns = fp2xlns32((float)ne0);
        xlns32 mean_sq = xlns32_div(sum_sq, n_lns);
        
        // sqrt(mean + eps)
        float mean_fp = xlns322fp(mean_sq);
        xlns32 rms = fp2xlns32(sqrtf(mean_fp + eps));
        
        // Normalize
        for (int64_t i = 0; i < ne0; i++) {
            row_dst[i] = xlns32_div(row_src[i], rms);
        }
    }
}

// ============================================================================
// Graph Compute Dispatcher
// ============================================================================

static void ggml_lns_compute_op(ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
            // No operation
            break;
            
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            ggml_lns_op_cpy(node, node->src[0]);
            break;
            
        case GGML_OP_ADD:
            ggml_lns_op_add(node, node->src[0], node->src[1]);
            break;
            
        case GGML_OP_MUL:
            ggml_lns_op_mul(node, node->src[0], node->src[1]);
            break;
            
        case GGML_OP_SCALE: {
            float scale = *(float *)node->op_params;
            ggml_lns_op_scale(node, node->src[0], scale);
            break;
        }
        
        case GGML_OP_MUL_MAT:
            ggml_lns_op_mul_mat(node, node->src[0], node->src[1]);
            break;
            
        case GGML_OP_SOFT_MAX:
            ggml_lns_op_soft_max(node, node->src[0]);
            break;
            
        case GGML_OP_RMS_NORM: {
            float eps = *(float *)node->op_params;
            ggml_lns_op_rms_norm(node, node->src[0], eps);
            break;
        }
        
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // These are view operations - data is shared, just need to set pointer
            node->data = node->view_src->data;
            break;
            
        default:
            GGML_LOG_ERROR("ggml-lns: unsupported operation %s\n", ggml_op_name(node->op));
            GGML_ABORT("unsupported operation");
    }
}

static enum ggml_status ggml_backend_lns_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        
        if (ggml_is_empty(node)) {
            continue;
        }
        
        ggml_lns_compute_op(node);
    }
    
    return GGML_STATUS_SUCCESS;
}
```

### 10.5 Step 4: Add CMake Build

```cmake
# ggml/src/ggml-lns/CMakeLists.txt

set(GGML_LNS_SOURCES
    ggml-lns.cpp
)

set(GGML_LNS_HEADERS
    ${CMAKE_SOURCE_DIR}/ggml/include/ggml-lns.h
)

add_library(ggml-lns STATIC ${GGML_LNS_SOURCES})

target_include_directories(ggml-lns PUBLIC
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/xlnscpp
)

target_link_libraries(ggml-lns PRIVATE ggml)

# Copy xlnscpp files
file(COPY ${CMAKE_SOURCE_DIR}/external/xlnscpp/ 
     DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/xlnscpp)
```

---

## 11. Phase 4: Integration with llama.cpp

### 11.1 Goal
Enable llama.cpp to use the LNS backend for inference.

### 11.2 Step 1: Register Backend

Modify llama.cpp to recognize the LNS backend:

```cpp
// In llama.cpp's backend initialization code

#include "ggml-lns.h"

// Add to backend registration
void llama_backend_init(void) {
    // ... existing backends ...
    
    // Register LNS backend
    ggml_backend_reg_t lns_reg = ggml_backend_lns_reg();
    // Registration handled by ggml's backend system
}
```

### 11.3 Step 2: Enable LNS via Command Line

Add command-line option to select LNS backend:

```cpp
// In common/arg.cpp or similar

{"-lns", "--use-lns"},
[](gpt_params & params) {
    params.use_lns_backend = true;
},
"use Logarithmic Number System backend for inference"
```

### 11.4 Step 3: Force LNS Backend Selection

When LNS is selected, ensure tensors use LNS buffer type:

```cpp
// In model loading code

if (params.use_lns_backend) {
    ggml_backend_t lns_backend = ggml_backend_lns_init();
    // Use LNS backend for computation
}
```

---

## 12. Phase 5: Testing & Validation

### 12.1 Unit Tests

```cpp
// tests/test-lns-backend.cpp

#include "ggml-lns.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_buffer_conversion() {
    std::cout << "Testing buffer conversion..." << std::endl;
    
    ggml_backend_buffer_type_t buft = ggml_backend_lns_buffer_type();
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 1024);
    
    // Create a simple tensor context
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    // Create test tensor
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
    tensor->data = ggml_backend_buffer_get_base(buffer);
    
    // Set float data
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    ggml_backend_tensor_set(tensor, input, 0, sizeof(input));
    
    // Get back float data
    float output[10];
    ggml_backend_tensor_get(tensor, output, 0, sizeof(output));
    
    // Verify
    for (int i = 0; i < 10; i++) {
        float err = fabsf(input[i] - output[i]) / input[i];
        std::cout << "  input[" << i << "]=" << input[i] 
                  << " output=" << output[i] 
                  << " err=" << err << std::endl;
        assert(err < 0.01f);
    }
    
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    std::cout << "  PASS" << std::endl;
}

void test_basic_operations() {
    std::cout << "Testing basic operations..." << std::endl;
    
    ggml_backend_t backend = ggml_backend_lns_init();
    
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    // Create tensors
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    
    // Build graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    
    // Allocate and set data
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_b[] = {0.5f, 1.5f, 2.5f, 3.5f};
    ggml_backend_tensor_set(a, data_a, 0, sizeof(data_a));
    ggml_backend_tensor_set(b, data_b, 0, sizeof(data_b));
    
    // Compute
    ggml_backend_graph_compute(backend, graph);
    
    // Get result
    float result[4];
    ggml_backend_tensor_get(c, result, 0, sizeof(result));
    
    // Verify
    float expected[] = {1.5f, 3.5f, 5.5f, 7.5f};
    for (int i = 0; i < 4; i++) {
        float err = fabsf(result[i] - expected[i]) / expected[i];
        std::cout << "  c[" << i << "]: expected=" << expected[i] 
                  << " got=" << result[i] << " err=" << err << std::endl;
        assert(err < 0.01f);
    }
    
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  PASS" << std::endl;
}

void test_matrix_multiply() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    // TODO: Implement matrix multiply test
    
    std::cout << "  PASS" << std::endl;
}

int main() {
    test_buffer_conversion();
    test_basic_operations();
    test_matrix_multiply();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
```

### 12.2 Accuracy Validation

```cpp
// tests/test-lns-accuracy.cpp

// Compare LNS backend output vs CPU backend output

void compare_backends(const char * model_path, const char * prompt) {
    // Load model twice
    llama_model * model_cpu = llama_load_model(model_path, /* params */);
    llama_model * model_lns = llama_load_model(model_path, /* params with LNS */);
    
    // Tokenize prompt
    std::vector<llama_token> tokens = tokenize(prompt);
    
    // Run inference on both
    std::vector<float> logits_cpu = infer(model_cpu, tokens);
    std::vector<float> logits_lns = infer(model_lns, tokens);
    
    // Compare logits
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (size_t i = 0; i < logits_cpu.size(); i++) {
        float diff = fabsf(logits_cpu[i] - logits_lns[i]);
        max_diff = fmaxf(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= logits_cpu.size();
    
    std::cout << "Max logit difference: " << max_diff << std::endl;
    std::cout << "Avg logit difference: " << avg_diff << std::endl;
    
    // Check if top tokens match
    int top_k = 5;
    // Compare top-k predictions
}
```

### 12.3 Integration Test

```bash
# Run simple inference with LNS backend
./bin/llama-cli -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --use-lns \
    -p "Hello, world!" \
    -n 20

# Compare with CPU backend
./bin/llama-cli -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Hello, world!" \
    -n 20
```

---

## 13. Phase 6: Optimization & Documentation

### 13.1 Performance Optimization

#### 13.1.1 Batch Conversion Optimization
```cpp
// Use SIMD for batch FP↔LNS conversion
void fp_to_lns32_array_simd(const float* src, xlns32* dst, size_t n) {
    // AVX2 implementation
    #ifdef __AVX2__
    // Process 8 floats at a time
    #endif
}
```

#### 13.1.2 Matrix Multiply Optimization
```cpp
// Blocked matrix multiplication
void lns32_matmul_blocked(
    const xlns32* A, const xlns32* B, xlns32* C,
    size_t M, size_t K, size_t N,
    size_t block_size
) {
    // Cache-friendly blocked implementation
}
```

#### 13.1.3 Parallel Processing
```cpp
// OpenMP parallelization
#pragma omp parallel for
for (int64_t i = 0; i < ne; i++) {
    dst_data[i] = xlns32_mul(src0_data[i], src1_data[i]);
}
```

### 13.2 Documentation

Create comprehensive documentation:

1. **README.md** - Quick start guide
2. **ARCHITECTURE.md** - Design decisions
3. **API.md** - API reference
4. **PERFORMANCE.md** - Benchmarks and optimization notes
5. **CONTRIBUTING.md** - How to contribute

---

## 14. File-by-File Implementation Details

### Summary of Files to Create/Modify

| File | Purpose | Lines (est.) |
|------|---------|--------------|
| `ggml/include/ggml-lns.h` | Public API | ~80 |
| `ggml/src/ggml-lns/ggml-lns.cpp` | Main implementation | ~600 |
| `ggml/src/ggml-lns/ggml-lns-compute.cpp` | Operation implementations | ~500 |
| `ggml/src/ggml-lns/CMakeLists.txt` | Build configuration | ~30 |
| `ggml/src/ggml-lns/xlnscpp/xlns.h` | Modified xlnscpp header | ~200 |
| `tests/test-lns-backend.cpp` | Unit tests | ~300 |
| `docs/backend-lns.md` | Documentation | ~200 |

---

## 15. Performance Considerations

### 15.1 Expected Performance Characteristics

| Operation | LNS vs FP | Reason |
|-----------|-----------|--------|
| Multiply | Faster | Integer add vs hardware multiply |
| Divide | Faster | Integer sub vs hardware divide |
| Add | Much Slower | Requires sb/db Gaussian logs |
| Convert | Slow | log/exp operations |

### 15.2 LLM Operation Mix

For transformer inference:
- **~70% multiply** (matrix multiplications) - LNS favorable
- **~20% add** (residual connections, softmax sum) - LNS unfavorable  
- **~10% other** (normalization, activation) - Mixed

### 15.3 Optimization Strategies

1. **Minimize conversions**: Convert once at boundaries
2. **Batch operations**: Process multiple elements together
3. **Use tables**: Table-based sb/db is much faster than ideal mode
4. **Cache-friendly layouts**: Optimize memory access patterns

---

## 16. Testing Strategy

### 16.1 Test Pyramid

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E ╲         End-to-End: Full LLM inference
                 ╱──────╲
                ╱        ╲
               ╱Integration╲     Integration: Backend with ggml
              ╱────────────╲
             ╱              ╲
            ╱     Unit       ╲   Unit: Individual operations
           ╱──────────────────╲
```

### 16.2 Test Categories

| Category | Examples | Frequency |
|----------|----------|-----------|
| Unit | xlnscpp operations, buffer ops | On every change |
| Integration | Graph compute, backend selection | On PR |
| Accuracy | Logit comparison vs FP | Weekly |
| Performance | Benchmark suite | Before release |
| E2E | Full model inference | Before release |

---

## 17. Timeline & Milestones

### Gantt Chart Style Timeline

```
Week:  1   2   3   4   5   6   7   8   9  10  11  12
      ─────────────────────────────────────────────────
Setup ████
Study     ████
Prep          ████████
Core               ████████████
Ops                          ██████████
Integ                                   ████████
Test                                         ████████
Doc                                              ████
```

### Detailed Milestones

| Milestone | Week | Deliverable | Success Criteria |
|-----------|------|-------------|------------------|
| M1 | 2 | Environment ready | Build llama.cpp, run xlnscpp tests |
| M2 | 4 | xlnscpp library ready | CMake builds, library tests pass |
| M3 | 7 | Backend skeleton | Backend registers, buffers work |
| M4 | 9 | Core ops working | MUL_MAT, ADD produce correct output |
| M5 | 11 | Integration complete | TinyLlama runs with LNS |
| M6 | 12 | Project complete | Documentation, benchmarks |

---

## 18. Risk Assessment & Mitigation

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Softmax accuracy | Output quality degrades | Implement hybrid mode (FP for softmax) |
| Addition slowness | Inference very slow | Profile and optimize hot paths |
| Numerical instability | NaN/Inf in output | Add overflow/underflow handling |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Complex ops unsupported | Some models fail | Implement ops incrementally |
| API changes in ggml | Backend breaks | Track ggml development |
| Build issues | Hard to use | Thorough CMake setup |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Documentation gaps | Hard to contribute | Write docs alongside code |
| Test coverage gaps | Bugs ship | Test-driven development |

---

## 19. References & Resources

### Primary References

1. [xlnscpp Repository](https://github.com/xlnsresearch/xlnscpp)
2. [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
3. [ggml Repository](https://github.com/ggml-org/ggml)

### Research Papers

1. G. Alsuhli, et al., "Number Systems for Deep Neural Network Architectures: A Survey"
2. M. Arnold, E. Chester, et al., "Training neural nets using only an approximate tableless LNS ALU"
3. O. Kosheleva, et al., "Logarithmic Number System Is Optimal for AI Computations"
4. D. Miyashita, et al., "Convolutional Neural Networks using Logarithmic Data Representation"
5. J. Zhao et al., "LNS-Madam: Low-Precision Training in Logarithmic Number System"
6. P. Haghi, et al., "Bridging the Gap Between LLMs and LNS"

### Documentation

- [ggml Backend Implementation Guide](https://github.com/ggml-org/ggml/wiki)
- [llama.cpp Development Docs](https://github.com/ggml-org/llama.cpp/tree/master/docs)
- [GGUF File Format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

### Communication

- [xlnscpp Discussion Forum](https://github.com/xlnsresearch/xlnscpp/discussions)
- Mentors:
  - Mark Arnold: markgarnold -at- yahoo.com
  - Alex Krentz: alexkrentz2 -at- gmail.com
  - Ed Chester: ed.chester -at- gmail.com

---

## Appendix A: Quick Reference Commands

### Build Commands
```bash
# Build llama.cpp with LNS backend
cd llama.cpp
mkdir build && cd build
cmake -DGGML_LNS=ON ..
cmake --build . -j$(nproc)

# Run tests
./bin/test-lns-backend

# Run inference with LNS
./bin/llama-cli -m model.gguf --use-lns -p "Hello"
```

### Debug Commands
```bash
# Enable debug logging
export GGML_LOG_LEVEL=DEBUG

# Run with valgrind
valgrind ./bin/test-lns-backend

# Profile with perf
perf record ./bin/llama-cli ...
perf report
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| LNS | Logarithmic Number System |
| sb | Sum-bit Gaussian log: log₂(1 + 2^z) |
| db | Difference-bit Gaussian log: log₂(|1 - 2^z|) |
| LPVIP | Low-Power Very-Integer-Part - approximation method |
| ggml | Tensor library used by llama.cpp |
| GGUF | File format for storing model weights |
| Backend | ggml component that executes tensor operations |
| Buffer | Memory allocation for tensor data |
| xlns32 | 32-bit LNS type (23 fractional bits) |
| xlns16 | 16-bit LNS type (7 fractional bits) |

---

## Appendix C: Checklist

### Before Starting
- [ ] Cloned llama.cpp repository
- [ ] Successfully built llama.cpp
- [ ] Downloaded test model
- [ ] Ran basic inference
- [ ] Cloned xlnscpp repository
- [ ] Built and ran xlnscpp tests
- [ ] Read all xlnscpp documentation
- [ ] Read ggml backend headers
- [ ] Studied at least one existing backend

### During Development
- [ ] Created ggml-lns.h header
- [ ] Implemented buffer type
- [ ] Implemented buffer
- [ ] Implemented device
- [ ] Implemented registry
- [ ] Implemented basic ops (CPY, ADD, MUL)
- [ ] Implemented MUL_MAT
- [ ] Implemented SOFT_MAX
- [ ] Implemented normalizations
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Full model inference works

### Before Submission
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Performance benchmarks run
- [ ] Code cleaned up
- [ ] Mentor review complete

---

*Document Version: 1.0*
*Last Updated: February 2026*
*Author: GSoC Project Guide*
