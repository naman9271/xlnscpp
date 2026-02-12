# Project Plan: Support for Logarithmic Number Systems in Large Language Models

## Integrating xlnscpp into ggml / llama.cpp

---

## Table of Contents

1. [The Big Picture — Yes, You're Right](#1-the-big-picture--yes-youre-right)
2. [How to Use a Library From One Repo in Another](#2-how-to-use-a-library-from-one-repo-in-another)
3. [Final File Structure of the Entire Project](#3-final-file-structure-of-the-entire-project)
4. [Step-by-Step Checklist — What to Do and When](#4-step-by-step-checklist--what-to-do-and-when)
5. [Architecture Overview](#5-architecture-overview)
6. [Prerequisites & Environment Setup](#6-prerequisites--environment-setup)
7. [Phase 1 — Understand the Codebase](#7-phase-1--understand-the-codebase)
8. [Phase 2 — Build xlnscpp as an Installable CMake Library](#8-phase-2--build-xlnscpp-as-an-installable-cmake-library)
9. [Phase 3 — Create the ggml-lns Backend Skeleton](#9-phase-3--create-the-ggml-lns-backend-skeleton)
10. [Phase 4 — Implement Tensor Operations in LNS](#10-phase-4--implement-tensor-operations-in-lns)
11. [Phase 5 — Register the Backend & Build Integration](#11-phase-5--register-the-backend--build-integration)
12. [Phase 6 — End-to-End Testing with llama.cpp](#12-phase-6--end-to-end-testing-with-llamacpp)
13. [Phase 7 — Performance Optimization](#13-phase-7--performance-optimization)
14. [Phase 8 — Validation with a Real LLM (DeepSeek)](#14-phase-8--validation-with-a-real-llm-deepseek)
15. [File-by-File Change Map](#15-file-by-file-change-map)
16. [Key Design Decisions](#16-key-design-decisions)
17. [Risk Register](#17-risk-register)
18. [Milestones & Timeline](#18-milestones--timeline)

---

## 1. The Big Picture — Yes, You're Right

Your understanding is correct. The project is **three big steps** done in **two repos**:

```
┌──────────────────────────────────────────────────────────────────┐
│                      YOUR WORKFLOW                               │
│                                                                  │
│  STEP 1 ─ xlnscpp repo                                          │
│  ┌──────────────────────────────────────────────┐                │
│  │ Turn xlnscpp into a proper CMake library     │                │
│  │ that can be INSTALLED on your system.         │                │
│  │                                               │                │
│  │   cmake -B build                              │                │
│  │   cmake --build build                         │                │
│  │   sudo cmake --install build                  │  ← installs   │
│  │                                               │    libxlns.a   │
│  │  Result: /usr/local/lib/libxlns.a             │    and headers │
│  │          /usr/local/include/xlns/xlns32.h     │    to system   │
│  └──────────────────────────────────────────────┘                │
│                         │                                        │
│                         ▼                                        │
│  STEP 2 ─ llama.cpp repo (your fork)                             │
│  ┌──────────────────────────────────────────────┐                │
│  │ Create ggml-lns backend that does:           │                │
│  │   find_package(xlns)  ← finds your library   │                │
│  │   calls xlns functions for LNS arithmetic    │                │
│  │                                               │                │
│  │  Result: ggml-lns backend compiles into       │                │
│  │          llama.cpp as a new backend           │                │
│  └──────────────────────────────────────────────┘                │
│                         │                                        │
│                         ▼                                        │
│  STEP 3 ─ same llama.cpp repo                                    │
│  ┌──────────────────────────────────────────────┐                │
│  │ Test with:                                    │                │
│  │   ./build/bin/llama-cli --list-devices        │  ← see "LNS"  │
│  │   ./build/bin/test-backend-ops -b LNS         │  ← op tests   │
│  │   ./build/bin/llama-cli -m model.gguf ...     │  ← run a model │
│  └──────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────┘
```

### Which Repos and What You Do in Each

| Repo | URL | What You Do |
|------|-----|-------------|
| **xlnscpp** (this repo) | `github.com/xlnsresearch/xlnscpp` | Add `CMakeLists.txt`, `include/xlns/`, `src/`, `tests/`. Build & install as `libxlns`. |
| **llama.cpp** (fork it) | `github.com/ggml-org/llama.cpp` | Add `ggml/src/ggml-lns/` backend, edit `ggml/CMakeLists.txt` + `ggml/src/ggml-backend-reg.cpp`. Test. |
| **ggml** (standalone) | `github.com/ggml-org/ggml` | **DO NOT touch.** It's a mirror synced from llama.cpp. |

> **Key fact:** ggml lives INSIDE llama.cpp at `llama.cpp/ggml/`. You don't clone ggml separately.

---

## 2. How to Use a Library From One Repo in Another

This is the standard CMake workflow. Here's exactly how it works:

### The Problem
xlnscpp is in `/home/naman/Desktop/xlnscpp`. The ggml-lns backend is in `llama.cpp/ggml/src/ggml-lns/`. How does one find the other?

### The Solution: CMake `find_package`

**In xlnscpp:** You create a `CMakeLists.txt` that builds a library and generates an "export" file (a `.cmake` config). When you run `cmake --install`, it copies:
- `libxlns.a` → `/usr/local/lib/`
- header files → `/usr/local/include/xlns/`
- `xlnsConfig.cmake` → `/usr/local/lib/cmake/xlns/`

**In llama.cpp/ggml-lns:** You write:
```cmake
find_package(xlns REQUIRED)                        # Finds /usr/local/lib/cmake/xlns/xlnsConfig.cmake
target_link_libraries(ggml-lns PRIVATE xlns::xlns)  # Links libxlns.a and adds include paths
```

That's it. CMake handles everything — include paths, linking, dependencies.

### Visual Explanation

```
 xlnscpp repo                              llama.cpp repo
 ───────────                               ──────────────
                                           ggml/src/ggml-lns/CMakeLists.txt:
 CMakeLists.txt:                           ┌────────────────────────────────┐
 ┌──────────────────────┐                  │ find_package(xlns REQUIRED)    │
 │ add_library(xlns     │   cmake install  │                                │
 │   STATIC             │ ═══════════════> │ # Now ggml-lns can do:         │
 │   src/xlns32.cpp     │   copies to      │ #include <xlns/xlns32.h>       │
 │   src/xlns16.cpp)    │   /usr/local/    │ fp2xlns32(3.14f);              │
 │                      │                  │                                │
 │ install(TARGETS xlns │                  │ target_link_libraries(         │
 │   EXPORT xlnsTargets)│                  │   ggml-lns PRIVATE xlns::xlns) │
 └──────────────────────┘                  └────────────────────────────────┘
```

### Alternative: Custom Install Prefix (No sudo needed)

If you don't want to install system-wide:
```bash
# In xlnscpp:
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build
cmake --install build    # No sudo needed!

# In llama.cpp:
cmake -B build -DGGML_LNS=ON -DCMAKE_PREFIX_PATH=$HOME/.local
```

### Other Approaches (and why find_package is best)

| Approach | How | Verdict |
|----------|-----|---------|
| **A. Copy source files** | `cp xlns32.cpp` into ggml-lns/ | Quick-and-dirty. Duplicates code. Hard to update. |
| **B. Git submodule** | `git submodule add <url>` | Decent. But adds complexity for contributors. |
| **C. find_package** ✅ | Install as library, `find_package(xlns)` | **Clean, professional, standard CMake practice.** |

---

## 3. Final File Structure of the Entire Project

### Your Workspace
```
~/lns-llm-project/
├── xlnscpp/                          ← REPO 1 (this repo — you modify it)
│   └── (see below)
└── llama.cpp/                        ← REPO 2 (your fork — you modify it)
    └── (see below)
```

### REPO 1: xlnscpp — Final Structure

Files marked ★ are **new files you create**. Everything else already exists.

```
xlnscpp/
│
├── CMakeLists.txt                     ★ NEW — root build file
│
├── include/                           ★ NEW — public headers (get installed)
│   └── xlns/
│       ├── xlns32.h                   ★ NEW — clean 32-bit API header
│       ├── xlns16.h                   ★ NEW — clean 16-bit API header
│       └── xlns.h                     ★ NEW — convenience header (#includes both)
│
├── src/                               ★ NEW — library source files
│   ├── CMakeLists.txt                 ★ NEW — library build rules
│   ├── xlns32.cpp                     ★ NEW — 32-bit implementation (refactored from root)
│   ├── xlns16.cpp                     ★ NEW — 16-bit implementation (refactored from root)
│   ├── xlns32tbl.h                    → copied/moved from root
│   ├── xlns16sbdbtbl.h               → copied/moved from root
│   ├── xlns16cvtbl.h                 → copied/moved from root
│   └── xlns16revcvtbl.h              → copied/moved from root
│
├── tests/                             ★ NEW — test suite
│   ├── CMakeLists.txt                 ★ NEW
│   ├── test_xlns32.cpp                ★ NEW — 32-bit arithmetic tests
│   ├── test_xlns16.cpp                ★ NEW — 16-bit arithmetic tests
│   └── test_matmul.cpp                ★ NEW — matrix multiply correctness test
│
├── cmake/                             ★ NEW — CMake package config
│   └── xlnsConfig.cmake.in            ★ NEW — template for find_package support
│
│── ── ── ── EXISTING FILES (unchanged) ── ── ── ──
│
├── xlns32.cpp                         (original — kept for backwards compat)
├── xlns16.cpp                         (original — kept for backwards compat)
├── xlns32tbl.h                        (original)
├── xlns16sbdbtbl.h                    (original)
├── xlns16cvtbl.h                      (original)
├── xlns16revcvtbl.h                   (original)
├── xlns32test.cpp                     (original tests)
├── xlns16test.cpp                     (original tests)
├── xlns32funtest.cpp                  (original tests)
├── xlns16funtest.cpp                  (original tests)
├── xlnsbothtest.cpp                   (original tests)
├── *.py                               (original python tests)
├── docs/                              (documentation)
├── project/                           (this plan)
├── LICENSE
└── README.md
```

### What Gets Installed (after `cmake --install`)

```
$HOME/.local/                          (or /usr/local/)
├── lib/
│   ├── libxlns.a                      ← the static library
│   └── cmake/
│       └── xlns/
│           ├── xlnsConfig.cmake        ← lets find_package(xlns) work
│           └── xlnsTargets.cmake       ← import targets
└── include/
    └── xlns/
        ├── xlns32.h                    ← public header
        ├── xlns16.h                    ← public header
        └── xlns.h                      ← convenience header
```

### REPO 2: llama.cpp (your fork) — Only Changed/New Files

You only touch **5 items** in the entire llama.cpp repo (3 new files, 2 edited files):

```
llama.cpp/
├── ggml/
│   ├── CMakeLists.txt                 ✏️  EDIT — add 2 lines (GGML_LNS option + add_subdirectory)
│   │
│   ├── include/
│   │   ├── ggml.h                     (untouched)
│   │   ├── ggml-backend.h             (untouched)
│   │   ├── ggml-blas.h                (untouched, study this for reference)
│   │   └── ggml-lns.h                 ★ NEW — public backend header (~25 lines)
│   │
│   └── src/
│       ├── ggml-backend-reg.cpp       ✏️  EDIT — add 4 lines (#include + register_backend)
│       │
│       ├── ggml-blas/                 (untouched, study this as your reference)
│       │   ├── CMakeLists.txt
│       │   └── ggml-blas.cpp
│       │
│       └── ggml-lns/                  ★ NEW DIRECTORY — your backend
│           ├── CMakeLists.txt         ★ NEW — build rules + find_package(xlns) (~15 lines)
│           └── ggml-lns.cpp           ★ NEW — backend implementation (~800-1500 lines)
│
├── src/                               (untouched — llama.cpp source)
├── tests/                             (untouched — can run test-backend-ops)
├── CMakeLists.txt                     (untouched — top-level)
└── ...                                (everything else untouched)
```

### Summary: Total Files You Write

| Repo | New Files | Edited Files | Total |
|------|-----------|-------------|-------|
| xlnscpp | ~10 files (CMake, headers, src, tests) | 0 | 10 |
| llama.cpp | 3 files (ggml-lns.h, ggml-lns.cpp, CMakeLists.txt) | 2 files (ggml/CMakeLists.txt, ggml-backend-reg.cpp) | 5 |
| **Total** | **13** | **2** | **15** |

---

## 4. Step-by-Step Checklist — What to Do and When

Here is the **exact order** of tasks. Do them in sequence. Each step has a ✅ checkpoint.

### STAGE A: Setup (Day 1)

- [ ] **A1.** Install build tools: `sudo apt install build-essential cmake git`
- [ ] **A2.** Fork `ggml-org/llama.cpp` on GitHub
- [ ] **A3.** Clone both repos side by side:
  ```bash
  mkdir ~/lns-llm-project && cd ~/lns-llm-project
  git clone https://github.com/<YOU>/llama.cpp.git
  cp -r /home/naman/Desktop/xlnscpp ./xlnscpp  # or symlink/clone
  ```
- [ ] **A4.** Verify llama.cpp builds from scratch:
  ```bash
  cd llama.cpp && cmake -B build && cmake --build build -j$(nproc)
  ```
- [ ] **A5.** Verify xlnscpp existing tests work:
  ```bash
  cd ../xlnscpp && g++ -O2 -o xlns32test xlns32test.cpp && echo "" | ./xlns32test
  ```
- ✅ **Checkpoint:** Both repos build. You have a working baseline.

### STAGE B: Study (Week 1-2)

- [ ] **B1.** Read `llama.cpp/ggml/src/ggml-backend-impl.h` — understand the 5 interface structs
- [ ] **B2.** Read `llama.cpp/ggml/src/ggml-blas/ggml-blas.cpp` end-to-end (~520 lines) — this is your template
- [ ] **B3.** Read `xlns32.cpp` end-to-end (644 lines) — understand every function
- [ ] **B4.** Read `docs/api-reference.md` and `docs/architecture.md`
- [ ] **B5.** Try writing a small program that uses xlns32 to multiply two matrices
- ✅ **Checkpoint:** You can explain how `graph_compute` works and how `fp2xlns32` works.

### STAGE C: Build xlnscpp as a Library (Week 3)

- [ ] **C1.** Create `xlnscpp/include/xlns/xlns32.h` — extract function declarations from xlns32.cpp
- [ ] **C2.** Create `xlnscpp/src/xlns32.cpp` — the implementation (refactored to use the new header)
- [ ] **C3.** Create `xlnscpp/CMakeLists.txt` — builds `libxlns.a` static library
- [ ] **C4.** Create `xlnscpp/cmake/xlnsConfig.cmake.in` — enables `find_package(xlns)`
- [ ] **C5.** Create `xlnscpp/tests/test_xlns32.cpp` — basic correctness tests
- [ ] **C6.** Build, test, and install:
  ```bash
  cd ~/lns-llm-project/xlnscpp
  cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
  cmake --build build
  ctest --test-dir build          # run tests
  cmake --install build           # install to ~/.local/
  ```
- [ ] **C7.** Verify install worked:
  ```bash
  ls ~/.local/lib/libxlns.a              # library exists
  ls ~/.local/include/xlns/xlns32.h      # header exists
  ls ~/.local/lib/cmake/xlns/            # cmake config exists
  ```
- ✅ **Checkpoint:** `libxlns.a` is installed. Any CMake project can now `find_package(xlns)`.

### STAGE D: Create the ggml-lns Backend Skeleton (Week 4-5)

- [ ] **D1.** Create `llama.cpp/ggml/include/ggml-lns.h` (~25 lines)
- [ ] **D2.** Create `llama.cpp/ggml/src/ggml-lns/CMakeLists.txt` — with `find_package(xlns)`
- [ ] **D3.** Create `llama.cpp/ggml/src/ggml-lns/ggml-lns.cpp` — start with EMPTY `graph_compute`:
  - Implement all boilerplate: registry, device, backend, GUID
  - `graph_compute` just returns `GGML_STATUS_SUCCESS` (does nothing yet)
  - `supports_op` returns `false` for everything (backend is a no-op)
- [ ] **D4.** Edit `llama.cpp/ggml/CMakeLists.txt` — add `GGML_LNS` option + `add_subdirectory`
- [ ] **D5.** Edit `llama.cpp/ggml/src/ggml-backend-reg.cpp` — register LNS backend
- [ ] **D6.** Build llama.cpp with LNS:
  ```bash
  cd ~/lns-llm-project/llama.cpp
  cmake -B build -DGGML_LNS=ON -DCMAKE_PREFIX_PATH=$HOME/.local
  cmake --build build -j$(nproc)
  ```
- [ ] **D7.** Verify backend appears:
  ```bash
  ./build/bin/llama-cli --list-devices   # Should show "LNS" device
  ```
- ✅ **Checkpoint:** LNS backend compiles, links to xlnscpp, and shows up as a device. It doesn't DO anything yet.

### STAGE E: Implement MUL_MAT (Week 6-7)

- [ ] **E1.** In `ggml-lns.cpp`, implement `ggml_lns_op_mul_mat()` — convert to LNS, dot products, convert back
- [ ] **E2.** In `supports_op()`, return `true` for `GGML_OP_MUL_MAT` when both inputs are `GGML_TYPE_F32`
- [ ] **E3.** In `graph_compute()`, add the `GGML_OP_MUL_MAT` case
- [ ] **E4.** Test: `./build/bin/test-backend-ops -b LNS -o MUL_MAT`
- [ ] **E5.** Compare numerical output with CPU backend (should be close but not identical)
- ✅ **Checkpoint:** MUL_MAT produces correct results. This is the hardest operation.

### STAGE F: Add More Operations (Week 8-9)

- [ ] **F1.** Implement `GGML_OP_ADD` — element-wise addition in LNS
- [ ] **F2.** Implement `GGML_OP_MUL` — element-wise multiplication in LNS
- [ ] **F3.** Implement `GGML_OP_SCALE` — scalar multiplication
- [ ] **F4.** Implement `GGML_OP_RMS_NORM` — normalization
- [ ] **F5.** Implement `GGML_OP_SOFT_MAX` — softmax (hybrid: use float for exp())
- [ ] **F6.** Implement `GGML_OP_SILU` — activation function
- [ ] **F7.** Implement `GGML_OP_ROPE` — rotary position embedding (hybrid)
- [ ] **F8.** Test each: `./build/bin/test-backend-ops -b LNS`
- ✅ **Checkpoint:** All essential ops pass tests.

### STAGE G: Run a Real Model (Week 10-11)

- [ ] **G1.** Download a tiny model: `./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF`
- [ ] **G2.** Run with CPU (baseline): `./build/bin/llama-cli -m model.gguf -p "Hello" -n 20`
- [ ] **G3.** Run with LNS backend and check for coherent output
- [ ] **G4.** If model crashes, check which unsupported op was hit, implement it, repeat
- [ ] **G5.** Compare perplexity: CPU vs LNS
- ✅ **Checkpoint:** A real LLM produces coherent text using LNS arithmetic.

### STAGE H: Optimize & Validate with DeepSeek (Week 12-15)

- [ ] **H1.** Add LNS caching for weight tensors (convert once, reuse)
- [ ] **H2.** Try xlns16 for attention scores (lower precision, faster)
- [ ] **H3.** Run DeepSeek model and record output quality
- [ ] **H4.** Benchmark with `llama-bench`
- [ ] **H5.** Write final report comparing CPU vs LNS (perplexity, speed, output quality)
- ✅ **Checkpoint:** Project complete. DeepSeek runs on LNS.

### STAGE I: Documentation & Cleanup (Week 16)

- [ ] **I1.** Clean up code, add comments
- [ ] **I2.** Write README for the ggml-lns backend
- [ ] **I3.** Update xlnscpp README with CMake build instructions
- [ ] **I4.** Prepare for potential PR to ggml-org
- ✅ **Final Checkpoint:** Everything documented, tested, clean.

---

## 5. Architecture Overview

### Core Diagram
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  llama.cpp   │────>│   ggml       │────>│  ggml-lns       │
│  (unchanged) │     │  scheduler   │     │  backend        │
│              │     │  routes ops  │     │  (NEW - our     │
│              │     │  to backend  │     │   code)         │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │ calls
                                          ┌────────▼────────┐
                                          │    libxlns.a     │
                                          │  (xlns16/xlns32) │
                                          │  LNS arithmetic  │
                                          └─────────────────┘
```

### What This Is
- A **software emulation** of LNS arithmetic — a proof-of-concept, not hardware
- An **accelerator** backend (`GGML_BACKEND_DEVICE_TYPE_ACCEL`) that uses CPU host memory
- A **transparent plug-in** — llama.cpp doesn't know or care that LNS is being used

### What This Is NOT
- Not fast (software LNS is slower than native float)
- Not changing the model weight format
- Not modifying llama.cpp source code

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

---

## 6. Prerequisites & Environment Setup

### Step 6.1: Install Build Tools

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-pip

# Verify
g++ --version    # Need C++17 support (GCC 7+ or Clang 5+)
cmake --version  # Need 3.14+
```

### Step 6.2: Clone Repositories

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

### Step 6.3: Verify llama.cpp Builds Without Modifications

```bash
cd ~/lns-llm-project/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Test with a small model
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 20
```

### Step 6.4: Verify xlnscpp Works

```bash
cd ~/lns-llm-project/xlnscpp   # (or /home/naman/Desktop/xlnscpp)
g++ -O2 -o xlns32test xlns32test.cpp
echo "" | ./xlns32test
# Should see test output with matching FP and LNS results
```

---

## 7. Phase 1 — Understand the Codebase

**Time estimate:** 1–2 weeks
**Goal:** Understand how ggml backends work by studying existing ones

### Step 7.1: Study a Simple Backend (BLAS)

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

### Step 7.2: Study the Backend Implementation Interface

Read this file carefully:

```
llama.cpp/ggml/src/ggml-backend-impl.h
```

Key structs to understand:
- `ggml_backend_i` — The backend instance interface
- `ggml_backend_device_i` — The device interface
- `ggml_backend_reg_i` — The registry interface
- `ggml_backend_buffer_i` — The buffer interface
- `ggml_backend_buffer_type_i` — The buffer type interface

### Step 7.3: Study ggml Operations

Understand which operations LLMs actually use:

```
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

### Step 7.4: Study xlnscpp API

Read and understand:
- `xlns32.cpp` — The 32-bit implementation (644 lines)
- `xlns16.cpp` — The 16-bit implementation (609 lines)
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

## 8. Phase 2 — Build xlnscpp as an Installable CMake Library

**Time estimate:** 1 week
**Repo:** `xlnscpp`

Currently xlnscpp uses a header-include pattern (`#include "xlns32.cpp"`). For ggml integration, we need a proper library with:
- Separate header (`.h`) and implementation (`.cpp`) files
- A `CMakeLists.txt` that builds and installs the library
- A `xlnsConfig.cmake` so other CMake projects can `find_package(xlns)`

### Step 8.1: Create Public Headers

Create `xlnscpp/include/xlns/xlns32.h`:
```cpp
// xlns32.h — Public API for 32-bit Logarithmic Number System
#pragma once

#include <cstdint>
#include <cstddef>

// --- Core types ---
typedef unsigned int xlns32;
typedef signed int xlns32_signed;

// --- Constants ---
extern const xlns32 xlns32_zero;
extern const xlns32 xlns32_one;
extern const xlns32 xlns32_neg_one;

// --- Conversion ---
xlns32 fp2xlns32(float x);
float  xlns322fp(xlns32 x);

// --- Arithmetic ---
xlns32 xlns32_mul(xlns32 a, xlns32 b);
xlns32 xlns32_div(xlns32 a, xlns32 b);
xlns32 xlns32_add(xlns32 a, xlns32 b);
xlns32 xlns32_neg(xlns32 x);
xlns32 xlns32_abs(xlns32 x);
xlns32 xlns32_sqrt(xlns32 x);

// --- Batch operations (for tensor backends) ---
void xlns32_batch_from_float(const float * src, xlns32 * dst, size_t n);
void xlns32_batch_to_float(const xlns32 * src, float * dst, size_t n);
float xlns32_vec_dot(const float * a, const float * b, size_t n);
```

Create `xlnscpp/include/xlns/xlns.h` (convenience):
```cpp
#pragma once
#include "xlns32.h"
// #include "xlns16.h"  // uncomment when ready
```

### Step 8.2: Create Library Source

Create `xlnscpp/src/xlns32.cpp` — this wraps the existing code:
```cpp
// Refactored 32-bit LNS implementation
// This includes the original xlns32.cpp logic but exposes it
// through the clean header in include/xlns/xlns32.h

#define xlns32_alt   // Use streamlined addition
#include "../xlns32.cpp"  // Include original implementation

// Batch operations for tensor backends
void xlns32_batch_from_float(const float * src, xlns32 * dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp2xlns32(src[i]);
}

void xlns32_batch_to_float(const xlns32 * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = xlns322fp(src[i]);
}

float xlns32_vec_dot(const float * a, const float * b, size_t n) {
    xlns32 sum = xlns32_zero;
    for (size_t i = 0; i < n; i++) {
        sum = xlns32_add(sum, xlns32_mul(fp2xlns32(a[i]), fp2xlns32(b[i])));
    }
    return xlns322fp(sum);
}
```

### Step 8.3: Create Root CMakeLists.txt

Create `xlnscpp/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.14)
project(xlns VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Library ──
add_library(xlns STATIC
    src/xlns32.cpp
)

target_include_directories(xlns
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}   # For original xlns32.cpp and table headers
)

# ── Tests ──
option(XLNS_BUILD_TESTS "Build tests" ON)
if(XLNS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install ──
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xlns
    EXPORT xlnsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/xlns
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT xlnsTargets
    FILE xlnsTargets.cmake
    NAMESPACE xlns::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xlns
)

configure_package_config_file(
    cmake/xlnsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/xlnsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xlns
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xlnsConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xlns
)
```

### Step 8.4: Create CMake Config Template

Create `xlnscpp/cmake/xlnsConfig.cmake.in`:
```cmake
@PACKAGE_INIT@
include("${CMAKE_CURRENT_LIST_DIR}/xlnsTargets.cmake")
check_required_components(xlns)
```

### Step 8.5: Create Tests

Create `xlnscpp/tests/CMakeLists.txt`:
```cmake
add_executable(test_xlns32 test_xlns32.cpp)
target_link_libraries(test_xlns32 PRIVATE xlns)
add_test(NAME test_xlns32 COMMAND test_xlns32)
```

Create `xlnscpp/tests/test_xlns32.cpp`:
```cpp
#include <xlns/xlns32.h>
#include <cstdio>
#include <cmath>
#include <cassert>

bool approx(float a, float b, float tol = 0.01f) {
    return fabsf(a - b) < tol * (1.0f + fabsf(a));
}

int main() {
    // Test conversion round-trip
    float vals[] = {0.0f, 1.0f, -1.0f, 3.14f, -2.71f, 100.0f, 0.001f};
    for (float v : vals) {
        float rt = xlns322fp(fp2xlns32(v));
        printf("roundtrip: %.4f -> %.4f\n", v, rt);
        assert(approx(v, rt));
    }

    // Test multiplication
    float a = 3.14f, b = 2.71f;
    float result = xlns322fp(xlns32_mul(fp2xlns32(a), fp2xlns32(b)));
    printf("mul: %.4f * %.4f = %.4f (expected %.4f)\n", a, b, result, a*b);
    assert(approx(a*b, result));

    // Test addition
    result = xlns322fp(xlns32_add(fp2xlns32(a), fp2xlns32(b)));
    printf("add: %.4f + %.4f = %.4f (expected %.4f)\n", a, b, result, a+b);
    assert(approx(a+b, result));

    // Test dot product
    float va[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vb[] = {4.0f, 3.0f, 2.0f, 1.0f};
    float dot = xlns32_vec_dot(va, vb, 4);
    printf("dot: %.4f (expected 20.0)\n", dot);
    assert(approx(20.0f, dot));

    printf("\nAll tests passed!\n");
    return 0;
}
```

### Step 8.6: Build, Test, and Install

```bash
cd ~/lns-llm-project/xlnscpp

# Build
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure

# Install
cmake --install build

# Verify
ls $HOME/.local/lib/libxlns.a
ls $HOME/.local/include/xlns/xlns32.h
ls $HOME/.local/lib/cmake/xlns/xlnsConfig.cmake
```

---

## 9. Phase 3 — Create the ggml-lns Backend Skeleton

**Time estimate:** 2–3 weeks
**Repo:** `llama.cpp` (your fork)

### Step 9.1: Create the Public Header

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

### Step 9.2: Create the Backend CMakeLists.txt

Create `ggml/src/ggml-lns/CMakeLists.txt`:

```cmake
message(STATUS "Using LNS backend")

find_package(xlns REQUIRED)

ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
)

target_link_libraries(ggml-lns PRIVATE xlns::xlns)
```

That's it. `find_package(xlns)` finds your installed library. `target_link_libraries` links it.

### Step 9.3: Implement the Backend

Create `ggml/src/ggml-lns/ggml-lns.cpp`. This is the **core of the project**.

The full skeleton with all the boilerplate (registry, device, backend interfaces):

```cpp
// ggml-lns.cpp — LNS (Logarithmic Number System) backend for ggml
// Uses xlnscpp for arithmetic

#include "ggml-lns.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <xlns/xlns32.h>    // ← This comes from our installed xlnscpp library!

#include <cstring>
#include <cstdlib>
#include <cmath>

// ============================================================
// Backend context
// ============================================================
struct ggml_backend_lns_context {
    int n_threads;
};

// ============================================================
// LNS Tensor Operations (add these incrementally in Phase 4)
// ============================================================

// --- GGML_OP_MUL_MAT ---
static void ggml_lns_op_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // weights
    const struct ggml_tensor * src1 = dst->src[1];  // input

    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // N
    const int64_t ne11 = src1->ne[1];  // M

    const float * src0_data = (const float *)src0->data;
    const float * src1_data = (const float *)src1->data;
    float * dst_data = (float *)dst->data;

    for (int64_t i1 = 0; i1 < ne11; i1++) {
        for (int64_t i0 = 0; i0 < ne01; i0++) {
            dst_data[i1 * ne01 + i0] = xlns32_vec_dot(
                &src0_data[i0 * ne00],
                &src1_data[i1 * ne00],
                ne00
            );
        }
    }
}

// --- GGML_OP_ADD ---
static void ggml_lns_op_add(struct ggml_tensor * dst) {
    const float * a = (const float *)dst->src[0]->data;
    const float * b = (const float *)dst->src[1]->data;
    float * c = (float *)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t n0 = ggml_nelements(dst->src[0]);
    const int64_t n1 = ggml_nelements(dst->src[1]);

    for (int64_t i = 0; i < n; i++) {
        c[i] = xlns322fp(xlns32_add(fp2xlns32(a[i % n0]), fp2xlns32(b[i % n1])));
    }
}

// --- GGML_OP_MUL (element-wise) ---
static void ggml_lns_op_mul(struct ggml_tensor * dst) {
    const float * a = (const float *)dst->src[0]->data;
    const float * b = (const float *)dst->src[1]->data;
    float * c = (float *)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t n0 = ggml_nelements(dst->src[0]);
    const int64_t n1 = ggml_nelements(dst->src[1]);

    for (int64_t i = 0; i < n; i++) {
        c[i] = xlns322fp(xlns32_mul(fp2xlns32(a[i % n0]), fp2xlns32(b[i % n1])));
    }
}

// ============================================================
// graph_compute — THE MAIN DISPATCH FUNCTION
// ============================================================
static enum ggml_status ggml_backend_lns_graph_compute(
    ggml_backend_t backend, struct ggml_cgraph * cgraph)
{
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

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
                break;  // no-ops (metadata only)
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
    ggml_backend_lns_context * ctx = (ggml_backend_lns_context *)backend->context;
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
};

// ============================================================
// GUID
// ============================================================
static ggml_guid_t ggml_backend_lns_guid(void) {
    static ggml_guid guid = {
        0x4c, 0x4e, 0x53, 0x42, 0x41, 0x43, 0x4b, 0x45,
        0x4e, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    };
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
    GGML_UNUSED(dev); return "LNS";
}
static const char * ggml_backend_lns_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev); return "Logarithmic Number System (xlnscpp)";
}
static void ggml_backend_lns_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0; *total = 0; GGML_UNUSED(dev);
}
static enum ggml_backend_dev_type ggml_backend_lns_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev); return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static void ggml_backend_lns_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_lns_device_get_name(dev);
    props->description = ggml_backend_lns_device_get_description(dev);
    props->type        = ggml_backend_lns_device_get_type(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}
static ggml_backend_t ggml_backend_lns_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev); GGML_UNUSED(params);
    return ggml_backend_lns_init();
}
static ggml_backend_buffer_type_t ggml_backend_lns_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();  // We use CPU host memory
}
static bool ggml_backend_lns_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return op->src[0]->type == GGML_TYPE_F32;
        default:
            return false;
    }
}
static bool ggml_backend_lns_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
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
    GGML_UNUSED(reg); return GGML_LNS_NAME;
}
static size_t ggml_backend_lns_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg); return 1;
}
static ggml_backend_dev_t ggml_backend_lns_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(reg);
    static ggml_backend_device ggml_backend_lns_device = {
        /* .iface   = */ ggml_backend_lns_device_i,
        /* .reg     = */ ggml_backend_lns_reg(),
        /* .context = */ NULL,
    };
    return &ggml_backend_lns_device;
}
static void * ggml_backend_lns_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg); GGML_UNUSED(name); return NULL;
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

### Step 9.4: Edit ggml/CMakeLists.txt

Add these 2 changes (near where other backends like BLAS are):

```cmake
# Add near other backend options:
option(GGML_LNS "ggml: use LNS backend (Logarithmic Number System)" OFF)

# Add near other add_subdirectory calls:
if (GGML_LNS)
    add_subdirectory(src/ggml-lns)
endif()
```

### Step 9.5: Edit ggml/src/ggml-backend-reg.cpp

Add 2 blocks:

Near the top with other backend includes:
```cpp
#ifdef GGML_USE_LNS
#include "ggml-lns.h"
#endif
```

Inside the registry constructor:
```cpp
#ifdef GGML_USE_LNS
    register_backend(ggml_backend_lns_reg());
#endif
```

### Step 9.6: Build and Verify

```bash
cd ~/lns-llm-project/llama.cpp
cmake -B build -DGGML_LNS=ON -DCMAKE_PREFIX_PATH=$HOME/.local -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify
./build/bin/llama-cli --list-devices
# Should show "LNS" in the device list
```

---

## 10. Phase 4 — Implement Tensor Operations in LNS

**Time estimate:** 3–4 weeks
**Repo:** `llama.cpp` (in `ggml/src/ggml-lns/ggml-lns.cpp`)

### Operations Needed for a Minimal LLM (Priority Order)

#### Tier 1 — Essential (blocks inference without these)

| Op | Description | LNS Strategy |
|----|-------------|--------------|
| `GGML_OP_MUL_MAT` | Matrix multiply | Convert to LNS, multiply (cheap), accumulate (expensive) |
| `GGML_OP_ADD` | Tensor addition | Convert to LNS, add |
| `GGML_OP_MUL` | Element-wise multiply | Convert to LNS, multiply (cheap) |
| `GGML_OP_RMS_NORM` | RMS normalization | x²=cheap, sum=expensive, sqrt=cheap |
| `GGML_OP_SOFT_MAX` | Softmax | Hybrid: use float for exp(), LNS for rest |
| `GGML_OP_ROPE` | Rotary position embedding | Hybrid: float trig, LNS multiply+add |
| `GGML_OP_SILU` | SiLU activation | `x * sigmoid(x)` in LNS |
| `GGML_OP_CPY` | Tensor copy | Direct memcpy (no LNS needed) |
| `GGML_OP_CONT` | Make contiguous | Layout operation (no LNS needed) |
| `GGML_OP_SCALE` | Multiply by scalar | Convert scalar to LNS, multiply (cheap) |

#### Tier 2 — Needed for Full Models

| Op | Description | LNS Strategy |
|----|-------------|--------------|
| `GGML_OP_NORM` | Layer normalization | Similar to RMS_NORM |
| `GGML_OP_GELU` | GELU activation | Approximation or convert-compute-convert |
| `GGML_OP_DIAG_MASK_INF` | Causal attention mask | Set values to -inf |
| `GGML_OP_GET_ROWS` | Embedding lookup | No arithmetic, just data movement |

#### Tier 3 — Can Fall Back to CPU

| Op | Why CPU is OK |
|----|---------------|
| `GGML_OP_RESHAPE` / `VIEW` / `PERMUTE` / `TRANSPOSE` | No computation, metadata only |

### Implementation Strategies for Key Operations

**RMS_NORM:**
```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

In LNS: x² = cheap (double the log), mean = sum (expensive) / n (cheap),
sqrt = halve the log (very cheap), x/rms = subtract logs (cheap)
```

**SOFT_MAX (hybrid):**
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Use float for exp() (called once per attention head, not bottleneck),
LNS for the division (cheap).
```

**SILU:**
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

exp(-x) in LNS, 1+exp(-x) = LNS add (expensive), x / result = LNS div (cheap)
```

---

## 11. Phase 5 — Register the Backend & Build Integration

This was already covered in Phase 3 Steps 9.4–9.6. By the time you reach this phase, the backend should already be compiling and registered. This phase is about making sure the **full build** works cleanly.

```bash
cmake -B build -DGGML_LNS=ON -DCMAKE_PREFIX_PATH=$HOME/.local -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## 12. Phase 6 — End-to-End Testing with llama.cpp

**Time estimate:** 2–3 weeks
**Repo:** `llama.cpp` (your fork)

### Step 12.1: Test Individual Operations

```bash
./build/bin/test-backend-ops -b LNS
# Tests each supported operation individually
```

### Step 12.2: Test with a Tiny Model

```bash
# Download a small model
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 10

# Run with LNS backend
GGML_LNS=1 ./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 10
```

### Step 12.3: Numerical Validation

```bash
# CPU baseline
./build/bin/llama-cli -m model.gguf -p "Hello world" -n 50 --seed 42 > cpu_output.txt

# LNS backend
GGML_LNS=1 ./build/bin/llama-cli -m model.gguf -p "Hello world" -n 50 --seed 42 > lns_output.txt

# Compare (may differ slightly — that's OK)
diff cpu_output.txt lns_output.txt
```

### Step 12.4: Perplexity Test

```bash
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw
GGML_LNS=1 ./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw
```

**Expected:** LNS perplexity should be slightly worse but still reasonable (within ~5-20% for 32-bit LNS).

---

## 13. Phase 7 — Performance Optimization

**Time estimate:** 2–3 weeks
**Repo:** Both `xlnscpp` and `llama.cpp`

### Optimization 1: LNS Caching

Cache LNS representations in `tensor->extra` to avoid repeated conversion:

```cpp
struct lns_tensor_extra {
    xlns32 * lns_data;
    size_t n_elements;
    bool is_valid;
};
```

### Optimization 2: Pre-convert Weights

Weights are constant during inference — convert to LNS once at load time.

### Optimization 3: Use 16-bit LNS

For attention scores and intermediate activations, xlns16 may be sufficient.

### Optimization 4: Batch Conversions

Vectorize the float↔LNS conversion bottleneck.

---

## 14. Phase 8 — Validation with a Real LLM (DeepSeek)

**Time estimate:** 1–2 weeks
**Repo:** `llama.cpp` (your fork)

### Step 14.1: Run DeepSeek

```bash
# CPU baseline
./build/bin/llama-cli -hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
    -p "Explain quantum computing" -n 200 --seed 42 > deepseek_cpu.txt

# LNS backend
GGML_LNS=1 ./build/bin/llama-cli -hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
    -p "Explain quantum computing" -n 200 --seed 42 > deepseek_lns.txt
```

### Step 14.2: Measure Quality & Performance

```bash
# Perplexity
./build/bin/llama-perplexity -m deepseek.gguf -f test_data.txt
GGML_LNS=1 ./build/bin/llama-perplexity -m deepseek.gguf -f test_data.txt

# Benchmark
./build/bin/llama-bench -m deepseek.gguf
GGML_LNS=1 ./build/bin/llama-bench -m deepseek.gguf
```

### Expected Results

| Metric | CPU (baseline) | LNS Backend | Notes |
|--------|---------------|-------------|-------|
| Perplexity | ~X | ~X + 5-20% | LNS adds quantization noise |
| Speed | ~Y tok/s | ~Y/5-15x tok/s | Software emulation is slow |
| Output quality | Coherent | Coherent | Main goal: valid output |

---

## 15. File-by-File Change Map

### New Files You Create

| File | Repo | Purpose |
|------|------|---------|
| `CMakeLists.txt` | xlnscpp | Root build file |
| `include/xlns/xlns32.h` | xlnscpp | Public 32-bit API header |
| `include/xlns/xlns16.h` | xlnscpp | Public 16-bit API header |
| `include/xlns/xlns.h` | xlnscpp | Convenience header |
| `src/xlns32.cpp` | xlnscpp | Refactored implementation |
| `tests/test_xlns32.cpp` | xlnscpp | Library tests |
| `cmake/xlnsConfig.cmake.in` | xlnscpp | CMake package config |
| `ggml/include/ggml-lns.h` | llama.cpp | Backend public header |
| `ggml/src/ggml-lns/ggml-lns.cpp` | llama.cpp | Backend implementation |
| `ggml/src/ggml-lns/CMakeLists.txt` | llama.cpp | Backend build rules |

### Existing Files You Edit (minimal changes)

| File | Repo | Change |
|------|------|--------|
| `ggml/CMakeLists.txt` | llama.cpp | +2 lines: `GGML_LNS` option + `add_subdirectory` |
| `ggml/src/ggml-backend-reg.cpp` | llama.cpp | +4 lines: `#include` + `register_backend` |

### NOT Modified

| File | Why |
|------|-----|
| `src/*.cpp` (llama.cpp source) | Backend is transparent to llama.cpp |
| `ggml/src/ggml.c` | Core ggml doesn't change |
| `ggml/include/ggml.h` | No new ops needed |
| `ggml/src/ggml-cpu/` | CPU backend stays as-is |

---

## 16. Key Design Decisions

### Decision 1: 32-bit vs 16-bit LNS

**Recommendation:** Start with **32-bit LNS** (`xlns32`)

| Factor | xlns32 | xlns16 |
|--------|--------|--------|
| Precision | 23 fractional bits | 7 fractional bits |
| Accuracy | Very close to float32 | Noticeable errors |
| Proof of concept | Easier to validate | Harder to validate |

### Decision 2: Which Operations to Implement First

**Start with `MUL_MAT` only**, let everything else fall back to CPU. MUL_MAT is 90%+ of LLM compute.

### Decision 3: Backend Type

**Use `GGML_BACKEND_DEVICE_TYPE_ACCEL`** — shares CPU memory, no buffer management complexity.

### Decision 4: Library Integration

**Use CMake `find_package`** — clean, professional, standard practice. No source copying.

---

## 17. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LNS addition noise accumulates | High | Use 32-bit LNS; validate per-layer |
| ggml backend interface changes | Medium | Pin to specific llama.cpp release |
| Some ops too complex for LNS | Low | Fall back to CPU (hybrid) |
| Performance too slow for testing | Medium | Optimize MUL_MAT; use caching |

---

## 18. Milestones & Timeline

| Week | Milestone | Deliverable |
|------|-----------|------------|
| 1-2 | **M1: Understanding** | Can explain ggml backend architecture |
| 3 | **M2: xlnscpp library** | `libxlns.a` installed, tests passing |
| 4-5 | **M3: Backend skeleton** | ggml-lns compiles, registers, shows in device list |
| 6-7 | **M4: MUL_MAT works** | Matrix multiply produces correct results in LNS |
| 8-9 | **M5: Minimal ops** | ADD, MUL, RMS_NORM, SOFTMAX, SILU working |
| 10-11 | **M6: Small model runs** | Tiny model produces coherent output with LNS |
| 12-13 | **M7: Optimization** | Caching, pre-conversion of weights |
| 14-15 | **M8: DeepSeek validation** | DeepSeek model produces valid output |
| 16 | **M9: Documentation** | Final report, benchmarks, code cleanup |

---

## Appendix A: Useful Commands

```bash
# Build xlnscpp library
cd ~/lns-llm-project/xlnscpp
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build && ctest --test-dir build && cmake --install build

# Build llama.cpp with LNS
cd ~/lns-llm-project/llama.cpp
cmake -B build -DGGML_LNS=ON -DCMAKE_PREFIX_PATH=$HOME/.local -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Test
./build/bin/test-backend-ops -b LNS
./build/bin/llama-cli --list-devices
./build/bin/llama-bench -m model.gguf
```

## Appendix B: References

1. M. G. Arnold et al., "Arithmetic cotransformations in the Real and Complex Logarithmic Number Systems," IEEE Trans. Comput., vol. 47, no. 7, 1998.
2. ggml: https://github.com/ggml-org/ggml
3. llama.cpp: https://github.com/ggml-org/llama.cpp
4. xlnscpp: https://github.com/xlnsresearch/xlnscpp
