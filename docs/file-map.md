# XLNSCPP File Map & Repository Structure

Complete reference for every file in the repository, its purpose, dependencies, and key contents.

---

## Repository Layout

```
xlnscpp/
├── LICENSE                  MIT License (1999–2025 Mark G. Arnold)
├── README.md                Project overview and usage summary
│
├── ── Core Library ──────────────────────────────────────────────
├── xlns32.cpp               32-bit LNS implementation (644 lines)
├── xlns16.cpp               16-bit LNS implementation (609 lines)
│
├── ── Lookup Tables ─────────────────────────────────────────────
├── xlns32tbl.h              32-bit sb interpolation tables (28,632 lines)
├── xlns16sbdbtbl.h          16-bit sb/db direct lookup tables (2,565 lines)
├── xlns16cvtbl.h            16-bit LNS→float conversion table (65,539 lines)
├── xlns16revcvtbl.h         16-bit float→LNS conversion table (131,074 lines)
│
├── ── Test Programs ─────────────────────────────────────────────
├── xlns32test.cpp           32-bit arithmetic test suite (473 lines)
├── xlns16test.cpp           16-bit arithmetic test suite (490 lines)
├── xlnsbothtest.cpp         Cross-library coexistence test (93 lines)
├── xlns32funtest.cpp        Interactive 32-bit math function tester (35 lines)
├── xlns16funtest.cpp        Interactive 16-bit math function tester (32 lines)
├── xlns16testcase.h         Compile-time test configuration matrix (36 lines)
├── time16test.cpp           Performance benchmark (179 lines)
│
├── ── Gaussian Log Cross-Validation ─────────────────────────────
├── sb16.cpp                 Ideal sb computation (CLI, for Python cross-check)
├── db16.cpp                 Ideal db computation (CLI, for Python cross-check)
├── sbmit16.cpp              LPVIP sb computation (CLI, for Python cross-check)
├── dbmit16.cpp              LPVIP db computation (CLI, for Python cross-check)
├── sbtest.py                Python script: compare ideal sb (C++ vs Python xlns)
├── dbtest.py                Python script: compare ideal db (C++ vs Python xlns)
├── sblptest.py              Python script: compare LPVIP sb (C++ vs Python xlns)
├── dblptest.py              Python script: compare LPVIP db (C++ vs Python xlns)
│
└── docs/                    Documentation
    ├── inro.md              Introduction & overview
    ├── api-reference.md     Complete API reference
    ├── architecture.md      Internal architecture & algorithms
    ├── building.md          Build instructions & usage examples
    ├── file-map.md          This file
    └── testing.md           Testing guide & validation
```

---

## Core Library Files

### `xlns32.cpp` — 32-bit LNS Implementation

**Purpose:** Complete implementation of 32-bit Logarithmic Number System arithmetic, analogous to IEEE 754 `float32` in precision.

**Lines:** 644

**Depends on:**
- `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<iostream>`
- `xlns32tbl.h` (when **not** using `xlns32_ideal`)

**Defines:**
| Symbol | Type | Description |
|--------|------|-------------|
| `xlns32` | typedef | Unsigned 32-bit integer — raw LNS representation |
| `xlns32_signed` | typedef | Signed 32-bit integer — intermediate calculations |
| `xlns32_zero`, `xlns32_scale`, etc. | macro | Constants |
| `xlns32_sign`, `xlns32_neg`, etc. | macro | Bit manipulation utilities |
| `xlns32_mul` | inline function | Multiplication |
| `xlns32_div` | inline function | Division |
| `xlns32_add` | function | Addition (two versions: `xlns32_alt` or default) |
| `xlns32_sb` | function/macro | Sum-bit Gaussian log (ideal or table-based) |
| `xlns32_db` | function | Diff-bit Gaussian log (ideal or cotransformation) |
| `fp2xlns32` | function | Float → LNS32 conversion |
| `xlns322fp` | function | LNS32 → float conversion |
| `xlns32_float` | class | Operator-overloaded wrapper |
| `float2xlns32_` | function | Cached float → xlns32_float conversion |
| `xlns32_2float` | function | xlns32_float → float |
| `xlns32_internal` | function | Extract raw xlns32 from class |
| `sin`, `cos`, `exp`, `log`, `atan`, `sqrt`, `abs` | overloaded functions | Math functions for xlns32_float |

**Compile-time options:**
- `xlns32_ideal` — use math.h for sb/db
- `xlns32_alt` — streamlined addition
- `xlns32_arch16` — 16-bit architecture types

**Structure (in order):**
1. Lines 1–20: Comments and format diagram
2. Lines 22–59: Type definitions and constants
3. Lines 60–67: Utility macros
4. Lines 68–98: `xlns32_overflow`, `xlns32_mul`, `xlns32_div`
5. Lines 99–155: `xlns32_sb`/`xlns32_db` (ideal or table+interpolation)
6. Lines 155–210: `xlns32_dbtrans3` (cotransformation, non-ideal only)
7. Lines 211–270: `xlns32_add` (alt or default)
8. Lines 270–310: `fp2xlns32`, `xlns322fp` (conversion)
9. Lines 310–400: `xlns32_float` class declaration
10. Lines 400–470: Cache, access functions, stream output
11. Lines 470–600: Operator implementations
12. Lines 600–644: Math function overloads

---

### `xlns16.cpp` — 16-bit LNS Implementation

**Purpose:** Complete implementation of 16-bit LNS arithmetic, analogous to `bfloat16`.

**Lines:** 609

**Depends on:**
- `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<iostream>`
- `xlns16sbdbtbl.h` (when `xlns16_table` + `xlns16_alt` are defined)
- `xlns16revcvtbl.h` (when `xlns16_table` is defined)
- `xlns16cvtbl.h` (when `xlns16_table` is defined)

**Additional symbols vs xlns32:**
| Symbol | Description |
|--------|-------------|
| `xlns16_F` | Fractional bits constant = 7 (used by Mitchell/LPVIP) |
| `xlns16_mitch` | Mitchell approximation function |
| `xlns16_sb_premit_neg` | LPVIP sb for negative argument |
| `xlns16_db_premit_neg` | LPVIP db for negative argument |
| `xlns16_sb_premit` | LPVIP sb wrapper |
| `xlns16_db_premit` | LPVIP db wrapper |

**Compile-time options:**
- `xlns16_ideal` — use math.h for sb/db
- `xlns16_alt` — streamlined addition
- `xlns16_altopt` — further simplified LPVIP (within alt)
- `xlns16_table` — use lookup tables for conversions and (with alt) sb/db

---

## Lookup Table Files

### `xlns32tbl.h` — 32-bit sb Interpolation Tables

**Size:** 28,632 lines (~700 KB on disk)

**Contents:**
- `xlns32_sbltable[xlns32_tablesize]` — Low 16 bits of sb values (unsigned)
- `xlns32_sbhtable[xlns32_tablesize]` — High 16 bits of sb values (unsigned char)
- `xlns32_db0table[xlns32_db0size]` — db region 0 lookup (512 entries)
- `xlns32_db1table[xlns32_db1size]` — db region 1 lookup (512 entries)
- `xlns32_db2table[xlns32_db2size]` — db region 2 lookup (1024 entries)

where `xlns32_tablesize = 4096*3 + 1000 = 13,288`.

**Included by:** `xlns32.cpp` (when `xlns32_ideal` is NOT defined)

---

### `xlns16sbdbtbl.h` — 16-bit sb/db Direct Lookup Tables

**Size:** 2,565 lines (~10 KB on disk)

**Contents:**
- `xlns16sbtbl[1280]` — `xlns16_signed` values: `sb(z)` for `z = 0..1279`
- `xlns16dbtbl[1280]` — `xlns16_signed` values: `db(z)` for `z = 0..1279`

**Included by:** `xlns16.cpp` (when `xlns16_alt` + `xlns16_table` are defined)

---

### `xlns16cvtbl.h` — 16-bit LNS→Float Conversion Table

**Size:** 65,539 lines (~1.5 MB on disk, ~256 KB in memory)

**Contents:**
- `xlns16cvtbl[65536]` — `float` values: precomputed `xlns162fp(x)` for every possible 16-bit input

**Included by:** `xlns16.cpp` (when `xlns16_table` is defined)

---

### `xlns16revcvtbl.h` — 16-bit Float→LNS Conversion Table

**Size:** 131,074 lines (~3 MB on disk, ~256 KB in memory)

**Contents:**
- `xlns16revcvtbl[131072]` — `xlns16` values: precomputed `fp2xlns16(x)` for every float, indexed by the top 17 bits of the IEEE 754 representation

**Included by:** `xlns16.cpp` (when `xlns16_table` is defined)

---

## Test Programs

### `xlns32test.cpp` — 32-bit Arithmetic Test Suite

**Purpose:** Validates correctness of xlns32 by comparing with FP on several numerical benchmarks.

**Includes:** `xlns32.cpp`

**Tests:**

| Function | Test Description | Expected Result |
|----------|-----------------|-----------------|
| `test1fp` / `test1xlns32` / `test1xlns32_float` | Sum of first 10,000 odd numbers | 100,000,000 |
| `test2fp` / `test2xlns32` / `test2xlns32_float` | Partial sum of 1/n! series (e approximation) | ≈ 2.71828 |
| `test3fp` / `test3xlns32` / `test3xlns32_float` | Alternating factorial series (cos(1) approximation) | ≈ 0.54030 |
| `test4fp` / `test4xlns32` / `test4xlns32_float` | Mandelbrot set ASCII rendering | Visual comparison |
| `test5fp` / `test5xlns32` / `test5xlns32_float` | Leibniz formula for π | ≈ 3.14159 |
| `testops` | Interactive arithmetic operation tester | User enters values |
| `testcompare` | Comparison operator validation | Matrix of boolean results |

**Note:** `test4*` and `testops` require interactive input (Enter / `scanf`).

---

### `xlns16test.cpp` — 16-bit Arithmetic Test Suite

**Purpose:** Same tests as xlns32test, adapted for 16-bit precision (reduced iteration counts where needed).

**Includes:** `xlns16.cpp`

**Compile-time configuration:** Supports the `xlns16case` macro for automated regression testing across all option combinations.

---

### `xlnsbothtest.cpp` — Cross-Library Coexistence Test

**Purpose:** Verifies that `xlns16.cpp` and `xlns32.cpp` can be included in the same translation unit without symbol conflicts.

**Includes:** Both `xlns32.cpp` and `xlns16.cpp` (with `xlns32_ideal` and `xlns16_ideal`)

**Tests:** Runs `test1` and `test5` with both 16-bit and 32-bit versions.

---

### `xlns16testcase.h` — Test Configuration Matrix

**Purpose:** Defines six compile-time configurations for automated regression testing of `xlns16test.cpp`.

**Configurations:**

| Case | Macros Defined | Description |
|------|---------------|-------------|
| 0 | (none) | Default (LPVIP without alt) |
| 1 | `xlns16_ideal` | Ideal sb/db via math.h |
| 2 | `xlns16_alt` + `xlns16_ideal` | Alt addition with ideal sb/db |
| 3 | `xlns16_alt` | Alt addition with LPVIP sb/db |
| 4 | `xlns16_alt` + `xlns16_table` | Alt addition with table sb/db |
| 5 | `xlns16_alt` + `xlns16_altopt` | Alt addition with optimized LPVIP |

**Usage:**
```bash
g++ -Dxlns16case=4 -o xlns16test xlns16test.cpp
```

---

### `time16test.cpp` — Performance Benchmark

**Purpose:** Measures the time (in seconds for 10⁹ operations, interpretable as nanoseconds per operation) of:
1. Float → xlns16_float conversion
2. xlns16_float → float conversion
3. Float → xlns16 (raw) conversion
4. xlns16 addition (summing)
5. xlns16_float addition (summing via overloaded +)
6. Float addition (baseline)
7. xlns16 multiplication
8. Float multiplication

**Includes:** `xlns16.cpp`

---

### `xlns32funtest.cpp` / `xlns16funtest.cpp` — Interactive Function Testers

**Purpose:** Interactive CLI tool that takes a float value as input and displays the result of all overloaded math functions (`sin`, `cos`, `atan`, `exp`, `log`, `abs`, `sqrt`) along with their internal hexadecimal representation.

Enter `0` to quit.

---

## Gaussian Log Cross-Validation

These files form a verification pipeline that checks XLNSCPP's Gaussian log implementations against the reference Python xlns library.

### C++ CLI Programs

| File | Computes | Mode | Called by |
|------|----------|------|-----------|
| `sb16.cpp` | `sb(z)` = `xlns16_add(z, 1)` | `xlns16_ideal` | `sbtest.py` |
| `db16.cpp` | `db(z)` = `xlns16_add(z, -1)` | `xlns16_ideal` | `dbtest.py` |
| `sbmit16.cpp` | `sb(z)` = `xlns16_add(z, 1)` | Default (LPVIP) | `sblptest.py` |
| `dbmit16.cpp` | `db(z)` = `xlns16_add(z, -1)` | Default (LPVIP) | `dblptest.py` |

Each program takes a single integer argument `z` (the internal LNS representation of the input), performs the computation, and prints the result.

### Python Test Scripts

| File | Compares | Prerequisite |
|------|----------|-------------|
| `sbtest.py` | Ideal sb: C++ vs Python xlns | `pip install xlns`; `./sb16` built |
| `dbtest.py` | Ideal db: C++ vs Python xlns | `pip install xlns`; `./db16` built |
| `sblptest.py` | LPVIP sb: C++ vs Python xlns LPVIP | `pip install xlns xlnsconf`; `./sbmit16` built |
| `dblptest.py` | LPVIP db: C++ vs Python xlns LPVIP | `pip install xlns xlnsconf`; `./dbmit16` built |

**Algorithm:** For each `z` in the test range, both Python and C++ compute the Gaussian log. Any differences are printed. Silence means all values match.

---

## Dependency Graph

```
xlns32test.cpp ─────────────► xlns32.cpp ──────► xlns32tbl.h (non-ideal only)
xlns32funtest.cpp ──────────┘

xlns16test.cpp ─────────────► xlns16.cpp ──────► xlns16sbdbtbl.h (alt+table)
xlns16funtest.cpp ──────────┤                ├──► xlns16cvtbl.h   (table)
time16test.cpp ─────────────┤                └──► xlns16revcvtbl.h (table)
sb16.cpp ───────────────────┤
db16.cpp ───────────────────┤
sbmit16.cpp ────────────────┤
dbmit16.cpp ────────────────┘

xlnsbothtest.cpp ───────────► xlns32.cpp + xlns16.cpp

sbtest.py ──── calls ──── sb16 (compiled binary)
dbtest.py ──── calls ──── db16
sblptest.py ── calls ──── sbmit16
dblptest.py ── calls ──── dbmit16
```
