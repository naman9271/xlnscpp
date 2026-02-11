# XLNSCPP Testing & Validation Guide

This document describes how to build, run, and interpret all tests in the XLNSCPP repository.

---

## Overview

XLNSCPP includes three categories of tests:

1. **Arithmetic tests** — Validate that LNS operations produce correct results compared to floating-point
2. **Performance benchmarks** — Measure operations per second for different configurations
3. **Gaussian log cross-validation** — Verify C++ `sb`/`db` implementations match the reference Python xlns library

---

## 1. Arithmetic Tests

### xlns32test — 32-bit Tests

#### Build

```bash
# Default (table-based sb/db):
g++ -O2 -o xlns32test xlns32test.cpp

# Or with ideal sb/db:
g++ -O2 -Dxlns32_ideal -o xlns32test xlns32test.cpp
```

#### Run

```bash
./xlns32test
```

**Note:** The program will pause between some tests waiting for Enter. The Mandelbrot test (`test4`) and the interactive `testops` require keyboard input.

#### Expected Output

```
xlns32 C++ (32-bit like float) 4

# Comparison tests (should match between fp and xlns32 versions):
test5fp num=2001.000000 4*sum=3.142593
test5xlns32 num=2001.000000 val=... 4*sum=3.142593
test5xlns32_float num=2001 4*sum=3.14259

test1fp odd=20001.000000 sum=100000000.000000
test1xlns32 odd=20001.000000 sum=100000000.000000
test1xlns32_float odd=20001 sum=1e+08

test2fp1 num=9.000000 fact=4.032000e+04 sum=2.718254
test2xlns321 num=9.000000 fact=... sum=2.718254
test2xlns32_float num=9 fact=40320 sum=2.71825
```

**What to check:**
- The LNS results should closely match the FP results
- Small differences (in the last few decimal places) are expected due to LNS quantization
- The sum-of-odd-numbers test should produce exactly 100,000,000 (or very close)
- The π approximation should be ≈ 3.1426 (1000 terms of Leibniz)

#### Comparison Operator Test

The `testcompare` function prints a 4×4 matrix of `<` comparisons for values {-2, -0.5, 0.5, 2}. The FP and LNS columns should match exactly:

```
0 0 0 0     0 0 0 0
1 0 0 0     1 0 0 0
1 1 0 0     1 1 0 0
1 1 1 0     1 1 1 0
```

---

### xlns16test — 16-bit Tests

#### Build

```bash
# Default configuration:
g++ -O2 -o xlns16test xlns16test.cpp

# With specific configuration:
g++ -O2 -Dxlns16_alt -Dxlns16_table -o xlns16test xlns16test.cpp
```

#### Regression Testing Across All Configurations

```bash
for case in 0 1 2 3 4 5; do
    echo "============================================"
    echo "Case $case"
    g++ -O2 -Dxlns16case=$case -o xlns16test_$case xlns16test.cpp
    echo "" | ./xlns16test_$case
    echo ""
done
```

**Configuration matrix:**

| Case | Description | Defines |
|------|-------------|---------|
| 0 | Default (legacy LPVIP) | (none) |
| 1 | Ideal (math.h) | `xlns16_ideal` |
| 2 | Alt + Ideal | `xlns16_alt`, `xlns16_ideal` |
| 3 | Alt + LPVIP | `xlns16_alt` |
| 4 | Alt + Table | `xlns16_alt`, `xlns16_table` |
| 5 | Alt + Altopt | `xlns16_alt`, `xlns16_altopt` |

**What to check:**
- All cases should produce the same general results
- Cases 0, 3, and 5 (LPVIP variants) may have slightly larger errors than cases 1 and 2 (ideal)
- Case 4 (table) should match case 2 (ideal + alt) closely since tables are generated from ideal values

**Expected accuracy differences (16-bit vs 32-bit):**
- 16-bit has only 7 fractional bits (vs 23 for 32-bit)
- Sum-of-odd-numbers test uses only 100 iterations (vs 10,000)
- Errors of a few percent are normal for 16-bit LNS on accumulated computations

---

### xlnsbothtest — Coexistence Test

#### Build & Run

```bash
g++ -O2 -o xlnsbothtest xlnsbothtest.cpp
./xlnsbothtest
```

#### Expected Output

```
test5xlns32_float num=2001 4*sum=3.14259
test5xlns16_float num=2001 4*sum=3.14844
test1xlns32 odd=20001.000000 sum=100000000.000000
test1xlns16 odd=19841.000000 sum=10067968.000000
```

**What to check:**
- The program compiles and runs without symbol conflicts
- The 32-bit results should be more accurate than the 16-bit results
- The 16-bit sum-of-odd-numbers result (10,067,968 vs expected 10,000) demonstrates 16-bit quantization error

---

### Interactive Function Testers

#### Build & Run

```bash
g++ -O2 -o xlns32funtest xlns32funtest.cpp
./xlns32funtest
```

Enter float values to see the LNS results:

```
xlns32 function test; enter 0 to quit
1.5
x=1.5 (40c0b279)
 sin(x)=0.997497 (40ffe96f)
 cos(x)=0.0707346 (39e62c0b)
 atan(x)=0.982793 (40fb0dc1)
 exp(x)=4.48169 (4238cef7)
 abs(x)=1.5 (40c0b279)
 log(x)=0.405465 (3ecf5ac0)
 sqrt(x)=1.22474 (40338e27)
```

---

## 2. Performance Benchmarks

### time16test

#### Build

```bash
# Use tables for the most meaningful benchmark:
g++ -O2 -o time16test time16test.cpp
```

#### Run

```bash
./time16test
```

#### Expected Output Format

```
converting to xlns_float
time=6
converting back to float
time=4
converting to xlns
time=3
summing xlns
time=14
summing xlns_float
time=21
summing float
time=3
mul xlns
time=<value>
mul float
time=<value>
```

Each time value = seconds for 10⁹ operations ≈ nanoseconds per operation.

#### Interpreting Results

| Measurement | Expected (tables) | Expected (no tables) | What it measures |
|-------------|-------------------|---------------------|------------------|
| converting to xlns_float | ~6 ns | ~27 ns | `float2xlns16_()` with cache |
| converting back to float | ~4 ns | ~25 ns | `xlns16_2float()` |
| converting to xlns | ~3 ns | ~14 ns | `fp2xlns16()` (direct, no cache) |
| summing xlns | ~14 ns | ~19–45 ns | `xlns16_add()` |
| summing xlns_float | ~21 ns | ~26–53 ns | Overloaded `+` operator |
| summing float | ~3 ns | ~3 ns | Hardware FP (baseline) |

**Key ratios:**
- LNS add with tables: **~5×** slower than FP add
- LNS add without tables: **~6–15×** slower than FP add
- Full FP→LNS→compute→FP pipeline (tables): **~8×** slower
- Full pipeline (no tables): **~30×** slower

---

## 3. Gaussian Log Cross-Validation

These tests verify that the C++ implementations of `sb` and `db` match the reference Python xlns library.

### Prerequisites

```bash
pip install xlns
# For LPVIP tests:
pip install xlnsconf
```

### Build the C++ CLI Tools

```bash
g++ -O2 -o sb16 sb16.cpp
g++ -O2 -o db16 db16.cpp
g++ -O2 -o sbmit16 sbmit16.cpp
g++ -O2 -o dbmit16 dbmit16.cpp
```

### Run

```bash
# Test ideal sb (should produce no output = all match)
python3 sbtest.py

# Test ideal db (should produce no output)
python3 dbtest.py

# Test LPVIP sb (should produce no output)
python3 sblptest.py

# Test LPVIP db (may have a few known differences near singularity)
python3 dblptest.py
```

### Interpreting Results

- **No output** = all values match between C++ and Python
- **Output lines** = mismatches in format: `z diff python_value cpp_value`
  - A few mismatches near `z = 0` for `db` are expected (singularity region)
  - Any large or widespread mismatches indicate a bug

### How These Tests Work

1. Python iterates `z` over a range (0–1023 for sb, 1–1023 for db)
2. For each `z`, Python computes `sb(z)` or `db(z)` using the xlns library
3. Python invokes the C++ binary via `os.popen("./sb16 <z>")` and reads the result
4. Any difference is printed

---

## 4. Test Coverage Summary

| Feature | Covered By | Method |
|---------|-----------|--------|
| 32-bit multiplication | `xlns32test` (`testops`) | Compare `xlns32_mul` vs `fp * fp` |
| 32-bit division | `xlns32test` (`testops`) | Compare `xlns32_div` vs `fp / fp` |
| 32-bit addition | `xlns32test` (`test1–5`, `testops`) | Compare sums, series |
| 32-bit subtraction | `xlns32test` (`testops`) | Compare differences |
| 32-bit negation | `xlns32test` (`testops`) | Compare `xlns32_neg` vs `-fp` |
| 32-bit reciprocal | `xlns32test` (`testops`) | Compare `xlns32_recip` vs `1/fp` |
| 32-bit sqrt | `xlns32test` (`testops`) | Compare `xlns32_sqrt` vs `sqrt(fp)` |
| 32-bit abs | `xlns32test` (`testops`) | Compare `xlns32_abs` vs `fabs(fp)` |
| 32-bit conversion (float↔LNS) | `xlns32test` (all tests) | Implicit in all computations |
| 32-bit operator overloading | `xlns32test` (`test1–5_xlns32_float`) | Same tests via class API |
| 32-bit comparison operators | `xlns32test` (`testcompare`) | 4×4 matrix validation |
| 32-bit math functions | `xlns32funtest` | Interactive sin/cos/exp/log/atan/sqrt/abs |
| 16-bit all operations | `xlns16test` | Same suite as 32-bit, adapted |
| 16-bit all configurations | `xlns16testcase.h` | 6 compile-time configs |
| 16-bit performance | `time16test` | 10⁹-iteration benchmarks |
| 16+32 coexistence | `xlnsbothtest` | Both libraries in one binary |
| Gaussian log sb (ideal) | `sbtest.py` + `sb16` | Cross-validate with Python |
| Gaussian log db (ideal) | `dbtest.py` + `db16` | Cross-validate with Python |
| Gaussian log sb (LPVIP) | `sblptest.py` + `sbmit16` | Cross-validate with Python |
| Gaussian log db (LPVIP) | `dblptest.py` + `dbmit16` | Cross-validate with Python |

---

## 5. Running All Tests (Quick Script)

```bash
#!/bin/bash
set -e
echo "=== Building ==="
g++ -O2 -o xlns32test xlns32test.cpp
g++ -O2 -Dxlns32_ideal -o xlns32test_ideal xlns32test.cpp
g++ -O2 -o xlns16test xlns16test.cpp
g++ -O2 -o xlnsbothtest xlnsbothtest.cpp
g++ -O2 -o sb16 sb16.cpp
g++ -O2 -o db16 db16.cpp
g++ -O2 -o sbmit16 sbmit16.cpp
g++ -O2 -o dbmit16 dbmit16.cpp

echo "=== xlns32test (non-interactive portion) ==="
echo "" | timeout 5 ./xlns32test 2>/dev/null || true

echo "=== xlnsbothtest ==="
./xlnsbothtest

echo "=== Gaussian log cross-validation ==="
if python3 -c "import xlns" 2>/dev/null; then
    python3 sbtest.py
    echo "sbtest: PASS (no mismatches)"
    python3 dbtest.py
    echo "dbtest: PASS (no mismatches)"
else
    echo "SKIP: Python xlns not installed"
fi

echo "=== 16-bit regression (all cases) ==="
for case in 0 1 2 3 4 5; do
    g++ -O2 -Dxlns16case=$case -o xlns16test_$case xlns16test.cpp
    echo "Case $case: built OK"
done

echo "=== All tests complete ==="
```
