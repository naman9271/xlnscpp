# XLNSCPP CODEBASE - COMPLETE TECHNICAL EXPLANATION
## For GSoC Project: LNS Backend for GGML

---

## EXECUTIVE SUMMARY

xlnscpp provides a **Logarithmic Number System (LNS)** implementation as an alternative to IEEE 754 floating-point arithmetic. Instead of representing numbers as `sign × mantissa × 2^exponent`, LNS stores numbers as `sign × 2^log_value`.

**Why this matters for your project:**
In LNS, multiplication becomes addition of logarithms, making it potentially faster for multiply-heavy workloads like neural network inference.

---

## PART 1: WHAT IS LNS?

### Traditional Floating-Point (FP32):
```
Number = (-1)^sign × (1.mantissa) × 2^exponent
Example: 3.5 = 1 × 1.75 × 2^1
```

### Logarithmic Number System (LNS):
```
Number = (-1)^sign × 2^(log_value)
Example: 3.5 ≈ 1 × 2^1.807...
Internal storage: sign bit | log₂(3.5) encoded
```

### Operations in LNS:

| Operation | FP32 | LNS |
|-----------|------|-----|
| Multiply  | `a × b` | `2^(log(a) + log(b))` → **Just add logs!** |
| Divide    | `a / b` | `2^(log(a) - log(b))` → **Just subtract logs!** |
| Square root | `√a` | `2^(log(a)/2)` → **Divide log by 2!** |
| Reciprocal | `1/a` | `2^(-log(a))` → **Negate the log!** |
| Add/Sub   | `a + b` | **HARD - requires Gaussian logarithms** |

**The key insight:** Multiplication is cheap, addition is expensive (opposite of FP).

---

## PART 2: FILE-BY-FILE BREAKDOWN

### **xlns16.cpp** - 16-bit LNS (bfloat16-like)

#### Type definitions:
```cpp
typedef u_int16_t xlns16;           // 16-bit LNS value
typedef int16_t xlns16_signed;      // For intermediate calcs

// Memory layout:
// Bit 15: sign
// Bits 14-7: integer part of log₂
// Bits 6-0: fractional part of log₂
```

#### Key constants:
```cpp
xlns16_zero = 0x0000              // Exact zero
xlns16_signmask = 0x8000          // Isolate sign bit
xlns16_logmask = 0x7fff           // Isolate magnitude
xlns16_logsignmask = 0x4000       // Offset for log representation
xlns16_scale = 128                // Scaling factor (128 = 2^7 for 7 frac bits)
```

#### Fast operations (implemented as macros):
```cpp
xlns16_mul(x, y):
  // Add the logs, handle sign XOR
  log_result = (x & 0x7fff) + (y & 0x7fff) - 0x4000
  sign_result = sign(x) XOR sign(y)
  return sign_result | log_result
  
xlns16_div(x, y):
  // Subtract the logs
  log_result = (x & 0x7fff) - (y & 0x7fff) + 0x4000
  return sign_result | log_result
  
xlns16_sqrt(x):
  // Divide log by 2
  return (x << 1) / 4  // Clever bit shift
  
xlns16_recip(x):
  // Negate the log
  return sign(x) | (~x + 1)
```

#### The hard operation: Addition

LNS addition requires **Gaussian logarithms**:

```cpp
// To compute x + y in LNS:
// 1. Let x be the larger value
// 2. Compute z = log(x) - log(y)
// 3. Compute sb(z) = log₂(1 + 2^z)  -- "sum base" function
// 4. Result = log(x) + sb(z)

xlns16_add(x, y):
  if sign(x) == sign(y):
    // Same sign: x + y
    z = log(max) - log(min)
    return max + sb(z)
  else:
    // Opposite sign: x - y
    z = log(max) - log(min)
    if z == 0: return 0  // Exact cancellation
    return max + db(z)   // db = "difference base"
```

#### Gaussian logarithm functions:

**Ideal mode** (uses actual FP math, slow but accurate):
```cpp
sb(z) = log₂(1 + 2^z)
db(z) = log₂(2^z - 1)
```

**Fast mode** (uses Mitchell LPVIP approximation):
```cpp
// Linear interpolation with piecewise corrections
// Based on PhD research, ~100 bytes of code
sb_premit(z):
  precond = -z / 8          // Linear precondition
  mitch = Mitchell_approx(z + precond)
  postcond = small_correction(z)
  return mitch + postcond
```

**Table mode** (fastest, ~512KB memory):
```cpp
// Pre-computed lookup tables
sb(z) = xlns16sbtbl[z]
db(z) = xlns16dbtbl[z]
```

#### Conversion functions:

```cpp
fp2xlns16(float x):
  if x == 0: return 0x0000
  sign = (x < 0) ? 0x8000 : 0x0000
  log_val = log₂(|x|) × 128  // Scale to fixed-point
  return sign | (log_val XOR 0x4000)  // Apply offset

xlns162fp(xlns16 x):
  if x == 0x0000: return 0.0
  sign = (x & 0x8000) ? -1 : 1
  log_val = ((x & 0x7fff) - 0x4000) / 128.0
  return sign × 2^log_val
```

#### C++ wrapper class:

```cpp
class xlns16_float {
  xlns16 x;  // Internal LNS representation
  
  // Operator overloading for natural syntax:
  xlns16_float operator+(xlns16_float other) {
    xlns16_float result;
    result.x = xlns16_add(this->x, other.x);
    return result;
  }
  
  xlns16_float operator*(xlns16_float other) {
    xlns16_float result;
    result.x = xlns16_mul(this->x, other.x);
    return result;
  }
  
  // Allows this syntax:
  // xlns16_float a = 2.5, b = 3.7;
  // xlns16_float c = a * b + 1.0;
};
```

---

### **xlns32.cpp** - 32-bit LNS (float32-like)

Nearly identical to xlns16 but with:
- 32-bit storage (1 sign + 8 int + 23 frac bits)
- Higher precision Gaussian logarithms
- More sophisticated table-based approximations

**Key difference:** Uses 3-level hierarchical table lookup:
```cpp
db(z):
  z0 = high bits of z
  z1 = middle bits of z
  z2 = low bits of z
  
  // Combine three table lookups with sb corrections
  result = db_table0[z0] + sb(db_table1[z1] + ...)
```

---

### **xlns16test.cpp & xlns32test.cpp** - Test Suites

#### Test 1: Sum of odd numbers
```cpp
// FP:  sum = 1 + 3 + 5 + ... + 199
// Tests: Basic addition accuracy
test1xlns16():
  odd = fp2xlns16(1.0)
  sum = fp2xlns16(0.0)
  two = fp2xlns16(2.0)
  for i = 1 to 100:
    sum = xlns16_add(sum, odd)
    odd = xlns16_add(odd, two)
  print xlns162fp(sum)  // Should be 10000
```

**Observation from output:**
- FP32: sum = 10000.0 (exact)
- LNS16: sum = 10623.7 (6.2% error due to precision)
- LNS32: sum = 99944968 (much better, 0.055% error)

#### Test 2: Factorial series (approximates e)
```cpp
// Computes Σ(1/n!) for n=1..8
// Should approach e-1 ≈ 1.718
test2xlns16():
  num = fact = one = fp2xlns16(1.0)
  sum = fp2xlns16(0.0)
  for i = 1 to 8:
    sum = xlns16_add(sum, xlns16_recip(fact))
    fact = xlns16_mul(fact, num)
    num = xlns16_add(num, one)
```

**Tests:** Multiplication, reciprocal, and cumulative error

#### Test 3: Alternating factorial
```cpp
// Σ(1/(n!) × (-1)^(n+1))
// Tests signed operations
```

#### Test 4: Mandelbrot set
```cpp
// Iterative fractal computation
// z_new = z² - y² + x1
// Heavy multiplication workload
// Visualizes as ASCII art
```

**Purpose:** Stress test for many sequential operations

#### Test 5: Pi approximation (Leibniz formula)
```cpp
// π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
// Tests division and alternating signs
```

---

### **Header files** (Lookup tables)

#### xlns16cvtbl.h
```cpp
// Forward conversion: FP32 → LNS16
// 65536 entries (256 KB)
// Index by IEEE 754 bits directly
xlns16 xlns16revcvtbl[65536];
```

#### xlns16revcvtbl.h
```cpp
// Reverse conversion: LNS16 → FP32
// 65536 entries (128 KB as floats)
float xlns16cvtbl[65536];
```

#### xlns16sbdbtbl.h
```cpp
// Pre-computed sb and db values
// About 5KB each
xlns16 xlns16sbtbl[...];
xlns16 xlns16dbtbl[...];
```

#### xlns32tbl.h
```cpp
// Hierarchical cotransformation tables
// ~300 KB total
// Three levels for interpolation
```

---

## PART 3: COMPILATION MODES

### xlns16 modes:

| Define | Effect | Speed | Accuracy | Memory |
|--------|--------|-------|----------|--------|
| `xlns16_ideal` | Use actual FP math | Slow | Perfect | Minimal |
| (default) | Mitchell LPVIP | Fast | Good | ~100 bytes |
| `xlns16_table` | Lookup tables | Fastest | Good | 512 KB |
| `xlns16_alt` | Reduced branching | Variable | Same | Same |
| `xlns16_altopt` | Alt + optimized | Fastest | Slightly less | ~5 KB |

### Example compilation:
```bash
# Ideal mode (slow, accurate)
g++ -D xlns16_ideal xlns16test.cpp -o test_ideal

# Fast mode (default, no defines needed)
g++ xlns16test.cpp -o test_fast

# Table mode (fastest)
g++ -D xlns16_table xlns16test.cpp -o test_table
```

---

## PART 4: CRITICAL NUMERIC INSIGHTS

### Precision comparison (from test outputs):

| Test | FP32 | LNS16 | LNS32 | Notes |
|------|------|-------|-------|-------|
| Sum of 100 odds | 10000.0 | 10623.7 | 99944968.0 | LNS16 has visible error |
| e approximation | 2.718254 | 2.723218 | 2.718253 | LNS32 nearly perfect |
| Pi (10 terms) | 3.041840 | 3.067764 | 3.140246 | Accumulation error |

**Key observation:** LNS16 (like bfloat16) is suitable for **low-precision** tasks. LNS32 is comparable to FP32.

### Error sources:

1. **Conversion error:** `fp2xlns32()` and `xlns322fp()` each have ~1 ULP error
2. **sb/db approximation error:** Non-ideal modes trade accuracy for speed
3. **Accumulation error:** Many additions compound errors

### When LNS wins:
- Multiply-heavy operations (matrix multiply, convolutions)
- Can tolerate reduced precision (inference, not training)
- Power-constrained environments

### When LNS loses:
- Addition-heavy operations
- Requires high precision
- Transcendental functions (sin, cos, exp, log)

---

## PART 5: HOW THIS APPLIES TO YOUR GGML BACKEND

### Your target architecture:

```
┌────────────────────────────────────────┐
│  llama.cpp (FP32 interface)            │
└────────────────────────────────────────┘
               ↓ (FP32 tensors)
┌────────────────────────────────────────┐
│  ggml context & cgraph                 │
│  (operation: C = A × B)                │
└────────────────────────────────────────┘
               ↓ (dispatch to backend)
┌────────────────────────────────────────┐
│  YOUR LNS BACKEND                      │
│                                        │
│  1. Read FP32: float* A_fp = ...      │
│  2. Convert: xlns32* A_lns =          │
│              fp2xlns32(A_fp[i])       │
│  3. Compute: for (i,j,k)              │
│       C_lns[i,j] = xlns32_add(        │
│         C_lns[i,j],                   │
│         xlns32_mul(A_lns[i,k],        │
│                    B_lns[k,j]))       │
│  4. Convert back: xlns322fp(C_lns)    │
│  5. Return FP32                       │
└────────────────────────────────────────┘
               ↓ (FP32 result)
┌────────────────────────────────────────┐
│  llama.cpp receives result             │
└────────────────────────────────────────┘
```

### Minimal operations to implement:

1. **Matrix multiply** (matmul) - REQUIRED
2. **Element-wise add** - HELPFUL
3. **Activation function** (ReLU, maybe) - NICE TO HAVE

**Do NOT attempt to implement all 100+ ggml operations.**

### Expected numeric behavior:

```c
// FP32 reference
float C_fp[8] = {22, 28, 49, 64, 76, 100, 103, 136};

// Your LNS backend
float C_lns[8];  // After conversion back
// Expected: within 0.001% for LNS32-ideal
// Expected: within 0.1% for LNS32-fast
// Expected: within 1% for LNS16
```

---

## PART 6: YOUR IMPLEMENTATION CHECKLIST

### Week 1-2: Backend skeleton
- [ ] Clone ggml repository
- [ ] Study `ggml-cpu.c` backend
- [ ] Create `ggml-lns.c` file
- [ ] Implement `ggml_backend_lns_init()`
- [ ] Register backend with ggml
- [ ] Make it compile (even if non-functional)

### Week 3-4: Matrix multiply
- [ ] Implement `ggml_backend_lns_mul_mat()`
- [ ] Add xlnscpp as a dependency
- [ ] Write FP→LNS→FP conversion wrappers
- [ ] Test with tiny matrices (4×3 × 3×2)
- [ ] Validate against FP32 results
- [ ] Measure conversion overhead

### Week 5-6: Integration test
- [ ] Run llama.cpp with LNS backend
- [ ] Test with TinyLLaMA or similar small model
- [ ] Measure perplexity difference vs FP32
- [ ] Profile where time is spent
- [ ] Document numerical errors

### Week 7-8: Documentation
- [ ] Write design proposal
- [ ] Create accuracy report with graphs
- [ ] Document architectural decisions
- [ ] Identify future optimizations
- [ ] Prepare mentor presentation

---

## PART 7: COMMON PITFALLS TO AVOID

### ❌ **Don't:**
1. Try to store tensors in LNS format permanently
   - Convert on-the-fly during computation
   - Keep buffers as FP32

2. Implement operations you don't need
   - Just matmul is enough for proof-of-concept

3. Optimize before validating correctness
   - Use xlns32_ideal first
   - Switch to fast mode only after validation

4. Ignore numeric errors
   - Document every error source
   - Provide honest analysis

5. Attempt GPU implementation
   - LNS on CPU is hard enough

### ✅ **Do:**
1. Test incrementally
   - 2×2 matrices first
   - Then 4×4
   - Then realistic sizes

2. Compare every operation
   - Print FP32 vs LNS results side-by-side
   - Compute relative error

3. Profile conversion overhead
   - How much time is spent in fp2xlns32?
   - Is computation actually faster?

4. Document scope limits
   - "Only matmul supported"
   - "Maximum tensor size: X"

5. Prepare for failure cases
   - What happens with NaN?
   - What happens with denormals?
   - What happens with very large values?

---

## PART 8: DEBUGGING STRATEGY

### When LNS results don't match FP32:

```c
// Step 1: Test individual conversions
float test_val = 3.14159;
xlns32 lns_val = fp2xlns32(test_val);
float recovered = xlns322fp(lns_val);
printf("Input: %.6f, Recovered: %.6f, Error: %.6e\n", 
       test_val, recovered, fabs(test_val - recovered));

// Step 2: Test single operations
xlns32 a = fp2xlns32(2.0);
xlns32 b = fp2xlns32(3.0);
xlns32 c = xlns32_mul(a, b);
float c_fp = xlns322fp(c);
printf("2 * 3 = %.1f (expected 6.0)\n", c_fp);

// Step 3: Test one matrix element
float A[2] = {1.5, 2.5};
float B[2] = {3.0, 4.0};
// C[0] should be 1.5*3.0 + 2.5*4.0 = 4.5 + 10.0 = 14.5

xlns32 A_lns[2] = {fp2xlns32(A[0]), fp2xlns32(A[1])};
xlns32 B_lns[2] = {fp2xlns32(B[0]), fp2xlns32(B[1])};
xlns32 prod1 = xlns32_mul(A_lns[0], B_lns[0]);
xlns32 prod2 = xlns32_mul(A_lns[1], B_lns[1]);
xlns32 sum = xlns32_add(prod1, prod2);
float result = xlns322fp(sum);
printf("Dot product: %.6f (expected 14.5)\n", result);
```

### Typical errors and causes:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Result is NaN | Overflow in conversion | Check input range |
| Result is negative of expected | Sign bit error | Debug xlns32_canon |
| Small accumulated error (<0.1%) | Normal LNS precision | Document and accept |
| Large systematic error (>1%) | Bug in sb/db implementation | Use xlns32_ideal mode |
| Results exactly zero | Cancellation issue | Check db singularity handling |

---

## PART 9: TESTING DATA FROM YOUR RUNS

### xlns16test results:
```
test1xlns16: sum=10623.71 (expected 10000, 6.2% error)
test2xlns161: sum=2.723218 (expected 2.718254, 0.18% error)
test3xlns161: sum=0.563261 (expected 0.540302, 4.2% error)
test5xlns16: 4*sum=3.067764 (expected 3.141593 for π, 2.4% error)
```

**Conclusion:** LNS16 suitable for low-precision tasks only

### xlns32test results:
```
test1xlns32: sum=99944968 (expected 100000000, 0.055% error)
test2xlns321: sum=2.718253 (expected 2.718254, 0.0004% error)
test3xlns321: sum=0.540302 (expected 0.540302, perfect!)
test5xlns32: 4*sum=3.140246 (expected 3.141593 for π, 0.043% error)
```

**Conclusion:** LNS32 comparable to FP32 for most tasks

### matmul_fp32.cpp results:
```
Matrix A (4×3) × Matrix B (3×2) = Matrix C (4×2)
C[0,0] = 22.0 (verified correct)
C[1,0] = 49.0
C[2,0] = 76.0
C[3,0] = 103.0
```

This is the **exact behavior** your LNS backend should replicate.

---

## PART 10: FINAL ARCHITECTURE DECISION

### Your LNS backend will:

1. **Appear to ggml as:** Just another CPU backend
2. **Accept:** FP32 tensor pointers
3. **Internally:** Convert to LNS, compute, convert back
4. **Return:** FP32 results
5. **Support:** Matrix multiply only (initially)
6. **Use:** xlns32 with xlns32_ideal mode for validation

### Success criteria:

- ✅ Can multiply 64×64 FP32 matrices
- ✅ Results within 0.1% of FP32 reference
- ✅ Integrated into ggml backend system
- ✅ Can run at least one LLM operation
- ✅ Comprehensive numeric analysis report

### It's okay if:

- ⚠️ Conversion overhead makes it slower than FP32
- ⚠️ Only matmul is implemented
- ⚠️ Tensor size is limited (e.g., max 1024×1024)
- ⚠️ Some edge cases fail (document them!)

### It's NOT okay if:

- ❌ Results differ from FP32 by >1%
- ❌ No validation tests
- ❌ Scope expanded beyond proof-of-concept
- ❌ Numeric errors are hidden/ignored

---

## CONCLUSION

You now understand:

1. **xlnscpp architecture:** How LNS is represented and computed
2. **Numeric trade-offs:** Where LNS wins/loses vs FP
3. **ggml backend model:** How to integrate a new compute backend
4. **Your project scope:** Minimal viable LNS backend
5. **Success criteria:** What your mentor expects

**Next immediate steps:**

1. Re-read this document until crystal clear
2. Clone ggml repository
3. Find and read `ggml-cpu.c`
4. Identify where matmul is implemented in CPU backend
5. Create `ggml-lns.c` skeleton file
6. **Ask your mentor:** "Should I start with a stub backend or copy CPU backend?"

**Do not proceed further without mentor guidance on ggml integration strategy.**

Good luck! Remember: minimal scope, validate everything, document honestly.
