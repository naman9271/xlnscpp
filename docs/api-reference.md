# XLNSCPP API Reference

This document provides a complete reference for every public symbol, function, macro, type, and class in the XLNSCPP library.

---

## Table of Contents

1. [xlns32 — 32-bit LNS](#xlns32--32-bit-lns)
   - [Types](#xlns32-types)
   - [Constants / Macros](#xlns32-constants)
   - [Utility Macros](#xlns32-utility-macros)
   - [Core Functions](#xlns32-core-functions)
   - [Conversion Functions](#xlns32-conversion-functions)
   - [Class: xlns32_float](#class-xlns32_float)
2. [xlns16 — 16-bit LNS](#xlns16--16-bit-lns)
   - [Types](#xlns16-types)
   - [Constants / Macros](#xlns16-constants)
   - [Utility Macros](#xlns16-utility-macros)
   - [Core Functions](#xlns16-core-functions)
   - [Conversion Functions](#xlns16-conversion-functions)
   - [Class: xlns16_float](#class-xlns16_float)
3. [Helper / Access Functions](#helper--access-functions)
4. [Math Functions (Overloaded)](#math-functions-overloaded)

---

## xlns32 — 32-bit LNS

### xlns32 Types

```cpp
typedef unsigned int xlns32;         // Raw 32-bit LNS representation
typedef signed int   xlns32_signed;  // Signed variant for intermediate calculations
```

On 16-bit architectures (when `xlns32_arch16` is defined):
```cpp
typedef unsigned long xlns32;
typedef signed long   xlns32_signed;
```

### xlns32 Constants

| Macro | Value (32/64-bit arch) | Description |
|-------|------------------------|-------------|
| `xlns32_zero` | `0x00000000` | Encoding of exact 0.0 |
| `xlns32_scale` | `0x00800000` (2²³) | Fixed-point scaling factor: 1 unit of the integer log = `scale` units in raw representation |
| `xlns32_logmask` | `0x7FFFFFFF` | Bit mask to extract the 31-bit log magnitude (strips the sign bit) |
| `xlns32_signmask` | `0x80000000` | Bit mask to extract/test the sign bit |
| `xlns32_logsignmask` | `0x40000000` | Offset/bias constant. The raw log is XOR'd with this to convert from biased to unbiased representation |
| `xlns32_canonmask` | `0x80000000` | Used by `xlns32_canon` for comparison ordering |
| `xlns32_sqrtmask` | `0x20000000` | Used by `xlns32_sqrt` to offset the halved log |
| `xlns32_esszer` | `0x0CFA0000` | "Essentially zero" threshold. When `|z| ≥ esszer`, `sb(z) ≈ 0` and `db(z) ≈ z`, so the Gaussian log computation can be skipped |
| `xlns32_canonshift` | `31` | Shift amount for canonical form |

**Table-mode constants (non-ideal only):**

| Macro | Value | Description |
|-------|-------|-------------|
| `xlns32_tablesize` | `4096*3+1000 = 13288` | Number of entries in the `sb` interpolation tables |
| `xlns32_zhmask` | `0x0FFFC000` | Mask for high bits of `z` (table index) |
| `xlns32_zlmask` | `0x00003FFF` | Mask for low bits of `z` (interpolation fraction) |
| `xlns32_zhshift` | `14` | Shift to get table index from `z` |
| `xlns32_db0mask/shift/size` | Various | Partitioning masks for 3-region `db` cotransformation |
| `xlns32_db1mask/shift/size` | Various | |
| `xlns32_db2mask/size` | Various | |

### xlns32 Utility Macros

```cpp
xlns32_sign(x)    // Extract sign bit: (x) & signmask
xlns32_neg(x)     // Negate: flip sign bit: (x) ^ signmask
xlns32_abs(x)     // Absolute value: strip sign bit: (x) & logmask
xlns32_recip(x)   // Reciprocal: sign(x) | abs(~x + 1)
                  // In log domain, 1/x = -log(x), which is bitwise NOT + 1
xlns32_sqrt(x)    // Square root: halve the log magnitude, adjust offset
xlns32_canon(x)   // Canonical ordering form for comparison operators
xlns32_sub(x,y)   // Subtraction: xlns32_add(x, xlns32_neg(y))
```

### xlns32 Core Functions

#### `xlns32 xlns32_mul(xlns32 x, xlns32 y)`

**Multiplication** — the flagship cheap operation of LNS.

- **Algorithm:** Add the log magnitudes, subtract the bias, XOR the sign bits.
- **Cost:** 3 bitwise operations + 1 integer add + 1 integer sub + overflow check.
- **Overflow:** If the result log exceeds the representable range, returns the maximum magnitude with the correct sign (via `xlns32_overflow`).

```
result_log = (x & logmask) + (y & logmask) - logsignmask
result_sign = sign(x) XOR sign(y)
```

#### `xlns32 xlns32_div(xlns32 x, xlns32 y)`

**Division** — equally cheap.

- **Algorithm:** Subtract the log magnitudes, add the bias, XOR the sign bits.
- **Cost:** Same as `xlns32_mul`.

```
result_log = (x & logmask) - (y & logmask) + logsignmask
result_sign = sign(x) XOR sign(y)
```

#### `xlns32 xlns32_add(xlns32 x, xlns32 y)`

**Addition** — the expensive operation that requires Gaussian logarithms.

Two implementations are provided, selected at compile time:

**`xlns32_alt` (streamlined, modern arch):**
1. Extract log magnitudes: `xl`, `yl`
2. Find max and min: `maxxy` (with sign), `minxyl` (log only)
3. Compute `z = minxyl - log(maxxy)` — always ≤ 0
4. Determine if same-sign (addition) or different-sign (subtraction) via `usedb = sign(x) XOR sign(y)`
5. Compute adjustment:
   - **Ideal:** `adjust = z + log₂(±1 + 2^(-z/scale)) * scale`
   - **Table-based:** `adjust = z + sb(-z)` (same sign) or `z + db(-z)` (different sign)
6. If `z < -esszer`: adjustment is 0 (the smaller operand is negligible)
7. If `z == 0` and subtraction: return `xlns32_zero` (equal magnitudes cancel)
8. Result: `xlns32_mul(maxxy, logsignmask + adjustez)` — multiply max by `(1 + 2^z)` or `|1 - 2^z|`

**Default (legacy, less branching-sensitive):**
1. Compute `z = |log(x) - log(y)|`, swap to ensure `z ≥ 0`
2. If different sign:
   - `z == 0` → return 0 (cancellation)
   - `z < esszer` → return `neg(y + db(z))`
   - else → return `neg(y + z)` (since `db(z) ≈ z` for large `z`)
3. If same sign: return `y + sb(z)`

#### `xlns32 xlns32_sb(xlns32 z)` — Sum-bit Gaussian Logarithm

Computes: $sb(z) = \lfloor \log_2(1 + 2^{z/\text{scale}}) \cdot \text{scale} + 0.5 \rfloor$

- **Ideal mode:** Direct computation using `log()` and `pow()` from `<math.h>`.
- **Table mode:** Linear interpolation from `xlns32_sbhtable` and `xlns32_sbltable` (stored in `xlns32tbl.h`). The table has 13,288 entries. The high part of `z` selects the table entry; the low part drives linear interpolation between entries.

#### `xlns32 xlns32_db(xlns32 z)` — Difference-bit Gaussian Logarithm

Computes: $db(z) = \lfloor \log_2(|1 - 2^{z/\text{scale}}|) \cdot \text{scale} + 0.5 \rfloor$

- **Ideal mode:** Direct computation using `log()` and `pow()`.
- **Table mode:** Three-region cotransformation using `xlns32_db0table`, `xlns32_db1table`, `xlns32_db2table`, with recursive use of `xlns32_sb` for correction terms.

#### `xlns32 xlns32_overflow(xlns32 x, xlns32 y, xlns32 temp)`

Handles overflow in `mul`/`div` when the result log magnitude exceeds the representable range. Returns zero (if underflow) or max magnitude (if overflow), with the correct sign.

### xlns32 Conversion Functions

#### `xlns32 fp2xlns32(float x)`

Converts IEEE 754 `float` to LNS32 representation.

- **Algorithm:** `log₂(|x|) × scale`, then XOR with `logsignmask` to apply bias, set sign bit if negative.
- **Cost:** One `log()` and one `fabs()` call — expensive.
- **Special case:** `x == 0.0` → `xlns32_zero`.

#### `float xlns322fp(xlns32 x)`

Converts LNS32 representation back to IEEE 754 `float`.

- **Algorithm:** `±2^((log_magnitude - logsignmask) / scale)`.
- **Cost:** One `pow()` call — expensive.
- **Special case:** `abs(x) == xlns32_zero` → `0.0`.

### Class: xlns32_float

A wrapper class providing C++ operator overloading around the raw `xlns32` functions.

**Internal state:**
```cpp
class xlns32_float {
    xlns32 x;  // The raw LNS32 representation
    // ...
};
```

**Constructors / Assignment:**

| Signature | Behavior |
|-----------|----------|
| `xlns32_float::operator=(float rvalue)` | Converts float to LNS via `float2xlns32_()` (cached) |

**Arithmetic operators (LNS × LNS):**

| Operator | Maps To |
|----------|---------|
| `a + b` | `xlns32_add(a.x, b.x)` |
| `a - b` | `xlns32_sub(a.x, b.x)` |
| `a * b` | `xlns32_mul(a.x, b.x)` |
| `a / b` | `xlns32_div(a.x, b.x)` |
| `-a` | `xlns32_neg(a.x)` |
| `a += b` | `a = a + b` |
| `a -= b` | `a = a - b` |
| `a *= b` | `a = a * b` |
| `a /= b` | `a = a / b` |

**Mixed-mode operators (float × LNS and LNS × float):**

All mixed-mode operators auto-convert the float operand to LNS via `float2xlns32_()` (cached), then delegate to the LNS-only operator.

| Operator | Example |
|----------|---------|
| `float + xlns32_float` | `2.0f + a` |
| `xlns32_float + float` | `a + 2.0f` |
| `float * xlns32_float` | `2.0f * a` |
| etc. | All four arithmetic ops supported both ways |

**Comparison operators:**

| Operator | Algorithm |
|----------|-----------|
| `a == b` | `a.x == b.x` (raw bitwise equality) |
| `a != b` | `a.x != b.x` |
| `a < b` | `xlns32_canon(a.x) < xlns32_canon(b.x)` |
| `a > b` | `xlns32_canon(a.x) > xlns32_canon(b.x)` |
| `a <= b` | `xlns32_canon(a.x) <= xlns32_canon(b.x)` |
| `a >= b` | `xlns32_canon(a.x) >= xlns32_canon(b.x)` |

Comparisons with `float` auto-convert the float operand first.

**Access functions:**

| Function | Returns |
|----------|---------|
| `xlns32_internal(y)` | Raw `xlns32` value |
| `xlns32_2float(y)` | `float` value via `xlns322fp()` |
| `float2xlns32_(f)` | `xlns32_float` from `float`, with cache lookup |

**Stream output:**
```cpp
std::ostream& operator<<(std::ostream&, xlns32_float)
// Prints the float-converted value
```

---

## xlns16 — 16-bit LNS

The 16-bit API mirrors the 32-bit API exactly, with `xlns16` replacing `xlns32` throughout. Only the differences are noted here.

### xlns16 Types

```cpp
typedef uint16_t xlns16;          // Raw 16-bit LNS representation
typedef int16_t  xlns16_signed;   // Signed variant
```

### xlns16 Constants

| Macro | Value | Description |
|-------|-------|-------------|
| `xlns16_zero` | `0x0000` | Encoding of 0.0 |
| `xlns16_scale` | `0x0080` (128 = 2⁷) | Scale factor |
| `xlns16_logmask` | `0x7FFF` | 15-bit log magnitude mask |
| `xlns16_signmask` | `0x8000` | Sign bit mask |
| `xlns16_logsignmask` | `0x4000` | Offset/bias |
| `xlns16_esszer` | `0x0500` (1280) | Essentially-zero threshold |
| `xlns16_F` | `7` | Number of fractional bits (non-ideal mode) |

### xlns16 Utility Macros

Identical pattern to xlns32:
```cpp
xlns16_sign(x)    xlns16_neg(x)     xlns16_abs(x)
xlns16_recip(x)   xlns16_sqrt(x)    xlns16_canon(x)
xlns16_sub(x,y)
```

### xlns16 Core Functions

#### `xlns16 xlns16_mul(xlns16 x, xlns16 y)`
#### `xlns16 xlns16_div(xlns16 x, xlns16 y)`

Identical algorithm to the 32-bit versions, operating on 16-bit values.

#### `xlns16 xlns16_add(xlns16 x, xlns16 y)`

Same two implementations (`xlns16_alt` or default) as xlns32, but with additional options:

- **`xlns16_ideal`:** Uses `log`/`pow` (accurate, slow).
- **`xlns16_table` (with `xlns16_alt`):** Direct table lookup from `xlns16sbtbl[]` / `xlns16dbtbl[]` — fastest option.
- **`xlns16_altopt` (with `xlns16_alt`, without table):** Streamlined Mitchell/LPVIP with fewer operations but slightly less accuracy.
- **Default (without `xlns16_alt`):** LPVIP approximation using `xlns16_sb_premit` / `xlns16_db_premit`.

#### LPVIP / Mitchell Functions (non-ideal, non-table mode)

| Function | Description |
|----------|-------------|
| `xlns16_mitch(z)` | Mitchell's approximation: `(1 << F + (z & ((1<<F)-1))) >> (-(z >> F))` |
| `xlns16_sb_premit_neg(z)` | Pre-Mitchell sb for `z ≤ 0`, with post-condition correction |
| `xlns16_db_premit_neg(z)` | Pre-Mitchell db for `z < 0`, with pre-condition and singularity handling |
| `xlns16_sb_premit(z)` | sb wrapper for `z ≥ 0`: `sb_premit_neg(-z) + z` |
| `xlns16_db_premit(z)` | db wrapper for `z > 0`: `db_premit_neg(-z) + z` |

### xlns16 Conversion Functions

#### `xlns16 fp2xlns16(float x)`

- **Table mode (`xlns16_table`):** Direct lookup from `xlns16revcvtbl[]` indexed by the top 17 bits of the float's bit pattern. This table has 131,072 entries (~256 KB).
- **Computed mode:** Same `log₂` computation as xlns32.

#### `float xlns162fp(xlns16 x)`

- **Table mode:** Direct lookup from `xlns16cvtbl[]` indexed by the 16-bit value. This table has 65,536 entries (~256 KB).
- **Computed mode:** Same `pow(2, ...)` computation as xlns32.

### Class: xlns16_float

Identical interface to `xlns32_float`:

- All arithmetic operators (`+`, `-`, `*`, `/`, unary `-`)
- All compound assignment operators (`+=`, `-=`, `*=`, `/=`)
- Mixed-mode operators with `float`
- All comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- Math functions: `sin`, `cos`, `exp`, `log`, `atan`, `abs`, `sqrt`
- Access: `xlns16_internal()`, `xlns16_2float()`, `float2xlns16_()`
- Stream output: `operator<<`

**Cache:** 1024 entries, **off by default** (`xlns16_cacheon 0`) since table lookup is faster when tables are enabled.

---

## Helper / Access Functions

These functions bridge between the raw integer types and the class types:

| Function | Signature | Description |
|----------|-----------|-------------|
| `xlns32_internal` | `xlns32 xlns32_internal(xlns32_float y)` | Extract raw `xlns32` from class |
| `xlns32_2float` | `float xlns32_2float(xlns32_float y)` | Convert class to `float` |
| `float2xlns32_` | `xlns32_float float2xlns32_(float y)` | Convert `float` to class (with cache) |
| `xlns16_internal` | `xlns16 xlns16_internal(xlns16_float y)` | Extract raw `xlns16` from class |
| `xlns16_2float` | `float xlns16_2float(xlns16_float y)` | Convert class to `float` |
| `float2xlns16_` | `xlns16_float float2xlns16_(float y)` | Convert `float` to class (with cache) |

---

## Math Functions (Overloaded)

The following math functions are overloaded for both `xlns16_float` and `xlns32_float`. Unless noted, they work by converting to `float`, calling the standard `<math.h>` function, and converting back.

| Function | LNS-native? | Notes |
|----------|-------------|-------|
| `sin(x)` | No | Convert → `sin` → convert back |
| `cos(x)` | No | Convert → `cos` → convert back |
| `exp(x)` | No | Could be efficient in LNS (shift log) but currently uses FP |
| `log(x)` | No | Could be efficient in LNS but currently uses FP |
| `atan(x)` | No | Convert → `atan` → convert back |
| `sqrt(x)` | **Yes** | Efficient: `xlns32_sqrt` macro halves the log magnitude |
| `abs(x)` | **Yes** | Efficient: `xlns32_abs` macro strips the sign bit |
