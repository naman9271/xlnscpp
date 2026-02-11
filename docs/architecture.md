# XLNSCPP Architecture & Internals

This document describes the internal architecture of the XLNSCPP library — how the data is encoded, how arithmetic works at the bit level, and how the various approximation techniques function.

---

## Table of Contents

1. [Encoding Scheme](#1-encoding-scheme)
2. [Multiplication & Division — The Easy Operations](#2-multiplication--division)
3. [Addition & Subtraction — The Hard Operation](#3-addition--subtraction)
4. [Gaussian Logarithms: sb and db](#4-gaussian-logarithms-sb-and-db)
5. [sb/db Implementation Strategies](#5-sbdb-implementation-strategies)
6. [Conversion Between FP and LNS](#6-conversion-between-fp-and-lns)
7. [Conversion Cache Architecture](#7-conversion-cache-architecture)
8. [Comparison and Canonical Form](#8-comparison-and-canonical-form)
9. [Overflow and Special Values](#9-overflow-and-special-values)

---

## 1. Encoding Scheme

### Bit Layout

A real number $v$ is represented as:

$$v = (-1)^s \cdot 2^{L}$$

where:
- $s$ is the sign bit (1 = negative, 0 = positive)
- $L$ is the real-valued logarithm base 2 of $|v|$

$L$ is stored as a **fixed-point** number with the integer and fractional parts packed into the remaining bits. The integer part is stored in **biased** (offset) form, not two's complement.

### Concrete Layout — xlns32

```
Bit: 31  30       23  22                    0
     +---+----------+------------------------+
     | s | int(log₂)|     frac(log₂)         |
     +---+----------+------------------------+
       1      8              23 bits
```

The raw stored value (ignoring sign) is:

$$\text{raw} = (L \cdot 2^{23}) \oplus \texttt{0x40000000}$$

where $\oplus$ is XOR. The `logsignmask` (`0x40000000`) acts as a **bias** that shifts the zero point of the integer log to the middle of the representable range.

To recover $L$ from the raw value:

$$L = \frac{(\text{raw} \oplus \texttt{logsignmask}) - \texttt{logsignmask}}{2^{23}}$$

Wait — more precisely, the code does:

$$L = \frac{(\text{raw\_abs} - \texttt{logsignmask})}{\text{scale}}$$

where `raw_abs = raw & logmask` and `scale = 0x00800000 = 2^23`.

### Concrete Layout — xlns16

Same structure but with 7 fractional bits:

```
Bit: 15  14        7  6            0
     +---+----------+---------------+
     | s | int(log₂)|  frac(log₂)   |
     +---+----------+---------------+
       1      8           7 bits
```

Scale = $2^7 = 128$.

### Special Values

| Value | Encoding | Notes |
|-------|----------|-------|
| 0.0 | `0x00000000` / `0x0000` | All bits zero |
| +1.0 | `logsignmask` = `0x40000000` / `0x4000` | log₂(1) = 0, bias makes this the logsignmask value |
| -1.0 | `signmask \| logsignmask` = `0xC0000000` / `0xC000` | Same log, sign bit set |
| +2.0 | `logsignmask + scale` | log₂(2) = 1.0 = scale units |
| +0.5 | `logsignmask - scale` | log₂(0.5) = -1.0 |
| No NaN | — | Not representable |
| No ±∞ | — | Not representable |
| No subnormals | — | Not representable |

---

## 2. Multiplication & Division

### Multiplication

Given $x = (-1)^{s_x} \cdot 2^{L_x}$ and $y = (-1)^{s_y} \cdot 2^{L_y}$:

$$x \cdot y = (-1)^{s_x \oplus s_y} \cdot 2^{L_x + L_y}$$

In the biased representation:

```
raw_x = log_x + bias     (where bias = logsignmask)
raw_y = log_y + bias

raw_result = (raw_x + raw_y) - bias    // Adding logs, removing double-bias
sign_result = sign_x XOR sign_y
```

**Implementation in code:**
```cpp
xlns32_temp = (x & logmask) + (y & logmask) - logsignmask;
result = (signmask & (x ^ y)) | xlns32_temp;
```

**Cost:** 3 bitwise ops + 1 add + 1 sub + 1 branch (overflow check).

### Division

$$\frac{x}{y} = (-1)^{s_x \oplus s_y} \cdot 2^{L_x - L_y}$$

```
raw_result = (raw_x - raw_y) + bias    // Subtracting logs, adding bias back
sign_result = sign_x XOR sign_y
```

**Cost:** Same as multiplication.

---

## 3. Addition & Subtraction

### The Fundamental Challenge

Given $x = (-1)^{s_x} \cdot 2^{L_x}$ and $y = (-1)^{s_y} \cdot 2^{L_y}$, computing $x + y$ requires:

$$x + y = (-1)^{s_x} \cdot 2^{L_x} + (-1)^{s_y} \cdot 2^{L_y}$$

Without loss of generality, assume $L_x \geq L_y$ (swap if needed). Factor out $2^{L_x}$:

$$x + y = (-1)^{s_x} \cdot 2^{L_x} \left(1 + (-1)^{s_x \oplus s_y} \cdot 2^{L_y - L_x}\right)$$

Let $z = L_y - L_x \leq 0$:

- **Same sign** ($s_x = s_y$): $|x + y| = 2^{L_x} \cdot (1 + 2^z)$
  - Result log: $L_x + \log_2(1 + 2^z) = L_x + sb(z)$
- **Different sign** ($s_x \neq s_y$): $|x + y| = 2^{L_x} \cdot |1 - 2^z|$
  - Result log: $L_x + \log_2|1 - 2^z| = L_x + db(z)$

The functions $sb(z)$ and $db(z)$ are the **Gaussian logarithms** — the core difficulty of LNS.

### The "Essentially Zero" Optimization

When $|z|$ is large (the two operands differ greatly in magnitude), the smaller operand has negligible effect:

- $sb(z) \to 0$ as $z \to -\infty$ (since $2^z \to 0$, so $\log_2(1 + 2^z) \to 0$)
- $db(z) \to z$ as $z \to -\infty$ (since $|1 - 2^z| \to 1$, so $\log_2|1-2^z| \to 0$, and $db(z) = z + \log_2|2^{-z}-1| \to z$)

The threshold `esszer` (0x0CFA0000 for 32-bit, 0x0500 for 16-bit) defines the cutoff below which the sb/db correction is skipped.

### Cancellation Case

When $z = 0$ (equal magnitudes) and the signs differ, $x + y = 0$ exactly. This is handled as a special case returning `xlns32_zero`.

---

## 4. Gaussian Logarithms: sb and db

### Mathematical Definitions

$$sb(z) = \log_2(1 + 2^z) \qquad \text{for } z \leq 0$$

$$db(z) = \log_2(2^z - 1) \qquad \text{for } z > 0$$

Note: In the code, the sign conventions may be flipped (the functions are called with the absolute value, and the sign is handled separately).

### Properties

**sb (sum-bit):**
- $sb(0) = 1.0$ (i.e., $\log_2(1+1) = 1$)
- $sb(z) \to 0$ as $z \to -\infty$
- Monotonically decreasing for $z \leq 0$
- Smooth, well-behaved — relatively easy to approximate

**db (difference-bit):**
- $db(z) \to -\infty$ as $z \to 0^+$ (singularity! $\log_2(2^z - 1) \to -\infty$)
- $db(z) \to z$ as $z \to +\infty$
- Monotonically increasing
- The singularity near $z = 0$ makes approximation harder

---

## 5. sb/db Implementation Strategies

### Strategy 1: Ideal (math.h)

Enabled by `xlns32_ideal` or `xlns16_ideal`.

```cpp
sb_ideal(z) = log(1 + pow(2.0, z/scale)) / log(2.0) * scale + 0.5
db_ideal(z) = log(pow(2.0, z/scale) - 1) / log(2.0) * scale + 0.5
```

**Pros:** Maximum accuracy.
**Cons:** Very slow — two transcendental function calls per operation.

### Strategy 2: Cotransformation + Linear Interpolation (xlns32, non-ideal)

For **sb** (32-bit):

The `z` value is split into high bits (table index) and low bits (interpolation fraction):

```
z_high = (z >> 14)           →  index into sbhtable/sbltable
z_low  = (z & 0x3FFF)        →  fraction for linear interpolation
```

The table stores `sb` values at evenly-spaced `z_high` points. Between points, linear interpolation is used:

$$sb(z) \approx \text{table}[z_h] + \frac{(\text{table}[z_h] - \text{table}[z_h+1]) \cdot z_l}{2^{14}}$$

The table is split into `sbhtable` (high 16 bits) and `sbltable` (low 16 bits) for memory efficiency.

For **db** (32-bit):

Because `db` has a singularity near 0 and varies more rapidly, a **three-region cotransformation** is used:

```
z is split into three parts: z0 (bits 19-27), z1 (bits 10-18), z2 (bits 0-9)
```

Each region has its own lookup table (`db0table`, `db1table`, `db2table`). The full `db(z)` is reconstructed by combining table lookups with recursive `sb` calls for correction:

$$db(z_0 | z_1 | z_2) = db(z_2) + sb(z_2 + db(z_1) - db(z_2)) + \ldots$$

This "cotransformation" technique is described in Arnold et al. (1998).

### Strategy 3: LPVIP / Mitchell Approximation (xlns16, non-ideal, non-table)

The **Mitchell** approximation is a classic method for cheap logarithmic computation:

$$\text{mitch}(z) = (2^F + (z \bmod 2^F)) \gg (-z \div 2^F)$$

where $F = 7$ is the number of fractional bits.

For **sb** (`xlns16_sb_premit_neg`):
1. Apply a post-condition correction based on the magnitude of `z`
2. Compute the Mitchell approximation of `log(1 + 2^z)`
3. Special case: `z == 0` returns `1 << F` (i.e., 1.0)

For **db** (`xlns16_db_premit_neg`):
1. Apply a pre-condition correction based on the magnitude of `z`
2. Compute the Mitchell approximation
3. Near the singularity ($|z| < 2^F$), fall back to `db_ideal` for accuracy

**Pros:** No lookup tables needed, minimal memory.
**Cons:** Lower accuracy than table-based methods.

### Strategy 4: Direct Table Lookup (xlns16 with `xlns16_table` + `xlns16_alt`)

When both `xlns16_table` and `xlns16_alt` are defined, `sb` and `db` use **direct table lookup** from `xlns16sbtbl[]` and `xlns16dbtbl[]` (stored in `xlns16sbdbtbl.h`):

```cpp
adjustez = usedb ? xlns16dbtbl[non_ez_z] : xlns16sbtbl[non_ez_z];
```

Each table has 1,280 entries (~5 KB each). No interpolation needed — the 16-bit precision is low enough that every possible input has a precomputed answer.

**Pros:** Fastest possible computation.
**Cons:** ~10 KB memory for the tables.

---

## 6. Conversion Between FP and LNS

### Float → LNS (computed)

```cpp
xlns32 fp2xlns32(float x) {
    if (x == 0.0) return xlns32_zero;
    // log₂(|x|) × scale, XOR with logsignmask, set sign if negative
    if (x > 0.0)
        return abs((int)(log(x)/log(2.0) * scale)) ^ logsignmask;
    else
        return ((int)(log(fabs(x))/log(2.0) * scale) | signmask) ^ logsignmask;
}
```

This involves a call to `log()` — a relatively expensive transcendental operation.

### Float → LNS16 (table-based)

When `xlns16_table` is defined, conversion uses a 131,072-entry lookup table indexed by the top 17 bits of the IEEE 754 float representation:

```cpp
inline xlns16 fp2xlns16(float x) {
    return xlns16revcvtbl[(*(unsigned *)&x) >> 15];
}
```

This is an extremely fast constant-time operation — just a type-pun, shift, and array access.

### LNS → Float (computed)

```cpp
float xlns322fp(xlns32 x) {
    if (abs(x) == xlns32_zero) return 0.0;
    float magnitude = pow(2.0, (signed)(abs(x) - logsignmask) / (float)scale);
    return (sign(x)) ? -magnitude : magnitude;
}
```

This involves a call to `pow()` — expensive.

### LNS16 → Float (table-based)

```cpp
inline float xlns162fp(xlns16 x) {
    return xlns16cvtbl[x];  // Direct lookup: all 65536 values precomputed
}
```

Constant-time, ~256 KB table.

---

## 7. Conversion Cache Architecture

The `float2xlns32_()` and `float2xlns16_()` functions (used by the class API for auto-conversion) include a **direct-mapped cache**:

```
┌─────────────────────────────────────────────┐
│  Float input  →  Hash(bytes) → Cache index  │
│                                             │
│  Cache[index].tag == input?                 │
│    YES → return Cache[index].content (HIT)  │
│    NO  → compute, store, return    (MISS)   │
└─────────────────────────────────────────────┘
```

### Hash Function

```cpp
unsigned char * fpbyte = (unsigned char *)(&y);  // Alias float as bytes
int addr = (fpbyte[2]) ^ (fpbyte[3] << 2);       // XOR of mantissa/exponent bytes
addr &= (cachesize - 1);                          // Mask to cache size
```

- **Cache size:** 1024 entries
- **Collision resolution:** None (direct-mapped) — a collision simply evicts the old entry
- **xlns32:** Cache **on** by default (computed conversion is expensive)
- **xlns16:** Cache **off** by default (table lookup is faster than cache overhead)

### Statistics

Global counters track cache performance:
```cpp
long xlns32_hits = 0;   // Cache hits
long xlns32_misses = 0; // Cache misses (includes cold misses and evictions)
```

---

## 8. Comparison and Canonical Form

LNS values cannot be compared directly by their raw bit patterns because the biased log representation doesn't preserve ordering for negative numbers. The `canon` macro converts to a form where standard integer comparison gives the correct result:

```cpp
#define xlns32_canon(x) ((x) ^ (-((x) >> canonshift) | signmask))
```

This works by:
1. `(x) >> 31` — arithmetic right shift produces `0x00000000` (positive) or `0xFFFFFFFF` (negative)
2. `| signmask` — ensures the sign bit is always set in the mask
3. XOR with the original — for positive numbers, flips only the sign bit; for negative numbers, flips all bits except the sign bit

The result is a representation where:
- All negative numbers < all positive numbers
- Within positive numbers, larger log = larger value
- Within negative numbers, the ordering is reversed (as it should be)

---

## 9. Overflow and Special Values

### Overflow Detection

`xlns32_mul` and `xlns32_div` can produce results outside the representable log range. The `xlns32_overflow` function checks the sign of the intermediate `temp` value:

```cpp
inline xlns32 xlns32_overflow(xlns32 x, xlns32 y, xlns32 temp) {
    if (logsignmask & temp)
        return (signmask & (x ^ y));         // Underflow → zero (with correct sign)
    else
        return (signmask & (x ^ y)) | logmask; // Overflow → max magnitude
}
```

The `signmask & temp` check detects whether `temp` went negative (underflow toward zero) or wrapped around (overflow toward infinity).

### Zero Handling

- **Zero representation:** All bits zero (`0x00000000` / `0x0000`).
- **Zero in arithmetic:** All functions explicitly check for zero operands where needed (e.g., conversion functions return 0.0 immediately).
- **Cancellation:** When `z == 0` and signs differ, `xlns32_add` returns `xlns32_zero` immediately, avoiding the db singularity.

### Reciprocal

```cpp
#define xlns32_recip(x) (xlns32_sign(x) | xlns32_abs((~x) + 1))
```

In LNS, $1/x = 2^{-L_x}$. In the biased representation, negating the log is done by bitwise NOT + 1 (two's complement negation) of the magnitude, preserving the sign.

### Square Root

```cpp
#define xlns32_sqrt(x) (xlns32_abs(((xlns32_signed)((x) << 1)) / 4) ^ xlns32_sqrtmask)
```

In LNS, $\sqrt{x} = 2^{L_x / 2}$. Halving the log in the biased representation requires careful manipulation of the bias. The shift-left-by-1, divide-by-4, and XOR-with-sqrtmask sequence achieves this correctly.
