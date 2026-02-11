# Building & Using XLNSCPP

This guide covers how to compile and use the XLNSCPP library in your projects.

---

## Prerequisites

- A C++ compiler supporting C++11 or later: **g++**, **clang++**, or MSVC
- Standard C library with `<math.h>` (always required for conversions; also needed in ideal mode)
- ~1 MB RAM if using lookup tables (`xlns16_table`)

---

## Quick Start

XLNSCPP uses a **header-include** pattern. There is no separate compilation step for the library itself — you `#include` the `.cpp` files directly into your program:

```cpp
// my_program.cpp
#include "xlns32.cpp"   // Pulls in the entire 32-bit LNS implementation

int main() {
    xlns32_float a = 3.14f;
    xlns32_float b = 2.71f;
    std::cout << "a + b = " << (a + b) << std::endl;
    return 0;
}
```

Compile:
```bash
g++ -O2 -o my_program my_program.cpp
```

---

## Compilation Configurations

### 32-bit LNS only (table-based sb/db, the default)

```bash
g++ -O2 -o my_program my_program.cpp
```

Your source file:
```cpp
#include "xlns32.cpp"
```

### 32-bit LNS with ideal (math.h) sb/db

```bash
g++ -O2 -o my_program my_program.cpp
```

Your source file:
```cpp
#define xlns32_ideal
#include "xlns32.cpp"
```

Or use compiler flags:
```bash
g++ -O2 -Dxlns32_ideal -o my_program my_program.cpp
```

### 16-bit LNS with all tables (fastest)

```bash
g++ -O2 -o my_program my_program.cpp
```

Your source file:
```cpp
#define xlns16_alt
#define xlns16_table
#include "xlns16.cpp"
```

### 16-bit LNS with LPVIP (no tables)

```cpp
#define xlns16_alt
#include "xlns16.cpp"
```

### 16-bit LNS with LPVIP and altopt (slightly less accurate, fewer ops)

```cpp
#define xlns16_alt
#define xlns16_altopt
#include "xlns16.cpp"
```

### Both 16-bit and 32-bit together

```cpp
#define xlns32_ideal
#include "xlns32.cpp"
#define xlns16_ideal
#include "xlns16.cpp"
```

**Important:** Both libraries can coexist in the same translation unit because all symbols are prefixed with `xlns16_` or `xlns32_`.

---

## Building the Test Programs

### xlns32test — 32-bit arithmetic tests

```bash
# Default (table-based sb/db)
g++ -O2 -o xlns32test xlns32test.cpp

# With ideal sb/db
g++ -O2 -Dxlns32_ideal -o xlns32test xlns32test.cpp
```

### xlns16test — 16-bit arithmetic tests

```bash
# Default (LPVIP with altopt)
g++ -O2 -o xlns16test xlns16test.cpp

# With all tables
g++ -O2 -Dxlns16_alt -Dxlns16_table -o xlns16test xlns16test.cpp

# With ideal
g++ -O2 -Dxlns16_ideal -o xlns16test xlns16test.cpp

# Automated regression testing across all configurations
for case in 0 1 2 3 4 5; do
    echo "=== Case $case ==="
    g++ -O2 -Dxlns16case=$case -o xlns16test_case$case xlns16test.cpp
    echo "0 0" | ./xlns16test_case$case
done
```

### xlnsbothtest — Both 16-bit and 32-bit together

```bash
g++ -O2 -o xlnsbothtest xlnsbothtest.cpp
```

### time16test — Performance benchmark

```bash
# Recommended: all tables for meaningful benchmark
g++ -O2 -o time16test time16test.cpp
```

### Function test programs

```bash
# Interactive 32-bit function tester
g++ -O2 -o xlns32funtest xlns32funtest.cpp

# Interactive 16-bit function tester
g++ -O2 -o xlns16funtest xlns16funtest.cpp
```

### Gaussian log cross-validation programs (sb/db)

These are used with companion Python scripts to verify that C++ implementations match the Python xlns library.

```bash
# Ideal sb (used by sbtest.py)
g++ -O2 -o sb16 sb16.cpp

# Ideal db (used by dbtest.py)
g++ -O2 -o db16 db16.cpp

# LPVIP sb (used by sblptest.py)
g++ -O2 -o sbmit16 sbmit16.cpp

# LPVIP db (used by dblptest.py)
g++ -O2 -o dbmit16 dbmit16.cpp
```

---

## Usage Examples

### Example 1: Sum of Odd Numbers

Compute $\sum_{i=0}^{n-1} (2i+1) = n^2$ using the function API:

```cpp
#include "xlns32.cpp"

int main() {
    xlns32 odd = fp2xlns32(1.0);
    xlns32 sum = fp2xlns32(0.0);
    xlns32 two = fp2xlns32(2.0);

    for (int i = 1; i <= 10000; i++) {
        sum = xlns32_add(sum, odd);
        odd = xlns32_add(odd, two);
    }
    printf("sum = %f (expected 100000000)\n", xlns322fp(sum));
    return 0;
}
```

### Example 2: Leibniz Formula for π

Compute $\pi \approx 4 \sum_{k=0}^{n} \frac{(-1)^k}{2k+1}$ using operator overloading:

```cpp
#include "xlns32.cpp"

int main() {
    xlns32_float num = 1.0f;
    xlns32_float sum = 0.0f;
    xlns32_float val = 1.0f;

    for (long i = 1; i <= 1000; i++) {
        sum = sum + val / num;
        val = -val;
        num = num + 2.0f;
    }
    std::cout << "π ≈ " << 4.0f * sum << std::endl;
    return 0;
}
```

### Example 3: Mixed-mode Mandelbrot

```cpp
#include "xlns32.cpp"

int main() {
    xlns32_float x, y, x1, y1, xnew, ynew;
    int count, iter = 2000;

    // Use natural C++ syntax — float literals auto-convert
    y1 = 0.3f;
    x1 = -0.5f;
    x = x1;
    y = y1;
    count = 0;
    while ((x*x + y*y < 4.0f) && (count < iter)) {
        xnew = x*x - y*y + x1;
        ynew = x*y * 2.0f + y1;
        count++;
        x = xnew;
        y = ynew;
    }
    std::cout << "Escaped after " << count << " iterations" << std::endl;
    return 0;
}
```

### Example 4: Inspecting Internal Representation

```cpp
#include "xlns32.cpp"
#include <iomanip>

int main() {
    xlns32_float x = 3.14f;
    std::cout << "Value: " << x << std::endl;
    std::cout << "Internal (hex): " << std::hex << xlns32_internal(x) << std::endl;
    std::cout << "Back to float: " << xlns32_2float(x) << std::endl;
    return 0;
}
```

### Example 5: Using 16-bit with Tables

```cpp
#define xlns16_alt
#define xlns16_table
#include "xlns16.cpp"

int main() {
    xlns16_float a = 1.5f;
    xlns16_float b = 2.5f;
    xlns16_float c = a * b;   // Multiplication: integer add (cheap!)
    xlns16_float d = a + b;   // Addition: table lookup (fast with xlns16_table)

    std::cout << "1.5 * 2.5 = " << c << std::endl;
    std::cout << "1.5 + 2.5 = " << d << std::endl;
    return 0;
}
```

---

## Integration Into Your Project

Since XLNSCPP uses the header-include pattern, there are two approaches:

### Approach A: Direct Include (simplest)

Copy the files you need into your project and `#include` them:

```
your_project/
├── third_party/
│   └── xlnscpp/
│       ├── xlns16.cpp
│       ├── xlns32.cpp
│       ├── xlns16sbdbtbl.h
│       ├── xlns16cvtbl.h
│       ├── xlns16revcvtbl.h
│       ├── xlns32tbl.h
│       └── ...
└── src/
    └── main.cpp      ←  #include "../third_party/xlnscpp/xlns32.cpp"
```

### Approach B: Include Path

Add the xlnscpp directory to your include path:

```bash
g++ -O2 -I/path/to/xlnscpp -o my_program my_program.cpp
```

Then in your code:
```cpp
#include "xlns32.cpp"
```

### Approach C: Git Submodule

```bash
cd your_project
git submodule add https://github.com/xlnsresearch/xlnscpp.git third_party/xlnscpp
```

---

## Optimization Flags

The following compiler flags are recommended for performance:

```bash
# GCC / Clang
g++ -O2 -march=native -o my_program my_program.cpp

# For even more aggressive optimization
g++ -O3 -march=native -ffast-math -o my_program my_program.cpp
```

**Note:** `-ffast-math` can affect the accuracy of the `ideal` mode since it changes the behavior of `log()` and `pow()`. Only use it if you're using table-based modes.

---

## Troubleshooting

### "Multiple definition" errors when including in multiple translation units

Since `xlns16.cpp` and `xlns32.cpp` define global variables and functions, they can only be included **once** per executable. If you need to use LNS in multiple `.cpp` files, include it in exactly one file and use `extern` declarations in the others. Alternatively, create a wrapper header.

### Large binary size with tables

The table files add significant data:
- `xlns16cvtbl.h`: 65,536 floats ≈ 256 KB
- `xlns16revcvtbl.h`: 131,072 uint16_t ≈ 256 KB
- `xlns16sbdbtbl.h`: 2,560 int16_t ≈ 5 KB
- `xlns32tbl.h`: ~13,288 entries × 2 tables ≈ 100 KB

Total with all tables: ~620 KB of static data.

### Interactive tests block on `scanf`

Some test programs (like `xlns32test`) call `scanf()` between tests. Press Enter to advance. To quit `testops`, enter `0 0`.
