// ============================================================================
// XLNSCPP — Basic usage example
// Demonstrates 32-bit and 16-bit Logarithmic Number System arithmetic.
// ============================================================================

#include <iostream>
#include <cmath>
#include <xlns/xlns.h>    // convenience header: includes xlns32 + xlns16

int main()
{
    // -----------------------------------------------------------------------
    // 32-bit LNS
    // -----------------------------------------------------------------------
    std::cout << "===== 32-bit LNS =====\n";

    xlns32_float a, b;
    a = 3.14f;
    b = 2.71f;

    std::cout << "a = " << a << "  (expect ~3.14)\n";
    std::cout << "b = " << b << "  (expect ~2.71)\n";
    std::cout << "a + b = " << (a + b) << "  (expect ~5.85)\n";
    std::cout << "a - b = " << (a - b) << "  (expect ~0.43)\n";
    std::cout << "a * b = " << (a * b) << "  (expect ~8.51)\n";
    std::cout << "a / b = " << (a / b) << "  (expect ~1.16)\n";

    // -----------------------------------------------------------------------
    // 16-bit LNS
    // -----------------------------------------------------------------------
    std::cout << "\n===== 16-bit LNS =====\n";

    xlns16_float x, y;
    x = 3.14f;
    y = 2.71f;

    std::cout << "x = " << x << "  (expect ~3.14)\n";
    std::cout << "y = " << y << "  (expect ~2.71)\n";
    std::cout << "x + y = " << (x + y) << "  (expect ~5.85)\n";
    std::cout << "x * y = " << (x * y) << "  (expect ~8.51)\n";

    // -----------------------------------------------------------------------
    // Low-level API
    // -----------------------------------------------------------------------
    std::cout << "\n===== Low-level API =====\n";

    xlns32 raw_a = fp2xlns32(100.0);
    xlns32 raw_b = fp2xlns32(200.0);
    xlns32 raw_sum = xlns32_add(raw_a, raw_b);
    std::cout << "100 + 200 = " << xlns322fp(raw_sum)
              << "  (expect ~300)\n";

    xlns32 raw_prod = xlns32_mul(raw_a, raw_b);
    std::cout << "100 * 200 = " << xlns322fp(raw_prod)
              << "  (expect ~20000)\n";

    // -----------------------------------------------------------------------
    // Sum of first N odd numbers (should equal N²)
    // -----------------------------------------------------------------------
    std::cout << "\n===== Sum of odds (32-bit) =====\n";
    int N = 100;
    xlns32_float sum;
    sum = 0.0f;
    for (int i = 0; i < N; i++) {
        xlns32_float term;
        term = (float)(2 * i + 1);
        sum = sum + term;
    }

    float expected = (float)(N * N);
    std::cout << "Sum of first " << N << " odd numbers = " << sum
              << "  (expect " << expected << ")\n";

    return 0;
}
