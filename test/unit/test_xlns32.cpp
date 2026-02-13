// XLNSCPP — 32-bit Arithmetic Test Suite
// Validates xlns32 operations by comparing with floating-point results.
// Original: xlns32test.cpp by Mark G. Arnold
// Adapted for the new library structure.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <xlns/xlns32.h>

// ============================================================================
// Test 1: Sum of odd numbers — should equal n²
// ============================================================================

void test1fp()
{
	float odd, sum;
	int i;
	odd = 1;
	sum = 0;
	for (i = 1; i <= 10000; i++) {
		sum += odd;
		odd += 2.0;
	}
	printf("test1fp odd=%f sum=%f\n", odd, sum);
}

void test1xlns32()
{
	xlns32 odd, sum, two;
	int i;
	odd = fp2xlns32(1.0);
	sum = fp2xlns32(0.0);
	two = fp2xlns32(2.0);
	for (i = 1; i <= 10000; i++) {
		sum = xlns32_add(sum, odd);
		odd = xlns32_add(odd, two);
	}
	printf("test1xlns32 odd=%f sum=%f\n", xlns322fp(odd), xlns322fp(sum));
}

void test1xlns32_float()
{
	xlns32_float odd, sum, two;
	int i;
	odd = 1.0;
	sum = 0.0;
	two = 2.0;
	for (i = 1; i <= 10000; i++) {
		sum = sum + odd;
		odd = odd + two;
	}
	printf("test1xlns32_float odd=%f sum=%f\n", xlns32_2float(odd), xlns32_2float(sum));
	std::cout << "test1xlns32_float cout odd=" << odd << " sum=" << sum << "\n";
}

// ============================================================================
// Test 2: e approximation via factorial series
// ============================================================================

void test2fp()
{
	float num, fact, sum;
	int i;
	num = 1.0; fact = 1.0; sum = 0.0;
	for (i = 1; i <= 8; i++) {
		sum = sum + 1.0 / fact;
		fact = fact * num;
		num = num + 1.0;
	}
	printf("test2fp1 num=%f fact=%e sum=%f\n", num, fact, sum);
}

void test2xlns32()
{
	xlns32 num, fact, sum, one;
	int i;
	one = fp2xlns32(1.0);
	num = one; fact = one;
	sum = fp2xlns32(0.0);
	for (i = 1; i <= 8; i++) {
		sum = xlns32_add(sum, xlns32_recip(fact));
		fact = xlns32_mul(fact, num);
		num = xlns32_add(num, one);
	}
	printf("test2xlns321 num=%f fact=%e sum=%f\n",
	       xlns322fp(num), xlns322fp(fact), xlns322fp(sum));
}

void test2xlns32_float()
{
	xlns32_float num, fact, sum;
	int i;
	num = 1.0; fact = 1.0; sum = 0.0;
	for (i = 1; i <= 8; i++) {
		sum = sum + 1.0 / fact;
		fact = fact * num;
		num = num + 1.0;
	}
	std::cout << "test2xlns32_float num=" << num << " fact=" << fact << " sum=" << sum << "\n";
}

// ============================================================================
// Test 3: cos(1) approximation via Taylor series
// ============================================================================

void test3fp()
{
	double num, fact, sum;
	int i;
	num = 1.0; fact = 1.0; sum = 0.0;
	for (i = 1; i <= 10; i++) {
		sum = sum + 1.0 / fact;
		fact = -fact * num * (num + 1.0);
		num = num + 2.0;
	}
	printf("test3fp1 num=%f fact=%e sum=%f\n", num, fact, sum);
}

void test3xlns32()
{
	xlns32 num, fact, sum, one, two;
	int i;
	one = fp2xlns32(1.0); num = one; fact = one;
	sum = fp2xlns32(0.0); two = fp2xlns32(2.0);
	for (i = 1; i <= 10; i++) {
		sum = xlns32_add(sum, xlns32_recip(fact));
		fact = xlns32_neg(xlns32_mul(fact, xlns32_mul(num, xlns32_add(num, one))));
		num = xlns32_add(num, two);
	}
	printf("test3xlns321 num=%f fact= sum=%f\n",
	       xlns322fp(num), xlns322fp(sum));
}

void test3xlns32_float()
{
	xlns32_float num, fact, sum;
	int i;
	num = 1.0; fact = 1.0; sum = 0.0;
	for (i = 1; i <= 10; i++) {
		sum = sum + 1.0 / fact;
		fact = -fact * num * (num + 1.0);
		num = num + 2.0;
	}
	std::cout << "test3xlns32_float num=" << num << " fact=" << fact << "sum=" << sum << "\n";
}

// ============================================================================
// Test 5: Leibniz π approximation
// ============================================================================

void test5fp()
{
	float num, val, sum;
	long i;
	num = 1.0; sum = 0.0; val = num;
	for (i = 1; i <= 1000; i++) {
		sum = sum + val / num;
		val = -val;
		num = num + 2.0;
	}
	printf("test5fp num=%f 4*sum=%f\n", num, 4 * sum);
}

void test5xlns32()
{
	xlns32 num, val, sum, two;
	long i;
	two = fp2xlns32(2.0); num = fp2xlns32(1.0);
	sum = fp2xlns32(0.0); val = num;
	for (i = 1; i <= 1000; i++) {
		sum = xlns32_add(sum, xlns32_div(val, num));
		val = xlns32_neg(val);
		num = xlns32_add(num, two);
	}
	printf("test5xlns32 num=%f val=%e 4*sum=%f\n",
	       xlns322fp(num), xlns322fp(val),
	       xlns322fp(xlns32_mul(fp2xlns32(4.0), sum)));
}

void test5xlns32_float()
{
	xlns32_float num, val, sum;
	long i;
	num = 1.0; sum = 0.0; val = num;
	for (i = 1; i <= 1000; i++) {
		sum = sum + val / num;
		val = -val;
		num = num + 2.0;
	}
	std::cout << "test5xlns32_float num=" << num << " 4*sum=" << 4 * sum << "\n";
}

// ============================================================================
// Test: Comparison operators
// ============================================================================

void testcompare()
{
	xlns32_float x[4];
	float f[4];
	int i, j;
	f[0] = -2.0; f[1] = -0.5; f[2] = 0.5; f[3] = 2.0;
	x[0] = -2.0; x[1] = -0.5; x[2] = 0.5; x[3] = 2.0;
	for (i = 0; i <= 3; i++) {
		for (j = 0; j <= 3; j++)
			printf("%d ", f[i] < f[j]);
		printf("    ");
		for (j = 0; j <= 3; j++)
			printf("%d ", x[i] < x[j]);
		printf("\n");
	}
}

// ============================================================================
// main — automated test runner (no interactive prompts)
// ============================================================================

int main(void)
{
	printf("xlns32 C++ (32-bit like float) %ld\n", sizeof(xlns32));

	testcompare();
	test5fp();
	test5xlns32();
	test5xlns32_float();
	test1fp();
	test1xlns32();
	test1xlns32_float();
	test2fp();
	test2xlns32();
	test2xlns32_float();
	test3fp();
	test3xlns32();
	test3xlns32_float();

	printf("\nAll xlns32 tests completed.\n");
	return 0;
}
