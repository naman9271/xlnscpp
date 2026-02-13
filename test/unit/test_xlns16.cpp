// XLNSCPP — 16-bit Arithmetic Test Suite
// Validates xlns16 operations by comparing with floating-point results.
// Original: xlns16test.cpp by Mark G. Arnold
// Adapted for the new library structure.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <xlns/xlns16.h>

// ============================================================================
// Test 1: Sum of odd numbers (scaled to 100)
// ============================================================================

void test1fp()
{
	float odd, sum;
	int i;
	odd = 1; sum = 0;
	for (i = 1; i <= 100; i++) {
		sum += odd;
		odd += 2.0;
	}
	printf("test1fp odd=%f sum=%f\n", odd, sum);
}

void test1xlns16()
{
	xlns16 odd, sum, two;
	int i;
	odd = fp2xlns16(1.0);
	sum = fp2xlns16(0.0);
	two = fp2xlns16(2.0);
	for (i = 1; i <= 100; i++) {
		sum = xlns16_add(sum, odd);
		odd = xlns16_add(odd, two);
	}
	printf("test1xlns16 odd=%f sum=%f\n", xlns162fp(odd), xlns162fp(sum));
}

void test1xlns16_float()
{
	xlns16_float odd, sum, two;
	int i;
	odd = 1.0; sum = 0.0; two = 2.0;
	for (i = 1; i <= 100; i++) {
		sum = sum + odd;
		odd = odd + two;
	}
	printf("test1xlns16_float odd=%f sum=%f\n", xlns16_2float(odd), xlns16_2float(sum));
	std::cout << "test1xlns16_float cout odd=" << odd << " sum=" << sum << "\n";
}

// ============================================================================
// Test 2: e approximation
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

void test2xlns16()
{
	xlns16 num, fact, sum, one;
	int i;
	one = fp2xlns16(1.0); num = one; fact = one;
	sum = fp2xlns16(0.0);
	for (i = 1; i <= 8; i++) {
		sum = xlns16_add(sum, xlns16_recip(fact));
		fact = xlns16_mul(fact, num);
		num = xlns16_add(num, one);
	}
	printf("test2xlns161 num=%f fact=%e sum=%f\n",
	       xlns162fp(num), xlns162fp(fact), xlns162fp(sum));
}

void test2xlns16_float()
{
	xlns16_float num, fact, sum;
	int i;
	num = 1.0; fact = 1.0; sum = 0.0;
	for (i = 1; i <= 8; i++) {
		sum = sum + 1.0 / fact;
		fact = fact * num;
		num = num + 1.0;
	}
	std::cout << "test2xlns16_float num=" << num << " fact=" << fact << " sum=" << sum << "\n";
}

// ============================================================================
// Test 3: cos(1) approximation
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

void test3xlns16()
{
	xlns16 num, fact, sum, one, two;
	int i;
	one = fp2xlns16(1.0); num = one; fact = one;
	sum = fp2xlns16(0.0); two = fp2xlns16(2.0);
	for (i = 1; i <= 10; i++) {
		sum = xlns16_add(sum, xlns16_recip(fact));
		fact = xlns16_neg(xlns16_mul(fact, xlns16_mul(num, xlns16_add(num, one))));
		num = xlns16_add(num, two);
	}
	printf("test3xlns161 num=%f fact= sum=%f\n",
	       xlns162fp(num), xlns162fp(sum));
}

void test3xlns16_float()
{
	xlns16_float num, fact, sum;
	int i;
	num = 1.0; fact = 1.0; sum = 0.0;
	for (i = 1; i <= 10; i++) {
		sum = sum + 1.0 / fact;
		fact = -fact * num * (num + 1.0);
		num = num + 2.0;
	}
	std::cout << "test3xlns16_float num=" << num << " fact=" << fact << "sum=" << sum << "\n";
}

// ============================================================================
// Test 5: Leibniz π approximation (scaled to 10 terms)
// ============================================================================

void test5fp()
{
	float num, val, sum;
	long i;
	num = 1.0; sum = 0.0; val = num;
	for (i = 1; i <= 10; i++) {
		sum = sum + val / num;
		val = -val;
		num = num + 2.0;
	}
	printf("test5fp num=%f 4*sum=%f\n", num, 4 * sum);
}

void test5xlns16()
{
	xlns16 num, val, sum, two;
	long i;
	two = fp2xlns16(2.0); num = fp2xlns16(1.0);
	sum = fp2xlns16(0.0); val = num;
	for (i = 1; i <= 10; i++) {
		sum = xlns16_add(sum, xlns16_div(val, num));
		val = xlns16_neg(val);
		num = xlns16_add(num, two);
	}
	printf("test5xlns16 num=%f val=%e 4*sum=%f\n",
	       xlns162fp(num), xlns162fp(val),
	       xlns162fp(xlns16_mul(fp2xlns16(4.0), sum)));
}

void test5xlns16_float()
{
	xlns16_float num, val, sum;
	long i;
	num = 1.0; sum = 0.0; val = num;
	for (i = 1; i <= 10; i++) {
		sum = sum + val / num;
		val = -val;
		num = num + 2.0;
	}
	std::cout << "test5xlns16_float num=" << num << " 4*sum=" << 4 * sum << "\n";
}

// ============================================================================
// Test: Comparison operators
// ============================================================================

void testcompare()
{
	xlns16_float x[4];
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
	printf("xlns16 C++ (16-bit like bfloat) %ld\n", sizeof(xlns16));

	testcompare();
	test5fp();
	test5xlns16();
	test5xlns16_float();
	test1fp();
	test1xlns16();
	test1xlns16_float();
	test2fp();
	test2xlns16();
	test2xlns16_float();
	test3fp();
	test3xlns16();
	test3xlns16_float();

	printf("\nAll xlns16 tests completed.\n");
	return 0;
}
