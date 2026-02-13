// XLNSCPP — 16-bit Performance Benchmark
// Original: time16test.cpp by Mark G. Arnold
// Measures operations per second for different LNS operations.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ctime>
#include <xlns/xlns16.h>

int main()
{
	float          f[1000000];
	xlns16_float  xf[1000000];
	xlns16      xfxf[1000000];
	xlns16       xsum;
	xlns16_float xfsum;
	float        fsum;
	time_t t1, t2;
	int i, cnt;

	for (i = 0; i < 1000000; i++)
		f[i] = exp(-i / 10000.) * sin(i);

	printf("converting to xlns_float\n");
	time(&t1);
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			xf[i] = float2xlns16_(f[i]);
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("converting back to float\n");
	time(&t1);
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			f[i] = xlns16_2float(xf[i]);
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("converting to xlns\n");
	time(&t1);
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			xfxf[i] = fp2xlns16(f[i]);
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("summing xlns\n");
	time(&t1);
	xsum = xlns16_zero;
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			xsum = xlns16_add(xsum, xfxf[i]);
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("summing xlns_float\n");
	time(&t1);
	xfsum = 0.0;
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			xfsum += xf[i];
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("summing float\n");
	time(&t1);
	fsum = 0.0;
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			fsum += f[i];
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("mul xlns\n");
	time(&t1);
	xsum = fp2xlns16(3.14159);
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			xfxf[i] = xlns16_mul(xsum, xfxf[i]);
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	printf("mul float\n");
	time(&t1);
	fsum = 3.14159;
	for (cnt = 0; cnt < 1000; cnt++)
		for (i = 0; i < 1000000; i++)
			f[i] = fsum * f[i];
	time(&t2);
	printf("time=%ld\n", t2 - t1);

	return 0;
}
