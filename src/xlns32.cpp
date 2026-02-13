// XLNS — 32-bit Logarithmic Number System Implementation
// Copyright (c) 1999–2025 Mark G. Arnold
// SPDX-License-Identifier: MIT
//
// This is the 32-bit LNS implementation source file.
// See include/xlns/xlns32.h for the public API documentation.

#include "xlns/xlns32.h"

// ============================================================================
// Core arithmetic — inline implementations
// ============================================================================

xlns32 xlns32_overflow(xlns32 xlns32_x, xlns32 xlns32_y, xlns32 xlns32_temp)
{
	if (xlns32_logsignmask & xlns32_temp)
	{
		return (xlns32_signmask & (xlns32_x ^ xlns32_y));
	}
	else
	{
		return (xlns32_signmask & (xlns32_x ^ xlns32_y)) | xlns32_logmask;
	}
}

xlns32 xlns32_mul(xlns32 x, xlns32 y)
{
	xlns32 xlns32_temp;
	xlns32_temp = (xlns32_logmask & (x)) + (xlns32_logmask & (y)) - xlns32_logsignmask;
	return (xlns32_signmask & (xlns32_temp)) ? xlns32_overflow(x, y, xlns32_temp)
	                                         : (xlns32_signmask & (x ^ y)) | xlns32_temp;
}

xlns32 xlns32_div(xlns32 x, xlns32 y)
{
	xlns32 xlns32_temp;
	xlns32_temp = (xlns32_logmask & (x)) - (xlns32_logmask & (y)) + xlns32_logsignmask;
	return (xlns32_signmask & (xlns32_temp)) ? xlns32_overflow(x, y, xlns32_temp)
	                                         : (xlns32_signmask & (x ^ y)) | xlns32_temp;
}

// ============================================================================
// Gaussian logarithms — sb and db
// ============================================================================

#ifdef xlns32_ideal

xlns32 xlns32_sb_ideal(xlns32 z)
{
	return ((xlns32)((log(1 + pow(2.0, ((double)z) / xlns32_scale)) / log(2.0)) * xlns32_scale + .5));
}

xlns32 xlns32_db_ideal(xlns32 z)
{
	return ((xlns32_signed)((log(pow(2.0, ((double)z) / xlns32_scale) - 1) / log(2.0)) * xlns32_scale + .5));
}

#else

#include "tables/xlns32tbl.h"

xlns32 xlns32_z, xlns32_zh;

#define xlns32_sb_macro(z) \
( \
((xlns32_zh=(xlns32_z=(z))>>xlns32_zhshift)>=(xlns32_tablesize-1))?xlns32_z: \
( \
 ( (xlns32_z&xlns32_zhmask) \
   +(((xlns32)xlns32_sbhtable[xlns32_zh])<<16)+xlns32_sbltable[xlns32_zh]) \
 +( \
   ( \
    (0x4000-(xlns32_zlmask& \
      ((xlns32_sbltable[xlns32_zh]-xlns32_sbltable[xlns32_zh+1])) \
    )) \
    *(xlns32_z&xlns32_zlmask) \
   )>>xlns32_zhshift \
  ) \
) \
)

xlns32 xlns32_dbtrans3(xlns32 z)
{
	xlns32 z0, z1, z2, temp2;
	z0 = z & xlns32_db0mask;
	z1 = z & xlns32_db1mask;
	z2 = z & xlns32_db2mask;
	if (z1 == 0)
	{
		if (z2 == 0)
		{
			if (z0 == 0)
				return 0;
			else
				return xlns32_db0table[z0 >> xlns32_db0shift];
		}
		else if (z0 == 0)
			return xlns32_db2table[z2];
		else
			return xlns32_db2table[z2] +
			       xlns32_sb(z2 +
			       xlns32_db0table[z0 >> xlns32_db0shift] - xlns32_db2table[z2]);
	}
	else if (z2 == 0)
	{
		if (z0 == 0)
			return xlns32_db1table[z1 >> xlns32_db1shift];
		else
			return xlns32_db1table[z1 >> xlns32_db1shift] +
			       xlns32_sb(z1 +
			       xlns32_db0table[z0 >> xlns32_db0shift] - xlns32_db1table[z1 >> xlns32_db1shift]);
	}
	else
	{
		if (z0 == 0)
			return xlns32_db2table[z2] +
			       xlns32_sb(z2 +
			       xlns32_db1table[z1 >> xlns32_db1shift] - xlns32_db2table[z2]);
		else
		{
			temp2 = xlns32_sb(z1 +
			        xlns32_db0table[z0 >> xlns32_db0shift] - xlns32_db1table[z1 >> xlns32_db1shift]);
			return xlns32_db2table[z2] +
			       xlns32_sb(z2 +
			       xlns32_db1table[z1 >> xlns32_db1shift] +
			       temp2 - xlns32_db2table[z2]);
		}
	}
}

#endif

// ============================================================================
// Addition
// ============================================================================

#ifdef xlns32_alt

xlns32 xlns32_add(xlns32 x, xlns32 y)
{
	xlns32 minxyl, maxxy, xl, yl, usedb;
	xlns32_signed adjust, adjustez;
	xlns32_signed z;
	xl = x & xlns32_logmask;
	yl = y & xlns32_logmask;
	minxyl = (yl > xl) ? xl : yl;
	maxxy  = (xl > yl) ? x  : y;
	z = minxyl - (maxxy & xlns32_logmask);
	usedb = xlns32_signmask & (x ^ y);
#ifdef xlns32_ideal
	float pm1 = usedb ? -1.0 : 1.0;
	adjust = z + ((xlns32_signed)(log(pm1 + pow(2.0, -((double)z) / xlns32_scale)) / log(2.0) * xlns32_scale + .5));
#else
	adjust = usedb ? z + ((xlns32_signed)xlns32_db(-z))
	              : z + ((xlns32_signed)xlns32_sb(-z));
#endif
	adjustez = (z < -xlns32_esszer) ? 0 : adjust;
	return ((z == 0) && usedb) ? xlns32_zero
	                           : xlns32_mul(maxxy, xlns32_logsignmask + adjustez);
}

#else

xlns32 xlns32_add(xlns32 x, xlns32 y)
{
	xlns32 t;
	xlns32_signed z;

	z = (x & xlns32_logmask) - (y & xlns32_logmask);
	if (z < 0)
	{
		z = -z;
		t = x;
		x = y;
		y = t;
	}
	if (xlns32_signmask & (x ^ y))
	{
		if (z == 0)
			return xlns32_zero;
		if (z < xlns32_esszer)
			return xlns32_neg(y + xlns32_db(z));
		else
			return xlns32_neg(y + z);
	}
	else
	{
		return y + xlns32_sb(z);
	}
}

#endif

// ============================================================================
// Float <-> LNS32 conversion
// ============================================================================

xlns32 fp2xlns32(float x)
{
	if (x == 0.0)
		return (xlns32_zero);
	else if (x > 0.0)
		return xlns32_abs((xlns32_signed)((log(x) / log(2.0)) * xlns32_scale))
		       ^ xlns32_logsignmask;
	else
		return (((xlns32_signed)((log(fabs(x)) / log(2.0)) * xlns32_scale))
		       | xlns32_signmask) ^ xlns32_logsignmask;
}

float xlns322fp(xlns32 x)
{
	if (xlns32_abs(x) == xlns32_zero)
		return (0.0);
	else if (xlns32_sign(x))
		return (float)(-pow(2.0, ((double)(((xlns32_signed)(xlns32_abs(x) - xlns32_logsignmask))))
		               / ((float)xlns32_scale)));
	else
		return (float)(+pow(2.0, ((double)(((xlns32_signed)(xlns32_abs(x) - xlns32_logsignmask))))
		               / ((float)xlns32_scale)));
}

// ============================================================================
// Access functions
// ============================================================================

xlns32 xlns32_internal(xlns32_float y)
{
	return y.x;
}

float xlns32_2float(xlns32_float y)
{
	return xlns322fp(y.x);
}

// ============================================================================
// Conversion cache
// ============================================================================

#define xlns32_cachesize 1024
static xlns32 xlns32_cachecontent[xlns32_cachesize];
static float xlns32_cachetag[xlns32_cachesize];
long xlns32_misses = 0;
long xlns32_hits = 0;
#define xlns32_cacheon 1

xlns32_float float2xlns32_(float y)
{
	xlns32_float z;
	unsigned char *fpbyte;
	int addr;
	fpbyte = (unsigned char *)(&y);
	addr = (fpbyte[2]) ^ (fpbyte[3] << 2);
	if ((xlns32_cachetag[addr] == y) && xlns32_cacheon)
	{
		z.x = xlns32_cachecontent[addr];
		xlns32_hits++;
	}
	else
	{
		z.x = fp2xlns32(y);
		xlns32_cachecontent[addr] = z.x;
		xlns32_cachetag[addr] = y;
		xlns32_misses++;
	}
	return z;
}

// ============================================================================
// Stream output
// ============================================================================

std::ostream& operator<<(std::ostream& s, xlns32_float y)
{
	return s << xlns32_2float(y);
}

// ============================================================================
// Unary operators
// ============================================================================

xlns32_float operator-(xlns32_float arg1)
{
	xlns32_float z;
	z.x = xlns32_neg(arg1.x);
	return z;
}

// ============================================================================
// Arithmetic operators (LNS × LNS)
// ============================================================================

xlns32_float operator+(xlns32_float arg1, xlns32_float arg2)
{
	xlns32_float z;
	z.x = xlns32_add(arg1.x, arg2.x);
	return z;
}

xlns32_float operator-(xlns32_float arg1, xlns32_float arg2)
{
	xlns32_float z;
	z.x = xlns32_sub(arg1.x, arg2.x);
	return z;
}

xlns32_float operator*(xlns32_float arg1, xlns32_float arg2)
{
	xlns32_float z;
	z.x = xlns32_mul(arg1.x, arg2.x);
	return z;
}

xlns32_float operator/(xlns32_float arg1, xlns32_float arg2)
{
	xlns32_float z;
	z.x = xlns32_div(arg1.x, arg2.x);
	return z;
}

// ============================================================================
// Arithmetic operators with auto type conversion (float ↔ LNS)
// ============================================================================

xlns32_float operator+(float arg1, xlns32_float arg2) { return float2xlns32_(arg1) + arg2; }
xlns32_float operator+(xlns32_float arg1, float arg2) { return arg1 + float2xlns32_(arg2); }
xlns32_float operator-(float arg1, xlns32_float arg2) { return float2xlns32_(arg1) - arg2; }
xlns32_float operator-(xlns32_float arg1, float arg2) { return arg1 - float2xlns32_(arg2); }
xlns32_float operator*(float arg1, xlns32_float arg2) { return float2xlns32_(arg1) * arg2; }
xlns32_float operator*(xlns32_float arg1, float arg2) { return arg1 * float2xlns32_(arg2); }
xlns32_float operator/(float arg1, xlns32_float arg2) { return float2xlns32_(arg1) / arg2; }
xlns32_float operator/(xlns32_float arg1, float arg2) { return arg1 / float2xlns32_(arg2); }

// ============================================================================
// Comparison operators with type conversion
// ============================================================================

int operator==(xlns32_float arg1, float arg2) { return arg1 == float2xlns32_(arg2); }
int operator!=(xlns32_float arg1, float arg2) { return arg1 != float2xlns32_(arg2); }
int operator<=(xlns32_float arg1, float arg2) { return arg1 <= float2xlns32_(arg2); }
int operator>=(xlns32_float arg1, float arg2) { return arg1 >= float2xlns32_(arg2); }
int operator<(xlns32_float arg1, float arg2)  { return arg1 < float2xlns32_(arg2); }
int operator>(xlns32_float arg1, float arg2)  { return arg1 > float2xlns32_(arg2); }

// ============================================================================
// Compound assignment operators
// ============================================================================

xlns32_float operator+=(xlns32_float& arg1, xlns32_float arg2) { arg1 = arg1 + arg2; return arg1; }
xlns32_float operator+=(xlns32_float& arg1, float arg2) { arg1 = arg1 + float2xlns32_(arg2); return arg1; }
xlns32_float operator-=(xlns32_float& arg1, xlns32_float arg2) { arg1 = arg1 - arg2; return arg1; }
xlns32_float operator-=(xlns32_float& arg1, float arg2) { arg1 = arg1 - float2xlns32_(arg2); return arg1; }
xlns32_float operator*=(xlns32_float& arg1, xlns32_float arg2) { arg1 = arg1 * arg2; return arg1; }
xlns32_float operator*=(xlns32_float& arg1, float arg2) { arg1 = arg1 * float2xlns32_(arg2); return arg1; }
xlns32_float operator/=(xlns32_float& arg1, xlns32_float arg2) { arg1 = arg1 / arg2; return arg1; }
xlns32_float operator/=(xlns32_float& arg1, float arg2) { arg1 = arg1 / float2xlns32_(arg2); return arg1; }

// ============================================================================
// Assignment with type conversion
// ============================================================================

xlns32_float xlns32_float::operator=(float rvalue)
{
	x = float2xlns32_(rvalue).x;
	return *this;
}

// ============================================================================
// Math function overloads
// ============================================================================

xlns32_float sin(xlns32_float x)  { return float2xlns32_(sin(xlns32_2float(x))); }
xlns32_float cos(xlns32_float x)  { return float2xlns32_(cos(xlns32_2float(x))); }
xlns32_float exp(xlns32_float x)  { return float2xlns32_(exp(xlns32_2float(x))); }
xlns32_float log(xlns32_float x)  { return float2xlns32_(log(xlns32_2float(x))); }
xlns32_float atan(xlns32_float x) { return float2xlns32_(atan(xlns32_2float(x))); }

xlns32_float sqrt(xlns32_float x)
{
	xlns32_float result;
	result.x = xlns32_sqrt(x.x);
	return result;
}

xlns32_float abs(xlns32_float x)
{
	xlns32_float result;
	result.x = xlns32_abs(x.x);
	return result;
}
