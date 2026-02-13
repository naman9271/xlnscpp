// XLNS — 16-bit Logarithmic Number System Implementation
// Copyright (c) 1999–2025 Mark G. Arnold
// SPDX-License-Identifier: MIT
//
// This is the 16-bit LNS implementation source file.
// See include/xlns/xlns16.h for the public API documentation.

#include "xlns/xlns16.h"

// ============================================================================
// Core arithmetic — inline implementations
// ============================================================================

xlns16 xlns16_overflow(xlns16 xlns16_x, xlns16 xlns16_y, xlns16 xlns16_temp)
{
	if (xlns16_logsignmask & xlns16_temp)
	{
		return (xlns16_signmask & (xlns16_x ^ xlns16_y));
	}
	else
	{
		return (xlns16_signmask & (xlns16_x ^ xlns16_y)) | xlns16_logmask;
	}
}

xlns16 xlns16_mul(xlns16 x, xlns16 y)
{
	xlns16 xlns16_temp;
	xlns16_temp = (xlns16_logmask & (x)) + (xlns16_logmask & (y)) - xlns16_logsignmask;
	return (xlns16_signmask & (xlns16_temp)) ? xlns16_overflow(x, y, xlns16_temp)
	                                         : (xlns16_signmask & (x ^ y)) | xlns16_temp;
}

xlns16 xlns16_div(xlns16 x, xlns16 y)
{
	xlns16 xlns16_temp;
	xlns16_temp = (xlns16_logmask & (x)) - (xlns16_logmask & (y)) + xlns16_logsignmask;
	return (xlns16_signmask & (xlns16_temp)) ? xlns16_overflow(x, y, xlns16_temp)
	                                         : (xlns16_signmask & (x ^ y)) | xlns16_temp;
}

// ============================================================================
// Gaussian logarithms — sb and db
// ============================================================================

#ifdef xlns16_ideal

xlns16 xlns16_sb_ideal(xlns16_signed z)
{
	return ((xlns16)((log(1 + pow(2.0, ((double)z) / xlns16_scale)) / log(2.0)) * xlns16_scale + .5));
}

xlns16 xlns16_db_ideal(xlns16_signed z)
{
	return ((xlns16_signed)((log(pow(2.0, ((double)z) / xlns16_scale) - 1) / log(2.0)) * xlns16_scale + .5));
}

#else

xlns16 xlns16_db_ideal(xlns16_signed z)  // only for singularity
{
	return ((xlns16_signed)((log(pow(2.0, ((double)z) / xlns16_scale) - 1) / log(2.0)) * xlns16_scale + .5));
}

xlns16 xlns16_mitch(xlns16 z)
{
	return (((1 << xlns16_F) + (z & ((1 << xlns16_F) - 1))) >> (-(z >> xlns16_F)));
}

xlns16 xlns16_sb_premit_neg(xlns16_signed zi)  // assumes zi<=0
{
	xlns16 postcond;
	xlns16 z;
	postcond = (zi <= -(3 << xlns16_F)) ? 0 : (zi >= -(3 << (xlns16_F - 2)) ? -1 : +1);
	z = ((zi << 3) + (zi ^ 0xffff) + 16) >> 3;
	return (zi == 0) ? 1 << xlns16_F : xlns16_mitch(z) + postcond;
}

xlns16 xlns16_db_premit_neg(xlns16_signed z)  // assumes zi<0
{
	xlns16_signed precond;
	precond = (z < -(2 << xlns16_F)) ?
	               5 << (xlns16_F - 3) :                 //  0.625
	               (z >> 2) + (9 << (xlns16_F - 3));     //  .25*zr + 9/8
	return (-z >= 1 << xlns16_F) ? -xlns16_mitch(z + precond) : xlns16_db_ideal(-z) + z;  // use ideal for singularity
}

xlns16 xlns16_sb_premit(xlns16_signed zi)  // assumes zi>=0
{
	return xlns16_sb_premit_neg(-zi) + zi;
}

xlns16 xlns16_db_premit(xlns16_signed z)  // assumes zi>0
{
	return xlns16_db_premit_neg(-z) + z;
}

#endif

// ============================================================================
// Addition
// ============================================================================

#ifdef xlns16_alt
 #ifdef xlns16_table
  #include "tables/xlns16sbdbtbl.h"
 #endif

xlns16 xlns16_add(xlns16 x, xlns16 y)
{
	xlns16 minxyl, maxxy, xl, yl, usedb, adjust, adjustez;
	xlns16_signed z;
	xl = x & xlns16_logmask;
	yl = y & xlns16_logmask;
	minxyl = (yl > xl) ? xl : yl;
	maxxy  = (xl > yl) ? x  : y;
	z = minxyl - (maxxy & xlns16_logmask);
	usedb = xlns16_signmask & (x ^ y);
#ifdef xlns16_ideal
	float pm1 = usedb ? -1.0 : 1.0;
	adjust = z + ((xlns16_signed)(log(pm1 + pow(2.0, -((double)z) / xlns16_scale)) / log(2.0) * xlns16_scale + .5));
	adjustez = (z < -xlns16_esszer) ? 0 : adjust;
#else
 #ifdef xlns16_table
	xlns16_signed non_ez_z = (z <= -xlns16_esszer) ? xlns16_esszer - 1 : -z;
	adjustez = usedb ? xlns16dbtbl[non_ez_z]
	                 : xlns16sbtbl[non_ez_z];
 #else
  #ifdef xlns16_altopt
	xlns16_signed precond = (usedb == 0) ? ((-z) >> 3) :           // -.125*z
	             (z < -(2 << xlns16_F)) ? 5 << (xlns16_F - 3) :   //  0.625
	                               (z >> 2) + (9 << (xlns16_F - 3)); //  .25*z + 9/8
	xlns16_signed postcond = (z <= -(3 << xlns16_F)) ? 0 :
	                     z >= -(3 << (xlns16_F - 2)) ? -(1 << (xlns16_F - 6)) :
	                                                   +(1 << (xlns16_F - 6));
	xlns16_signed mitch = (-z >= 1 << xlns16_F) || (usedb == 0) ? xlns16_mitch(z + precond)
	                                                             : -xlns16_db_ideal(-z) - z;  // use ideal for singularity
	adjust = usedb ? -mitch : (z == 0) ? 1 << xlns16_F : mitch + postcond;
  #else
	adjust = usedb ? xlns16_db_premit_neg(z)
	              : xlns16_sb_premit_neg(z);
  #endif
	adjustez = (z < -xlns16_esszer) ? 0 : adjust;
 #endif
#endif
	return ((z == 0) && usedb) ? xlns16_zero
	                           : xlns16_mul(maxxy, xlns16_logsignmask + adjustez);
}

#else

xlns16 xlns16_add(xlns16 x, xlns16 y)
{
	xlns16 t;
	xlns16_signed z;

	z = (x & xlns16_logmask) - (y & xlns16_logmask);
	if (z < 0)
	{
		z = -z;
		t = x;
		x = y;
		y = t;
	}
	if (xlns16_signmask & (x ^ y))
	{
		if (z == 0)
			return xlns16_zero;
		if (z < xlns16_esszer)
			return xlns16_neg(y + xlns16_db(z));
		else
			return xlns16_neg(y + z);
	}
	else
	{
		return y + xlns16_sb(z);
	}
}

#endif

// ============================================================================
// Float <-> LNS16 conversion
// ============================================================================

#ifdef xlns16_table

#include "tables/xlns16revcvtbl.h"

xlns16 fp2xlns16(float x)
{
	return xlns16revcvtbl[(*(unsigned *)&x) >> 15];
}

#include "tables/xlns16cvtbl.h"

float xlns162fp(xlns16 x)
{
	return xlns16cvtbl[x];
}

#else

xlns16 fp2xlns16(float x)
{
	if (x == 0.0)
		return (xlns16_zero);
	else if (x > 0.0)
		return xlns16_abs((xlns16_signed)((log(x) / log(2.0)) * xlns16_scale))
		       ^ xlns16_logsignmask;
	else
		return (((xlns16_signed)((log(fabs(x)) / log(2.0)) * xlns16_scale))
		       | xlns16_signmask) ^ xlns16_logsignmask;
}

float xlns162fp(xlns16 x)
{
	if (xlns16_abs(x) == xlns16_zero)
		return (0.0);
	else if (xlns16_sign(x))
		return (float)(-pow(2.0, ((double)(((xlns16_signed)(xlns16_abs(x) - xlns16_logsignmask))))
		               / ((float)xlns16_scale)));
	else
		return (float)(+pow(2.0, ((double)(((xlns16_signed)(xlns16_abs(x) - xlns16_logsignmask))))
		               / ((float)xlns16_scale)));
}

#endif

// ============================================================================
// Access functions
// ============================================================================

xlns16 xlns16_internal(xlns16_float y)
{
	return y.x;
}

float xlns16_2float(xlns16_float y)
{
	return xlns162fp(y.x);
}

// ============================================================================
// Conversion cache
// ============================================================================

#define xlns16_cachesize 1024
static xlns16 xlns16_cachecontent[xlns16_cachesize];
static float xlns16_cachetag[xlns16_cachesize];
long xlns16_misses = 0;
long xlns16_hits = 0;
#define xlns16_cacheon 0  // off for table

xlns16_float float2xlns16_(float y)
{
	xlns16_float z;
	unsigned char *fpbyte;
	int addr;
	fpbyte = (unsigned char *)(&y);
	addr = (fpbyte[2]) ^ (fpbyte[3] << 2);
	if ((xlns16_cachetag[addr] == y) && xlns16_cacheon)
	{
		z.x = xlns16_cachecontent[addr];
		xlns16_hits++;
	}
	else
	{
		z.x = fp2xlns16(y);
		xlns16_cachecontent[addr] = z.x;
		xlns16_cachetag[addr] = y;
		xlns16_misses++;
	}
	return z;
}

// ============================================================================
// Stream output
// ============================================================================

std::ostream& operator<<(std::ostream& s, xlns16_float y)
{
	return s << xlns16_2float(y);
}

// ============================================================================
// Unary operators
// ============================================================================

xlns16_float operator-(xlns16_float arg1)
{
	xlns16_float z;
	z.x = xlns16_neg(arg1.x);
	return z;
}

// ============================================================================
// Arithmetic operators (LNS × LNS)
// ============================================================================

xlns16_float operator+(xlns16_float arg1, xlns16_float arg2)
{
	xlns16_float z;
	z.x = xlns16_add(arg1.x, arg2.x);
	return z;
}

xlns16_float operator-(xlns16_float arg1, xlns16_float arg2)
{
	xlns16_float z;
	z.x = xlns16_sub(arg1.x, arg2.x);
	return z;
}

xlns16_float operator*(xlns16_float arg1, xlns16_float arg2)
{
	xlns16_float z;
	z.x = xlns16_mul(arg1.x, arg2.x);
	return z;
}

xlns16_float operator/(xlns16_float arg1, xlns16_float arg2)
{
	xlns16_float z;
	z.x = xlns16_div(arg1.x, arg2.x);
	return z;
}

// ============================================================================
// Arithmetic operators with auto type conversion (float ↔ LNS)
// ============================================================================

xlns16_float operator+(float arg1, xlns16_float arg2) { return float2xlns16_(arg1) + arg2; }
xlns16_float operator+(xlns16_float arg1, float arg2) { return arg1 + float2xlns16_(arg2); }
xlns16_float operator-(float arg1, xlns16_float arg2) { return float2xlns16_(arg1) - arg2; }
xlns16_float operator-(xlns16_float arg1, float arg2) { return arg1 - float2xlns16_(arg2); }
xlns16_float operator*(float arg1, xlns16_float arg2) { return float2xlns16_(arg1) * arg2; }
xlns16_float operator*(xlns16_float arg1, float arg2) { return arg1 * float2xlns16_(arg2); }
xlns16_float operator/(float arg1, xlns16_float arg2) { return float2xlns16_(arg1) / arg2; }
xlns16_float operator/(xlns16_float arg1, float arg2) { return arg1 / float2xlns16_(arg2); }

// ============================================================================
// Comparison operators with type conversion
// ============================================================================

int operator==(xlns16_float arg1, float arg2) { return arg1 == float2xlns16_(arg2); }
int operator!=(xlns16_float arg1, float arg2) { return arg1 != float2xlns16_(arg2); }
int operator<=(xlns16_float arg1, float arg2) { return arg1 <= float2xlns16_(arg2); }
int operator>=(xlns16_float arg1, float arg2) { return arg1 >= float2xlns16_(arg2); }
int operator<(xlns16_float arg1, float arg2)  { return arg1 < float2xlns16_(arg2); }
int operator>(xlns16_float arg1, float arg2)  { return arg1 > float2xlns16_(arg2); }

// ============================================================================
// Compound assignment operators
// ============================================================================

xlns16_float operator+=(xlns16_float& arg1, xlns16_float arg2) { arg1 = arg1 + arg2; return arg1; }
xlns16_float operator+=(xlns16_float& arg1, float arg2) { arg1 = arg1 + float2xlns16_(arg2); return arg1; }
xlns16_float operator-=(xlns16_float& arg1, xlns16_float arg2) { arg1 = arg1 - arg2; return arg1; }
xlns16_float operator-=(xlns16_float& arg1, float arg2) { arg1 = arg1 - float2xlns16_(arg2); return arg1; }
xlns16_float operator*=(xlns16_float& arg1, xlns16_float arg2) { arg1 = arg1 * arg2; return arg1; }
xlns16_float operator*=(xlns16_float& arg1, float arg2) { arg1 = arg1 * float2xlns16_(arg2); return arg1; }
xlns16_float operator/=(xlns16_float& arg1, xlns16_float arg2) { arg1 = arg1 / arg2; return arg1; }
xlns16_float operator/=(xlns16_float& arg1, float arg2) { arg1 = arg1 / float2xlns16_(arg2); return arg1; }

// ============================================================================
// Assignment with type conversion
// ============================================================================

xlns16_float xlns16_float::operator=(float rvalue)
{
	x = float2xlns16_(rvalue).x;
	return *this;
}

// ============================================================================
// Math function overloads
// ============================================================================

xlns16_float sin(xlns16_float x)  { return float2xlns16_(sin(xlns16_2float(x))); }
xlns16_float cos(xlns16_float x)  { return float2xlns16_(cos(xlns16_2float(x))); }
xlns16_float exp(xlns16_float x)  { return float2xlns16_(exp(xlns16_2float(x))); }
xlns16_float log(xlns16_float x)  { return float2xlns16_(log(xlns16_2float(x))); }
xlns16_float atan(xlns16_float x) { return float2xlns16_(atan(xlns16_2float(x))); }

xlns16_float sqrt(xlns16_float x)
{
	xlns16_float result;
	result.x = xlns16_sqrt(x.x);
	return result;
}

xlns16_float abs(xlns16_float x)
{
	xlns16_float result;
	result.x = xlns16_abs(x.x);
	return result;
}
