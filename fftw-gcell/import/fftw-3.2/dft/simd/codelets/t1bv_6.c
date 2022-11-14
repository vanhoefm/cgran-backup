/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Mon Nov 10 20:41:06 EST 2008 */

#include "codelet-dft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_twiddle_c -fma -reorder-insns -schedule-for-pipeline -simd -compact -variables 4 -pipeline-latency 8 -n 6 -name t1bv_6 -include t1b.h -sign 1 */

/*
 * This function contains 23 FP additions, 18 FP multiplications,
 * (or, 17 additions, 12 multiplications, 6 fused multiply/add),
 * 27 stack variables, 2 constants, and 12 memory accesses
 */
#include "t1b.h"

static void t1bv_6(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
     DVK(KP866025403, +0.866025403784438646763723170752936183471402627);
     INT m;
     R *x;
     x = ii;
     for (m = mb, W = W + (mb * ((TWVL / VL) * 10)); m < me; m = m + VL, x = x + (VL * ms), W = W + (TWVL * 10), MAKE_VOLATILE_STRIDE(rs)) {
	  V T1, T2, Ta, Tc, T5, T7;
	  T1 = LD(&(x[0]), ms, &(x[0]));
	  T2 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
	  Ta = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
	  Tc = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
	  T5 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
	  T7 = LD(&(x[WS(rs, 5)]), ms, &(x[WS(rs, 1)]));
	  {
	       V T3, Tb, Td, T6, T8;
	       T3 = BYTW(&(W[TWVL * 4]), T2);
	       Tb = BYTW(&(W[TWVL * 6]), Ta);
	       Td = BYTW(&(W[0]), Tc);
	       T6 = BYTW(&(W[TWVL * 2]), T5);
	       T8 = BYTW(&(W[TWVL * 8]), T7);
	       {
		    V Ti, T4, Tk, Te, Tj, T9;
		    Ti = VADD(T1, T3);
		    T4 = VSUB(T1, T3);
		    Tk = VADD(Tb, Td);
		    Te = VSUB(Tb, Td);
		    Tj = VADD(T6, T8);
		    T9 = VSUB(T6, T8);
		    {
			 V Tl, Tn, Tf, Th, Tm, Tg;
			 Tl = VADD(Tj, Tk);
			 Tn = VMUL(LDK(KP866025403), VSUB(Tj, Tk));
			 Tf = VADD(T9, Te);
			 Th = VMUL(LDK(KP866025403), VSUB(T9, Te));
			 ST(&(x[0]), VADD(Ti, Tl), ms, &(x[0]));
			 Tm = VFNMS(LDK(KP500000000), Tl, Ti);
			 ST(&(x[WS(rs, 3)]), VADD(T4, Tf), ms, &(x[WS(rs, 1)]));
			 Tg = VFNMS(LDK(KP500000000), Tf, T4);
			 ST(&(x[WS(rs, 4)]), VFMAI(Tn, Tm), ms, &(x[0]));
			 ST(&(x[WS(rs, 2)]), VFNMSI(Tn, Tm), ms, &(x[0]));
			 ST(&(x[WS(rs, 5)]), VFNMSI(Th, Tg), ms, &(x[WS(rs, 1)]));
			 ST(&(x[WS(rs, 1)]), VFMAI(Th, Tg), ms, &(x[WS(rs, 1)]));
		    }
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     VTW(0, 4),
     VTW(0, 5),
     {TW_NEXT, VL, 0}
};

static const ct_desc desc = { 6, "t1bv_6", twinstr, &GENUS, {17, 12, 6, 0}, 0, 0, 0 };

void X(codelet_t1bv_6) (planner *p) {
     X(kdft_dit_register) (p, t1bv_6, &desc);
}
#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_twiddle_c -simd -compact -variables 4 -pipeline-latency 8 -n 6 -name t1bv_6 -include t1b.h -sign 1 */

/*
 * This function contains 23 FP additions, 14 FP multiplications,
 * (or, 21 additions, 12 multiplications, 2 fused multiply/add),
 * 19 stack variables, 2 constants, and 12 memory accesses
 */
#include "t1b.h"

static void t1bv_6(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
     DVK(KP866025403, +0.866025403784438646763723170752936183471402627);
     INT m;
     R *x;
     x = ii;
     for (m = mb, W = W + (mb * ((TWVL / VL) * 10)); m < me; m = m + VL, x = x + (VL * ms), W = W + (TWVL * 10), MAKE_VOLATILE_STRIDE(rs)) {
	  V Tf, Ti, Ta, Tk, T5, Tj, Tc, Te, Td;
	  Tc = LD(&(x[0]), ms, &(x[0]));
	  Td = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
	  Te = BYTW(&(W[TWVL * 4]), Td);
	  Tf = VSUB(Tc, Te);
	  Ti = VADD(Tc, Te);
	  {
	       V T7, T9, T6, T8;
	       T6 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
	       T7 = BYTW(&(W[TWVL * 6]), T6);
	       T8 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
	       T9 = BYTW(&(W[0]), T8);
	       Ta = VSUB(T7, T9);
	       Tk = VADD(T7, T9);
	  }
	  {
	       V T2, T4, T1, T3;
	       T1 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
	       T2 = BYTW(&(W[TWVL * 2]), T1);
	       T3 = LD(&(x[WS(rs, 5)]), ms, &(x[WS(rs, 1)]));
	       T4 = BYTW(&(W[TWVL * 8]), T3);
	       T5 = VSUB(T2, T4);
	       Tj = VADD(T2, T4);
	  }
	  {
	       V Tb, Tg, Th, Tn, Tl, Tm;
	       Tb = VBYI(VMUL(LDK(KP866025403), VSUB(T5, Ta)));
	       Tg = VADD(T5, Ta);
	       Th = VFNMS(LDK(KP500000000), Tg, Tf);
	       ST(&(x[WS(rs, 1)]), VADD(Tb, Th), ms, &(x[WS(rs, 1)]));
	       ST(&(x[WS(rs, 3)]), VADD(Tf, Tg), ms, &(x[WS(rs, 1)]));
	       ST(&(x[WS(rs, 5)]), VSUB(Th, Tb), ms, &(x[WS(rs, 1)]));
	       Tn = VBYI(VMUL(LDK(KP866025403), VSUB(Tj, Tk)));
	       Tl = VADD(Tj, Tk);
	       Tm = VFNMS(LDK(KP500000000), Tl, Ti);
	       ST(&(x[WS(rs, 2)]), VSUB(Tm, Tn), ms, &(x[0]));
	       ST(&(x[0]), VADD(Ti, Tl), ms, &(x[0]));
	       ST(&(x[WS(rs, 4)]), VADD(Tn, Tm), ms, &(x[0]));
	  }
     }
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     VTW(0, 4),
     VTW(0, 5),
     {TW_NEXT, VL, 0}
};

static const ct_desc desc = { 6, "t1bv_6", twinstr, &GENUS, {21, 12, 2, 0}, 0, 0, 0 };

void X(codelet_t1bv_6) (planner *p) {
     X(kdft_dit_register) (p, t1bv_6, &desc);
}
#endif				/* HAVE_FMA */
