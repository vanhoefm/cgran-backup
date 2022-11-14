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
/* Generated on Mon Nov 10 20:54:56 EST 2008 */

#include "codelet-rdft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_hc2hc -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -sign 1 -n 2 -dif -name hb_2 -include hb.h */

/*
 * This function contains 6 FP additions, 4 FP multiplications,
 * (or, 4 additions, 2 multiplications, 2 fused multiply/add),
 * 11 stack variables, 0 constants, and 8 memory accesses
 */
#include "hb.h"

static void hb_2(R *cr, R *ci, const R *W, stride rs, INT mb, INT me, INT ms)
{
     INT m;
     for (m = mb, W = W + ((mb - 1) * 2); m < me; m = m + 1, cr = cr + ms, ci = ci - ms, W = W + 2, MAKE_VOLATILE_STRIDE(rs)) {
	  E T5, T6, T9, T8, T7, Ta;
	  {
	       E T1, T2, T3, T4;
	       T1 = cr[0];
	       T2 = ci[0];
	       T3 = ci[WS(rs, 1)];
	       T4 = cr[WS(rs, 1)];
	       T5 = W[0];
	       cr[0] = T1 + T2;
	       T6 = T1 - T2;
	       ci[0] = T3 - T4;
	       T9 = T3 + T4;
	       T8 = W[1];
	       T7 = T5 * T6;
	  }
	  Ta = T8 * T6;
	  cr[WS(rs, 1)] = FNMS(T8, T9, T7);
	  ci[WS(rs, 1)] = FMA(T5, T9, Ta);
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 1, 2},
     {TW_NEXT, 1, 0}
};

static const hc2hc_desc desc = { 2, "hb_2", twinstr, &GENUS, {4, 2, 2, 0} };

void X(codelet_hb_2) (planner *p) {
     X(khc2hc_register) (p, hb_2, &desc);
}
#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_hc2hc -compact -variables 4 -pipeline-latency 4 -sign 1 -n 2 -dif -name hb_2 -include hb.h */

/*
 * This function contains 6 FP additions, 4 FP multiplications,
 * (or, 4 additions, 2 multiplications, 2 fused multiply/add),
 * 9 stack variables, 0 constants, and 8 memory accesses
 */
#include "hb.h"

static void hb_2(R *cr, R *ci, const R *W, stride rs, INT mb, INT me, INT ms)
{
     INT m;
     for (m = mb, W = W + ((mb - 1) * 2); m < me; m = m + 1, cr = cr + ms, ci = ci - ms, W = W + 2, MAKE_VOLATILE_STRIDE(rs)) {
	  E T1, T2, T6, T3, T4, T8, T5, T7;
	  T1 = cr[0];
	  T2 = ci[0];
	  T6 = T1 - T2;
	  T3 = ci[WS(rs, 1)];
	  T4 = cr[WS(rs, 1)];
	  T8 = T3 + T4;
	  cr[0] = T1 + T2;
	  ci[0] = T3 - T4;
	  T5 = W[0];
	  T7 = W[1];
	  cr[WS(rs, 1)] = FNMS(T7, T8, T5 * T6);
	  ci[WS(rs, 1)] = FMA(T7, T6, T5 * T8);
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 1, 2},
     {TW_NEXT, 1, 0}
};

static const hc2hc_desc desc = { 2, "hb_2", twinstr, &GENUS, {4, 2, 2, 0} };

void X(codelet_hb_2) (planner *p) {
     X(khc2hc_register) (p, hb_2, &desc);
}
#endif				/* HAVE_FMA */
