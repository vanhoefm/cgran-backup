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
/* Generated on Mon Nov 10 20:45:20 EST 2008 */

#include "codelet-rdft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_r2cf -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -n 12 -name r2cf_12 -include r2cf.h */

/*
 * This function contains 38 FP additions, 10 FP multiplications,
 * (or, 30 additions, 2 multiplications, 8 fused multiply/add),
 * 31 stack variables, 2 constants, and 24 memory accesses
 */
#include "r2cf.h"

static void r2cf_12(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
     DK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DK(KP500000000, +0.500000000000000000000000000000000000000000000);
     INT i;
     for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(csr), MAKE_VOLATILE_STRIDE(csi)) {
	  E Tm, T6, Ty, Tp, T5, Tk, Tt, Tb, Tc, Td, T9, Tn;
	  {
	       E T1, Tg, Th, Ti, T4, T2, T3, T7, T8, Tj;
	       T1 = R0[0];
	       T2 = R0[WS(rs, 2)];
	       T3 = R0[WS(rs, 4)];
	       Tg = R1[WS(rs, 1)];
	       Th = R1[WS(rs, 3)];
	       Ti = R1[WS(rs, 5)];
	       T4 = T2 + T3;
	       Tm = T3 - T2;
	       T6 = R0[WS(rs, 3)];
	       Ty = Ti - Th;
	       Tj = Th + Ti;
	       Tp = FNMS(KP500000000, T4, T1);
	       T5 = T1 + T4;
	       T7 = R0[WS(rs, 5)];
	       Tk = FNMS(KP500000000, Tj, Tg);
	       Tt = Tg + Tj;
	       T8 = R0[WS(rs, 1)];
	       Tb = R1[WS(rs, 4)];
	       Tc = R1[0];
	       Td = R1[WS(rs, 2)];
	       T9 = T7 + T8;
	       Tn = T8 - T7;
	  }
	  {
	       E Te, Tz, To, TC;
	       Te = Tc + Td;
	       Tz = Td - Tc;
	       To = Tm - Tn;
	       TC = Tm + Tn;
	       {
		    E Ta, Tq, TA, TB;
		    Ta = T6 + T9;
		    Tq = FNMS(KP500000000, T9, T6);
		    TA = Ty - Tz;
		    TB = Ty + Tz;
		    {
			 E Tf, Tu, Tx, Tr;
			 Tf = FNMS(KP500000000, Te, Tb);
			 Tu = Tb + Te;
			 Tx = Tp - Tq;
			 Tr = Tp + Tq;
			 {
			      E Tv, Tw, Tl, Ts;
			      Tv = T5 + Ta;
			      Cr[WS(csr, 3)] = T5 - Ta;
			      Ci[WS(csi, 4)] = KP866025403 * (TC + TB);
			      Ci[WS(csi, 2)] = KP866025403 * (TB - TC);
			      Tw = Tt + Tu;
			      Ci[WS(csi, 3)] = Tt - Tu;
			      Tl = Tf - Tk;
			      Ts = Tk + Tf;
			      Cr[WS(csr, 1)] = FMA(KP866025403, TA, Tx);
			      Cr[WS(csr, 5)] = FNMS(KP866025403, TA, Tx);
			      Cr[0] = Tv + Tw;
			      Cr[WS(csr, 6)] = Tv - Tw;
			      Cr[WS(csr, 4)] = Tr + Ts;
			      Cr[WS(csr, 2)] = Tr - Ts;
			      Ci[WS(csi, 5)] = FNMS(KP866025403, To, Tl);
			      Ci[WS(csi, 1)] = FMA(KP866025403, To, Tl);
			 }
		    }
	       }
	  }
     }
}

static const kr2c_desc desc = { 12, "r2cf_12", {30, 2, 8, 0}, &GENUS };

void X(codelet_r2cf_12) (planner *p) {
     X(kr2c_register) (p, r2cf_12, &desc);
}

#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_r2cf -compact -variables 4 -pipeline-latency 4 -n 12 -name r2cf_12 -include r2cf.h */

/*
 * This function contains 38 FP additions, 8 FP multiplications,
 * (or, 34 additions, 4 multiplications, 4 fused multiply/add),
 * 21 stack variables, 2 constants, and 24 memory accesses
 */
#include "r2cf.h"

static void r2cf_12(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
     DK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DK(KP500000000, +0.500000000000000000000000000000000000000000000);
     INT i;
     for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(csr), MAKE_VOLATILE_STRIDE(csi)) {
	  E T5, Tp, Tb, Tn, Ty, Tt, Ta, Tq, Tc, Ti, Tz, Tu, Td, To;
	  {
	       E T1, T2, T3, T4;
	       T1 = R0[0];
	       T2 = R0[WS(rs, 2)];
	       T3 = R0[WS(rs, 4)];
	       T4 = T2 + T3;
	       T5 = T1 + T4;
	       Tp = FNMS(KP500000000, T4, T1);
	       Tb = T3 - T2;
	  }
	  {
	       E Tj, Tk, Tl, Tm;
	       Tj = R1[WS(rs, 1)];
	       Tk = R1[WS(rs, 3)];
	       Tl = R1[WS(rs, 5)];
	       Tm = Tk + Tl;
	       Tn = FNMS(KP500000000, Tm, Tj);
	       Ty = Tl - Tk;
	       Tt = Tj + Tm;
	  }
	  {
	       E T6, T7, T8, T9;
	       T6 = R0[WS(rs, 3)];
	       T7 = R0[WS(rs, 5)];
	       T8 = R0[WS(rs, 1)];
	       T9 = T7 + T8;
	       Ta = T6 + T9;
	       Tq = FNMS(KP500000000, T9, T6);
	       Tc = T8 - T7;
	  }
	  {
	       E Te, Tf, Tg, Th;
	       Te = R1[WS(rs, 4)];
	       Tf = R1[0];
	       Tg = R1[WS(rs, 2)];
	       Th = Tf + Tg;
	       Ti = FNMS(KP500000000, Th, Te);
	       Tz = Tg - Tf;
	       Tu = Te + Th;
	  }
	  Cr[WS(csr, 3)] = T5 - Ta;
	  Ci[WS(csi, 3)] = Tt - Tu;
	  Td = KP866025403 * (Tb - Tc);
	  To = Ti - Tn;
	  Ci[WS(csi, 1)] = Td + To;
	  Ci[WS(csi, 5)] = To - Td;
	  {
	       E Tx, TA, Tv, Tw;
	       Tx = Tp - Tq;
	       TA = KP866025403 * (Ty - Tz);
	       Cr[WS(csr, 5)] = Tx - TA;
	       Cr[WS(csr, 1)] = Tx + TA;
	       Tv = T5 + Ta;
	       Tw = Tt + Tu;
	       Cr[WS(csr, 6)] = Tv - Tw;
	       Cr[0] = Tv + Tw;
	  }
	  {
	       E Tr, Ts, TB, TC;
	       Tr = Tp + Tq;
	       Ts = Tn + Ti;
	       Cr[WS(csr, 2)] = Tr - Ts;
	       Cr[WS(csr, 4)] = Tr + Ts;
	       TB = Ty + Tz;
	       TC = Tb + Tc;
	       Ci[WS(csi, 2)] = KP866025403 * (TB - TC);
	       Ci[WS(csi, 4)] = KP866025403 * (TC + TB);
	  }
     }
}

static const kr2c_desc desc = { 12, "r2cf_12", {34, 4, 4, 0}, &GENUS };

void X(codelet_r2cf_12) (planner *p) {
     X(kr2c_register) (p, r2cf_12, &desc);
}

#endif				/* HAVE_FMA */
