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
/* Generated on Mon Nov 10 20:45:25 EST 2008 */

#include "codelet-rdft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_r2cf -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -n 16 -name r2cf_16 -include r2cf.h */

/*
 * This function contains 58 FP additions, 20 FP multiplications,
 * (or, 38 additions, 0 multiplications, 20 fused multiply/add),
 * 38 stack variables, 3 constants, and 32 memory accesses
 */
#include "r2cf.h"

static void r2cf_16(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
     DK(KP923879532, +0.923879532511286756128183189396788286822416626);
     DK(KP707106781, +0.707106781186547524400844362104849039284835938);
     DK(KP414213562, +0.414213562373095048801688724209698078569671875);
     INT i;
     for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(csr), MAKE_VOLATILE_STRIDE(csi)) {
	  E TQ, TP;
	  {
	       E TB, TN, Tf, T7, Te, Tv, TO, TE, Tq, TJ, Tp, TI, TT, Ty, Tm;
	       E Tr, TK, Ts;
	       {
		    E TC, Ta, Td, TD;
		    {
			 E T1, T2, T4, T5;
			 T1 = R0[0];
			 T2 = R0[WS(rs, 4)];
			 T4 = R0[WS(rs, 2)];
			 T5 = R0[WS(rs, 6)];
			 {
			      E T8, T3, T6, T9, Tb, Tc;
			      T8 = R0[WS(rs, 1)];
			      TB = T1 - T2;
			      T3 = T1 + T2;
			      TN = T4 - T5;
			      T6 = T4 + T5;
			      T9 = R0[WS(rs, 5)];
			      Tb = R0[WS(rs, 7)];
			      Tc = R0[WS(rs, 3)];
			      Tf = T3 - T6;
			      T7 = T3 + T6;
			      TC = T8 - T9;
			      Ta = T8 + T9;
			      Td = Tb + Tc;
			      TD = Tb - Tc;
			 }
		    }
		    {
			 E TG, Ti, Tj, Tk, Tg, Th;
			 Tg = R1[0];
			 Th = R1[WS(rs, 4)];
			 Te = Ta + Td;
			 Tv = Td - Ta;
			 TO = TD - TC;
			 TE = TC + TD;
			 TG = Tg - Th;
			 Ti = Tg + Th;
			 Tj = R1[WS(rs, 2)];
			 Tk = R1[WS(rs, 6)];
			 {
			      E Tn, To, TH, Tl;
			      Tn = R1[WS(rs, 7)];
			      To = R1[WS(rs, 3)];
			      Tq = R1[WS(rs, 1)];
			      TH = Tj - Tk;
			      Tl = Tj + Tk;
			      TJ = Tn - To;
			      Tp = Tn + To;
			      TI = FNMS(KP414213562, TH, TG);
			      TT = FMA(KP414213562, TG, TH);
			      Ty = Ti + Tl;
			      Tm = Ti - Tl;
			      Tr = R1[WS(rs, 5)];
			 }
		    }
	       }
	       Cr[WS(csr, 4)] = T7 - Te;
	       TK = Tr - Tq;
	       Ts = Tq + Tr;
	       {
		    E Tx, TV, TF, TS, Tz, Tt, TM, TL;
		    Tx = T7 + Te;
		    TV = FNMS(KP707106781, TE, TB);
		    TF = FMA(KP707106781, TE, TB);
		    TL = FNMS(KP414213562, TK, TJ);
		    TS = FMA(KP414213562, TJ, TK);
		    Tz = Tp + Ts;
		    Tt = Tp - Ts;
		    TM = TI + TL;
		    TQ = TL - TI;
		    {
			 E TR, TU, TW, TA, Tw, Tu;
			 TP = FMA(KP707106781, TO, TN);
			 TR = FNMS(KP707106781, TO, TN);
			 TA = Ty + Tz;
			 Ci[WS(csi, 4)] = Tz - Ty;
			 Tw = Tt - Tm;
			 Tu = Tm + Tt;
			 Cr[WS(csr, 1)] = FMA(KP923879532, TM, TF);
			 Cr[WS(csr, 7)] = FNMS(KP923879532, TM, TF);
			 Cr[0] = Tx + TA;
			 Cr[WS(csr, 8)] = Tx - TA;
			 Ci[WS(csi, 6)] = FMS(KP707106781, Tw, Tv);
			 Ci[WS(csi, 2)] = FMA(KP707106781, Tw, Tv);
			 Cr[WS(csr, 2)] = FMA(KP707106781, Tu, Tf);
			 Cr[WS(csr, 6)] = FNMS(KP707106781, Tu, Tf);
			 TU = TS - TT;
			 TW = TT + TS;
			 Ci[WS(csi, 7)] = FMA(KP923879532, TU, TR);
			 Ci[WS(csi, 1)] = FMS(KP923879532, TU, TR);
			 Cr[WS(csr, 3)] = FMA(KP923879532, TW, TV);
			 Cr[WS(csr, 5)] = FNMS(KP923879532, TW, TV);
		    }
	       }
	  }
	  Ci[WS(csi, 5)] = FMS(KP923879532, TQ, TP);
	  Ci[WS(csi, 3)] = FMA(KP923879532, TQ, TP);
     }
}

static const kr2c_desc desc = { 16, "r2cf_16", {38, 0, 20, 0}, &GENUS };

void X(codelet_r2cf_16) (planner *p) {
     X(kr2c_register) (p, r2cf_16, &desc);
}

#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_r2cf -compact -variables 4 -pipeline-latency 4 -n 16 -name r2cf_16 -include r2cf.h */

/*
 * This function contains 58 FP additions, 12 FP multiplications,
 * (or, 54 additions, 8 multiplications, 4 fused multiply/add),
 * 34 stack variables, 3 constants, and 32 memory accesses
 */
#include "r2cf.h"

static void r2cf_16(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
     DK(KP923879532, +0.923879532511286756128183189396788286822416626);
     DK(KP382683432, +0.382683432365089771728459984030398866761344562);
     DK(KP707106781, +0.707106781186547524400844362104849039284835938);
     INT i;
     for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(csr), MAKE_VOLATILE_STRIDE(csi)) {
	  E T3, T6, T7, Tz, Ti, Ta, Td, Te, TA, Th, Tq, TV, TF, TP, Tx;
	  E TU, TE, TM, Tg, Tf, TJ, TQ;
	  {
	       E T1, T2, T4, T5;
	       T1 = R0[0];
	       T2 = R0[WS(rs, 4)];
	       T3 = T1 + T2;
	       T4 = R0[WS(rs, 2)];
	       T5 = R0[WS(rs, 6)];
	       T6 = T4 + T5;
	       T7 = T3 + T6;
	       Tz = T1 - T2;
	       Ti = T4 - T5;
	  }
	  {
	       E T8, T9, Tb, Tc;
	       T8 = R0[WS(rs, 1)];
	       T9 = R0[WS(rs, 5)];
	       Ta = T8 + T9;
	       Tg = T8 - T9;
	       Tb = R0[WS(rs, 7)];
	       Tc = R0[WS(rs, 3)];
	       Td = Tb + Tc;
	       Tf = Tb - Tc;
	  }
	  Te = Ta + Td;
	  TA = KP707106781 * (Tg + Tf);
	  Th = KP707106781 * (Tf - Tg);
	  {
	       E Tm, TN, Tp, TO;
	       {
		    E Tk, Tl, Tn, To;
		    Tk = R1[WS(rs, 7)];
		    Tl = R1[WS(rs, 3)];
		    Tm = Tk - Tl;
		    TN = Tk + Tl;
		    Tn = R1[WS(rs, 1)];
		    To = R1[WS(rs, 5)];
		    Tp = Tn - To;
		    TO = Tn + To;
	       }
	       Tq = FNMS(KP923879532, Tp, KP382683432 * Tm);
	       TV = TN + TO;
	       TF = FMA(KP923879532, Tm, KP382683432 * Tp);
	       TP = TN - TO;
	  }
	  {
	       E Tt, TK, Tw, TL;
	       {
		    E Tr, Ts, Tu, Tv;
		    Tr = R1[0];
		    Ts = R1[WS(rs, 4)];
		    Tt = Tr - Ts;
		    TK = Tr + Ts;
		    Tu = R1[WS(rs, 2)];
		    Tv = R1[WS(rs, 6)];
		    Tw = Tu - Tv;
		    TL = Tu + Tv;
	       }
	       Tx = FMA(KP382683432, Tt, KP923879532 * Tw);
	       TU = TK + TL;
	       TE = FNMS(KP382683432, Tw, KP923879532 * Tt);
	       TM = TK - TL;
	  }
	  Cr[WS(csr, 4)] = T7 - Te;
	  Ci[WS(csi, 4)] = TV - TU;
	  {
	       E Tj, Ty, TD, TG;
	       Tj = Th - Ti;
	       Ty = Tq - Tx;
	       Ci[WS(csi, 1)] = Tj + Ty;
	       Ci[WS(csi, 7)] = Ty - Tj;
	       TD = Tz + TA;
	       TG = TE + TF;
	       Cr[WS(csr, 7)] = TD - TG;
	       Cr[WS(csr, 1)] = TD + TG;
	  }
	  {
	       E TB, TC, TH, TI;
	       TB = Tz - TA;
	       TC = Tx + Tq;
	       Cr[WS(csr, 5)] = TB - TC;
	       Cr[WS(csr, 3)] = TB + TC;
	       TH = Ti + Th;
	       TI = TF - TE;
	       Ci[WS(csi, 3)] = TH + TI;
	       Ci[WS(csi, 5)] = TI - TH;
	  }
	  TJ = T3 - T6;
	  TQ = KP707106781 * (TM + TP);
	  Cr[WS(csr, 6)] = TJ - TQ;
	  Cr[WS(csr, 2)] = TJ + TQ;
	  {
	       E TR, TS, TT, TW;
	       TR = Td - Ta;
	       TS = KP707106781 * (TP - TM);
	       Ci[WS(csi, 2)] = TR + TS;
	       Ci[WS(csi, 6)] = TS - TR;
	       TT = T7 + Te;
	       TW = TU + TV;
	       Cr[WS(csr, 8)] = TT - TW;
	       Cr[0] = TT + TW;
	  }
     }
}

static const kr2c_desc desc = { 16, "r2cf_16", {54, 8, 4, 0}, &GENUS };

void X(codelet_r2cf_16) (planner *p) {
     X(kr2c_register) (p, r2cf_16, &desc);
}

#endif				/* HAVE_FMA */
