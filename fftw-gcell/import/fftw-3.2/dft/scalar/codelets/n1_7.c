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
/* Generated on Mon Nov 10 20:26:04 EST 2008 */

#include "codelet-dft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_notw -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -n 7 -name n1_7 -include n.h */

/*
 * This function contains 60 FP additions, 42 FP multiplications,
 * (or, 18 additions, 0 multiplications, 42 fused multiply/add),
 * 51 stack variables, 6 constants, and 28 memory accesses
 */
#include "n.h"

static void n1_7(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs, INT ovs)
{
     DK(KP974927912, +0.974927912181823607018131682993931217232785801);
     DK(KP900968867, +0.900968867902419126236102319507445051165919162);
     DK(KP801937735, +0.801937735804838252472204639014890102331838324);
     DK(KP692021471, +0.692021471630095869627814897002069140197260599);
     DK(KP356895867, +0.356895867892209443894399510021300583399127187);
     DK(KP554958132, +0.554958132087371191422194871006410481067288862);
     INT i;
     for (i = v; i > 0; i = i - 1, ri = ri + ivs, ii = ii + ivs, ro = ro + ovs, io = io + ovs, MAKE_VOLATILE_STRIDE(is), MAKE_VOLATILE_STRIDE(os)) {
	  E Tz, TP, Ty, TK, TN, TE, Tw, TF;
	  {
	       E T1, TI, T4, TG, Ta, TT, Tp, TH, T7, Tk, TJ, TO, Tu, Tb, TB;
	       E Tg, Tl, Th, Ti;
	       T1 = ri[0];
	       Tz = ii[0];
	       {
		    E T5, T6, Te, Tf;
		    {
			 E T2, T3, T8, T9;
			 T2 = ri[WS(is, 1)];
			 T3 = ri[WS(is, 6)];
			 T8 = ri[WS(is, 3)];
			 T9 = ri[WS(is, 4)];
			 T5 = ri[WS(is, 2)];
			 TI = T3 - T2;
			 T4 = T2 + T3;
			 TG = T9 - T8;
			 Ta = T8 + T9;
			 T6 = ri[WS(is, 5)];
		    }
		    Te = ii[WS(is, 2)];
		    TT = FMA(KP554958132, TG, TI);
		    Tp = FNMS(KP356895867, T4, Ta);
		    TH = T6 - T5;
		    T7 = T5 + T6;
		    Tf = ii[WS(is, 5)];
		    Tk = ii[WS(is, 3)];
		    TJ = FNMS(KP554958132, TI, TH);
		    TO = FMA(KP554958132, TH, TG);
		    Tu = FNMS(KP356895867, Ta, T7);
		    Tb = FNMS(KP356895867, T7, T4);
		    TB = Te + Tf;
		    Tg = Te - Tf;
		    Tl = ii[WS(is, 4)];
		    Th = ii[WS(is, 1)];
		    Ti = ii[WS(is, 6)];
	       }
	       {
		    E Tm, TA, Tj, TD, Ts, TL, Tx, TU, To, TR, Td, TM, Tv;
		    {
			 E TC, TQ, Tn, Tc;
			 ro[0] = T1 + T4 + T7 + Ta;
			 TC = Tk + Tl;
			 Tm = Tk - Tl;
			 TA = Th + Ti;
			 Tj = Th - Ti;
			 TD = FNMS(KP356895867, TC, TB);
			 Ts = FMA(KP554958132, Tg, Tm);
			 TL = FNMS(KP356895867, TA, TC);
			 TQ = FNMS(KP356895867, TB, TA);
			 Tx = FNMS(KP554958132, Tj, Tg);
			 Tn = FMA(KP554958132, Tm, Tj);
			 io[0] = Tz + TA + TB + TC;
			 Tc = FNMS(KP692021471, Tb, Ta);
			 TU = FMA(KP801937735, TT, TH);
			 To = FMA(KP801937735, Tn, Tg);
			 TR = FNMS(KP692021471, TQ, TC);
			 Td = FNMS(KP900968867, Tc, T1);
		    }
		    {
			 E Tt, Tr, TS, Tq;
			 Tt = FNMS(KP801937735, Ts, Tj);
			 Tq = FNMS(KP692021471, Tp, T7);
			 TS = FNMS(KP900968867, TR, Tz);
			 ro[WS(os, 1)] = FMA(KP974927912, To, Td);
			 ro[WS(os, 6)] = FNMS(KP974927912, To, Td);
			 Tr = FNMS(KP900968867, Tq, T1);
			 io[WS(os, 6)] = FNMS(KP974927912, TU, TS);
			 io[WS(os, 1)] = FMA(KP974927912, TU, TS);
			 TP = FNMS(KP801937735, TO, TI);
			 ro[WS(os, 2)] = FMA(KP974927912, Tt, Tr);
			 ro[WS(os, 5)] = FNMS(KP974927912, Tt, Tr);
			 TM = FNMS(KP692021471, TL, TB);
		    }
		    Ty = FNMS(KP801937735, Tx, Tm);
		    Tv = FNMS(KP692021471, Tu, T4);
		    TK = FNMS(KP801937735, TJ, TG);
		    TN = FNMS(KP900968867, TM, Tz);
		    TE = FNMS(KP692021471, TD, TA);
		    Tw = FNMS(KP900968867, Tv, T1);
	       }
	  }
	  io[WS(os, 5)] = FNMS(KP974927912, TP, TN);
	  io[WS(os, 2)] = FMA(KP974927912, TP, TN);
	  TF = FNMS(KP900968867, TE, Tz);
	  ro[WS(os, 3)] = FMA(KP974927912, Ty, Tw);
	  ro[WS(os, 4)] = FNMS(KP974927912, Ty, Tw);
	  io[WS(os, 4)] = FNMS(KP974927912, TK, TF);
	  io[WS(os, 3)] = FMA(KP974927912, TK, TF);
     }
}

static const kdft_desc desc = { 7, "n1_7", {18, 0, 42, 0}, &GENUS, 0, 0, 0, 0 };
void X(codelet_n1_7) (planner *p) {
     X(kdft_register) (p, n1_7, &desc);
}

#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_notw -compact -variables 4 -pipeline-latency 4 -n 7 -name n1_7 -include n.h */

/*
 * This function contains 60 FP additions, 36 FP multiplications,
 * (or, 36 additions, 12 multiplications, 24 fused multiply/add),
 * 25 stack variables, 6 constants, and 28 memory accesses
 */
#include "n.h"

static void n1_7(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs, INT ovs)
{
     DK(KP222520933, +0.222520933956314404288902564496794759466355569);
     DK(KP900968867, +0.900968867902419126236102319507445051165919162);
     DK(KP623489801, +0.623489801858733530525004884004239810632274731);
     DK(KP433883739, +0.433883739117558120475768332848358754609990728);
     DK(KP781831482, +0.781831482468029808708444526674057750232334519);
     DK(KP974927912, +0.974927912181823607018131682993931217232785801);
     INT i;
     for (i = v; i > 0; i = i - 1, ri = ri + ivs, ii = ii + ivs, ro = ro + ovs, io = io + ovs, MAKE_VOLATILE_STRIDE(is), MAKE_VOLATILE_STRIDE(os)) {
	  E T1, Tu, T4, Tq, Te, Tx, T7, Ts, Tk, Tv, Ta, Tr, Th, Tw;
	  T1 = ri[0];
	  Tu = ii[0];
	  {
	       E T2, T3, Tc, Td;
	       T2 = ri[WS(is, 1)];
	       T3 = ri[WS(is, 6)];
	       T4 = T2 + T3;
	       Tq = T3 - T2;
	       Tc = ii[WS(is, 1)];
	       Td = ii[WS(is, 6)];
	       Te = Tc - Td;
	       Tx = Tc + Td;
	  }
	  {
	       E T5, T6, Ti, Tj;
	       T5 = ri[WS(is, 2)];
	       T6 = ri[WS(is, 5)];
	       T7 = T5 + T6;
	       Ts = T6 - T5;
	       Ti = ii[WS(is, 2)];
	       Tj = ii[WS(is, 5)];
	       Tk = Ti - Tj;
	       Tv = Ti + Tj;
	  }
	  {
	       E T8, T9, Tf, Tg;
	       T8 = ri[WS(is, 3)];
	       T9 = ri[WS(is, 4)];
	       Ta = T8 + T9;
	       Tr = T9 - T8;
	       Tf = ii[WS(is, 3)];
	       Tg = ii[WS(is, 4)];
	       Th = Tf - Tg;
	       Tw = Tf + Tg;
	  }
	  ro[0] = T1 + T4 + T7 + Ta;
	  io[0] = Tu + Tx + Tv + Tw;
	  {
	       E Tl, Tb, TB, TC;
	       Tl = FNMS(KP781831482, Th, KP974927912 * Te) - (KP433883739 * Tk);
	       Tb = FMA(KP623489801, Ta, T1) + FNMA(KP900968867, T7, KP222520933 * T4);
	       ro[WS(os, 5)] = Tb - Tl;
	       ro[WS(os, 2)] = Tb + Tl;
	       TB = FNMS(KP781831482, Tr, KP974927912 * Tq) - (KP433883739 * Ts);
	       TC = FMA(KP623489801, Tw, Tu) + FNMA(KP900968867, Tv, KP222520933 * Tx);
	       io[WS(os, 2)] = TB + TC;
	       io[WS(os, 5)] = TC - TB;
	  }
	  {
	       E Tn, Tm, Tz, TA;
	       Tn = FMA(KP781831482, Te, KP974927912 * Tk) + (KP433883739 * Th);
	       Tm = FMA(KP623489801, T4, T1) + FNMA(KP900968867, Ta, KP222520933 * T7);
	       ro[WS(os, 6)] = Tm - Tn;
	       ro[WS(os, 1)] = Tm + Tn;
	       Tz = FMA(KP781831482, Tq, KP974927912 * Ts) + (KP433883739 * Tr);
	       TA = FMA(KP623489801, Tx, Tu) + FNMA(KP900968867, Tw, KP222520933 * Tv);
	       io[WS(os, 1)] = Tz + TA;
	       io[WS(os, 6)] = TA - Tz;
	  }
	  {
	       E Tp, To, Tt, Ty;
	       Tp = FMA(KP433883739, Te, KP974927912 * Th) - (KP781831482 * Tk);
	       To = FMA(KP623489801, T7, T1) + FNMA(KP222520933, Ta, KP900968867 * T4);
	       ro[WS(os, 4)] = To - Tp;
	       ro[WS(os, 3)] = To + Tp;
	       Tt = FMA(KP433883739, Tq, KP974927912 * Tr) - (KP781831482 * Ts);
	       Ty = FMA(KP623489801, Tv, Tu) + FNMA(KP222520933, Tw, KP900968867 * Tx);
	       io[WS(os, 3)] = Tt + Ty;
	       io[WS(os, 4)] = Ty - Tt;
	  }
     }
}

static const kdft_desc desc = { 7, "n1_7", {36, 12, 24, 0}, &GENUS, 0, 0, 0, 0 };
void X(codelet_n1_7) (planner *p) {
     X(kdft_register) (p, n1_7, &desc);
}

#endif				/* HAVE_FMA */
