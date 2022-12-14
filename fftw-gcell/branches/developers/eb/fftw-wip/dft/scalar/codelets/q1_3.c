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
/* Generated on Mon Nov 10 20:31:17 EST 2008 */

#include "codelet-dft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_twidsq -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -reload-twiddle -dif -n 3 -name q1_3 -include q.h */

/*
 * This function contains 48 FP additions, 42 FP multiplications,
 * (or, 18 additions, 12 multiplications, 30 fused multiply/add),
 * 56 stack variables, 2 constants, and 36 memory accesses
 */
#include "q.h"

static void q1_3(R *rio, R *iio, const R *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     DK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DK(KP500000000, +0.500000000000000000000000000000000000000000000);
     INT m;
     for (m = mb, W = W + (mb * 4); m < me; m = m + 1, rio = rio + ms, iio = iio + ms, W = W + 4, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(vs)) {
	  E Tk, Tn, Tm, To, Tl;
	  {
	       E T1, Td, T4, Tg, Tp, T9, Te, T6, Tf, TB, TE, Ts, TZ, Tu, Tx;
	       E TC, TN, TO, TD, TV, T10, TP, Tq, Tr;
	       {
		    E T2, T3, T7, T8;
		    T1 = rio[0];
		    T2 = rio[WS(rs, 1)];
		    T3 = rio[WS(rs, 2)];
		    Td = iio[0];
		    T7 = iio[WS(rs, 1)];
		    T8 = iio[WS(rs, 2)];
		    T4 = T2 + T3;
		    Tg = T3 - T2;
		    Tp = rio[WS(vs, 1)];
		    T9 = T7 - T8;
		    Te = T7 + T8;
		    T6 = FNMS(KP500000000, T4, T1);
		    Tq = rio[WS(vs, 1) + WS(rs, 1)];
		    Tr = rio[WS(vs, 1) + WS(rs, 2)];
		    Tf = FNMS(KP500000000, Te, Td);
	       }
	       {
		    E Tv, Tw, TT, TU;
		    TB = iio[WS(vs, 1)];
		    Tv = iio[WS(vs, 1) + WS(rs, 1)];
		    TE = Tr - Tq;
		    Ts = Tq + Tr;
		    Tw = iio[WS(vs, 1) + WS(rs, 2)];
		    TZ = iio[WS(vs, 2)];
		    TT = iio[WS(vs, 2) + WS(rs, 1)];
		    Tu = FNMS(KP500000000, Ts, Tp);
		    Tx = Tv - Tw;
		    TC = Tv + Tw;
		    TU = iio[WS(vs, 2) + WS(rs, 2)];
		    TN = rio[WS(vs, 2)];
		    TO = rio[WS(vs, 2) + WS(rs, 1)];
		    TD = FNMS(KP500000000, TC, TB);
		    TV = TT - TU;
		    T10 = TT + TU;
		    TP = rio[WS(vs, 2) + WS(rs, 2)];
	       }
	       {
		    E T11, T12, TS, TQ;
		    rio[0] = T1 + T4;
		    iio[0] = Td + Te;
		    T11 = FNMS(KP500000000, T10, TZ);
		    T12 = TP - TO;
		    TQ = TO + TP;
		    rio[WS(rs, 1)] = Tp + Ts;
		    iio[WS(rs, 1)] = TB + TC;
		    iio[WS(rs, 2)] = TZ + T10;
		    TS = FNMS(KP500000000, TQ, TN);
		    rio[WS(rs, 2)] = TN + TQ;
		    {
			 E TW, T13, Ty, TI, TL, TF, TH, TK;
			 {
			      E Ta, Th, T5, Tc;
			      Tk = FNMS(KP866025403, T9, T6);
			      Ta = FMA(KP866025403, T9, T6);
			      Th = FMA(KP866025403, Tg, Tf);
			      Tn = FNMS(KP866025403, Tg, Tf);
			      T5 = W[0];
			      Tc = W[1];
			      {
				   E T16, T19, T18, T1a, T17, Ti, Tb, T15;
				   TW = FMA(KP866025403, TV, TS);
				   T16 = FNMS(KP866025403, TV, TS);
				   T19 = FNMS(KP866025403, T12, T11);
				   T13 = FMA(KP866025403, T12, T11);
				   Ti = T5 * Th;
				   Tb = T5 * Ta;
				   T15 = W[2];
				   T18 = W[3];
				   iio[WS(vs, 1)] = FNMS(Tc, Ta, Ti);
				   rio[WS(vs, 1)] = FMA(Tc, Th, Tb);
				   T1a = T15 * T19;
				   T17 = T15 * T16;
				   Ty = FMA(KP866025403, Tx, Tu);
				   TI = FNMS(KP866025403, Tx, Tu);
				   TL = FNMS(KP866025403, TE, TD);
				   TF = FMA(KP866025403, TE, TD);
				   iio[WS(vs, 2) + WS(rs, 2)] = FNMS(T18, T16, T1a);
				   rio[WS(vs, 2) + WS(rs, 2)] = FMA(T18, T19, T17);
				   TH = W[2];
				   TK = W[3];
			      }
			 }
			 {
			      E TA, TG, Tz, TM, TJ, Tt;
			      TM = TH * TL;
			      TJ = TH * TI;
			      Tt = W[0];
			      TA = W[1];
			      iio[WS(vs, 2) + WS(rs, 1)] = FNMS(TK, TI, TM);
			      rio[WS(vs, 2) + WS(rs, 1)] = FMA(TK, TL, TJ);
			      TG = Tt * TF;
			      Tz = Tt * Ty;
			      {
				   E TR, TY, T14, TX, Tj;
				   iio[WS(vs, 1) + WS(rs, 1)] = FNMS(TA, Ty, TG);
				   rio[WS(vs, 1) + WS(rs, 1)] = FMA(TA, TF, Tz);
				   TR = W[0];
				   TY = W[1];
				   T14 = TR * T13;
				   TX = TR * TW;
				   Tj = W[2];
				   Tm = W[3];
				   iio[WS(vs, 1) + WS(rs, 2)] = FNMS(TY, TW, T14);
				   rio[WS(vs, 1) + WS(rs, 2)] = FMA(TY, T13, TX);
				   To = Tj * Tn;
				   Tl = Tj * Tk;
			      }
			 }
		    }
	       }
	  }
	  iio[WS(vs, 2)] = FNMS(Tm, Tk, To);
	  rio[WS(vs, 2)] = FMA(Tm, Tn, Tl);
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 0, 3},
     {TW_NEXT, 1, 0}
};

static const ct_desc desc = { 3, "q1_3", twinstr, &GENUS, {18, 12, 30, 0}, 0, 0, 0 };

void X(codelet_q1_3) (planner *p) {
     X(kdft_difsq_register) (p, q1_3, &desc);
}
#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_twidsq -compact -variables 4 -pipeline-latency 4 -reload-twiddle -dif -n 3 -name q1_3 -include q.h */

/*
 * This function contains 48 FP additions, 36 FP multiplications,
 * (or, 30 additions, 18 multiplications, 18 fused multiply/add),
 * 35 stack variables, 2 constants, and 36 memory accesses
 */
#include "q.h"

static void q1_3(R *rio, R *iio, const R *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     DK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DK(KP500000000, +0.500000000000000000000000000000000000000000000);
     INT m;
     for (m = mb, W = W + (mb * 4); m < me; m = m + 1, rio = rio + ms, iio = iio + ms, W = W + 4, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(vs)) {
	  E T1, T4, T6, Tc, Td, Te, T9, Tf, Tl, To, Tq, Tw, Tx, Ty, Tt;
	  E Tz, TR, TS, TN, TT, TF, TI, TK, TQ;
	  {
	       E T2, T3, Tr, Ts;
	       T1 = rio[0];
	       T2 = rio[WS(rs, 1)];
	       T3 = rio[WS(rs, 2)];
	       T4 = T2 + T3;
	       T6 = FNMS(KP500000000, T4, T1);
	       Tc = KP866025403 * (T3 - T2);
	       {
		    E T7, T8, Tm, Tn;
		    Td = iio[0];
		    T7 = iio[WS(rs, 1)];
		    T8 = iio[WS(rs, 2)];
		    Te = T7 + T8;
		    T9 = KP866025403 * (T7 - T8);
		    Tf = FNMS(KP500000000, Te, Td);
		    Tl = rio[WS(vs, 1)];
		    Tm = rio[WS(vs, 1) + WS(rs, 1)];
		    Tn = rio[WS(vs, 1) + WS(rs, 2)];
		    To = Tm + Tn;
		    Tq = FNMS(KP500000000, To, Tl);
		    Tw = KP866025403 * (Tn - Tm);
	       }
	       Tx = iio[WS(vs, 1)];
	       Tr = iio[WS(vs, 1) + WS(rs, 1)];
	       Ts = iio[WS(vs, 1) + WS(rs, 2)];
	       Ty = Tr + Ts;
	       Tt = KP866025403 * (Tr - Ts);
	       Tz = FNMS(KP500000000, Ty, Tx);
	       {
		    E TL, TM, TG, TH;
		    TR = iio[WS(vs, 2)];
		    TL = iio[WS(vs, 2) + WS(rs, 1)];
		    TM = iio[WS(vs, 2) + WS(rs, 2)];
		    TS = TL + TM;
		    TN = KP866025403 * (TL - TM);
		    TT = FNMS(KP500000000, TS, TR);
		    TF = rio[WS(vs, 2)];
		    TG = rio[WS(vs, 2) + WS(rs, 1)];
		    TH = rio[WS(vs, 2) + WS(rs, 2)];
		    TI = TG + TH;
		    TK = FNMS(KP500000000, TI, TF);
		    TQ = KP866025403 * (TH - TG);
	       }
	  }
	  rio[0] = T1 + T4;
	  iio[0] = Td + Te;
	  rio[WS(rs, 1)] = Tl + To;
	  iio[WS(rs, 1)] = Tx + Ty;
	  iio[WS(rs, 2)] = TR + TS;
	  rio[WS(rs, 2)] = TF + TI;
	  {
	       E Ta, Tg, T5, Tb;
	       Ta = T6 + T9;
	       Tg = Tc + Tf;
	       T5 = W[0];
	       Tb = W[1];
	       rio[WS(vs, 1)] = FMA(T5, Ta, Tb * Tg);
	       iio[WS(vs, 1)] = FNMS(Tb, Ta, T5 * Tg);
	  }
	  {
	       E TW, TY, TV, TX;
	       TW = TK - TN;
	       TY = TT - TQ;
	       TV = W[2];
	       TX = W[3];
	       rio[WS(vs, 2) + WS(rs, 2)] = FMA(TV, TW, TX * TY);
	       iio[WS(vs, 2) + WS(rs, 2)] = FNMS(TX, TW, TV * TY);
	  }
	  {
	       E TC, TE, TB, TD;
	       TC = Tq - Tt;
	       TE = Tz - Tw;
	       TB = W[2];
	       TD = W[3];
	       rio[WS(vs, 2) + WS(rs, 1)] = FMA(TB, TC, TD * TE);
	       iio[WS(vs, 2) + WS(rs, 1)] = FNMS(TD, TC, TB * TE);
	  }
	  {
	       E Tu, TA, Tp, Tv;
	       Tu = Tq + Tt;
	       TA = Tw + Tz;
	       Tp = W[0];
	       Tv = W[1];
	       rio[WS(vs, 1) + WS(rs, 1)] = FMA(Tp, Tu, Tv * TA);
	       iio[WS(vs, 1) + WS(rs, 1)] = FNMS(Tv, Tu, Tp * TA);
	  }
	  {
	       E TO, TU, TJ, TP;
	       TO = TK + TN;
	       TU = TQ + TT;
	       TJ = W[0];
	       TP = W[1];
	       rio[WS(vs, 1) + WS(rs, 2)] = FMA(TJ, TO, TP * TU);
	       iio[WS(vs, 1) + WS(rs, 2)] = FNMS(TP, TO, TJ * TU);
	  }
	  {
	       E Ti, Tk, Th, Tj;
	       Ti = T6 - T9;
	       Tk = Tf - Tc;
	       Th = W[2];
	       Tj = W[3];
	       rio[WS(vs, 2)] = FMA(Th, Ti, Tj * Tk);
	       iio[WS(vs, 2)] = FNMS(Tj, Ti, Th * Tk);
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 0, 3},
     {TW_NEXT, 1, 0}
};

static const ct_desc desc = { 3, "q1_3", twinstr, &GENUS, {30, 18, 18, 0}, 0, 0, 0 };

void X(codelet_q1_3) (planner *p) {
     X(kdft_difsq_register) (p, q1_3, &desc);
}
#endif				/* HAVE_FMA */
