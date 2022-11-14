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
/* Generated on Mon Nov 10 20:28:11 EST 2008 */

#include "codelet-dft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_twiddle -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -twiddle-log3 -precompute-twiddles -n 8 -name t2_8 -include t.h */

/*
 * This function contains 74 FP additions, 50 FP multiplications,
 * (or, 44 additions, 20 multiplications, 30 fused multiply/add),
 * 64 stack variables, 1 constants, and 32 memory accesses
 */
#include "t.h"

static void t2_8(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DK(KP707106781, +0.707106781186547524400844362104849039284835938);
     INT m;
     for (m = mb, W = W + (mb * 6); m < me; m = m + 1, ri = ri + ms, ii = ii + ms, W = W + 6, MAKE_VOLATILE_STRIDE(rs)) {
	  E TS, T1m, TJ, T1l, T1k, Tw, T1w, T1u;
	  {
	       E T2, T3, Tl, Tn, T5, T4, Tm, Tr, T6;
	       T2 = W[0];
	       T3 = W[2];
	       Tl = W[4];
	       Tn = W[5];
	       T5 = W[1];
	       T4 = T2 * T3;
	       Tm = T2 * Tl;
	       Tr = T2 * Tn;
	       T6 = W[3];
	       {
		    E T1, T1s, TG, Td, T1r, Tu, TY, Tk, TW, T18, T1d, TD, TH, TA, T13;
		    E TE, T14;
		    {
			 E To, Ts, Tf, T7, T8, Ti, Tb, T9, Tc, TC, Ta, TF, TB, Tg, Th;
			 E Tj;
			 T1 = ri[0];
			 To = FMA(T5, Tn, Tm);
			 Ts = FNMS(T5, Tl, Tr);
			 Tf = FMA(T5, T6, T4);
			 T7 = FNMS(T5, T6, T4);
			 Ta = T2 * T6;
			 T1s = ii[0];
			 T8 = ri[WS(rs, 4)];
			 TF = Tf * Tn;
			 TB = Tf * Tl;
			 Ti = FNMS(T5, T3, Ta);
			 Tb = FMA(T5, T3, Ta);
			 T9 = T7 * T8;
			 Tc = ii[WS(rs, 4)];
			 TG = FNMS(Ti, Tl, TF);
			 TC = FMA(Ti, Tn, TB);
			 {
			      E Tp, T1q, Tt, Tq, TX;
			      Tp = ri[WS(rs, 6)];
			      Td = FMA(Tb, Tc, T9);
			      T1q = T7 * Tc;
			      Tt = ii[WS(rs, 6)];
			      Tq = To * Tp;
			      Tg = ri[WS(rs, 2)];
			      T1r = FNMS(Tb, T8, T1q);
			      TX = To * Tt;
			      Tu = FMA(Ts, Tt, Tq);
			      Th = Tf * Tg;
			      Tj = ii[WS(rs, 2)];
			      TY = FNMS(Ts, Tp, TX);
			 }
			 {
			      E TO, TQ, TN, TP, T1a, T1b;
			      {
				   E TK, TM, TL, T19, TV;
				   TK = ri[WS(rs, 7)];
				   TM = ii[WS(rs, 7)];
				   Tk = FMA(Ti, Tj, Th);
				   TV = Tf * Tj;
				   TL = Tl * TK;
				   T19 = Tl * TM;
				   TO = ri[WS(rs, 3)];
				   TW = FNMS(Ti, Tg, TV);
				   TQ = ii[WS(rs, 3)];
				   TN = FMA(Tn, TM, TL);
				   TP = T3 * TO;
				   T1a = FNMS(Tn, TK, T19);
				   T1b = T3 * TQ;
			      }
			      {
				   E Tx, Tz, Ty, T12, T1c, TR;
				   Tx = ri[WS(rs, 1)];
				   TR = FMA(T6, TQ, TP);
				   Tz = ii[WS(rs, 1)];
				   T1c = FNMS(T6, TO, T1b);
				   Ty = T2 * Tx;
				   T18 = TN - TR;
				   TS = TN + TR;
				   T12 = T2 * Tz;
				   T1d = T1a - T1c;
				   T1m = T1a + T1c;
				   TD = ri[WS(rs, 5)];
				   TH = ii[WS(rs, 5)];
				   TA = FMA(T5, Tz, Ty);
				   T13 = FNMS(T5, Tx, T12);
				   TE = TC * TD;
				   T14 = TC * TH;
			      }
			 }
		    }
		    {
			 E Te, T1p, T1t, Tv;
			 {
			      E T1g, T10, T1z, T1B, T1A, T1j, T1C, T1f;
			      {
				   E T1x, T11, T16, T1y;
				   {
					E TU, TZ, TI, T15;
					Te = T1 + Td;
					TU = T1 - Td;
					TZ = TW - TY;
					T1p = TW + TY;
					TI = FMA(TG, TH, TE);
					T15 = FNMS(TG, TD, T14);
					T1t = T1r + T1s;
					T1x = T1s - T1r;
					T1g = TU - TZ;
					T10 = TU + TZ;
					T11 = TA - TI;
					TJ = TA + TI;
					T1l = T13 + T15;
					T16 = T13 - T15;
					T1y = Tk - Tu;
					Tv = Tk + Tu;
				   }
				   {
					E T1i, T1e, T17, T1h;
					T1i = T18 + T1d;
					T1e = T18 - T1d;
					T17 = T11 + T16;
					T1h = T16 - T11;
					T1z = T1x - T1y;
					T1B = T1y + T1x;
					T1A = T1h + T1i;
					T1j = T1h - T1i;
					T1C = T1e - T17;
					T1f = T17 + T1e;
				   }
			      }
			      ri[WS(rs, 7)] = FNMS(KP707106781, T1j, T1g);
			      ii[WS(rs, 7)] = FNMS(KP707106781, T1C, T1B);
			      ri[WS(rs, 1)] = FMA(KP707106781, T1f, T10);
			      ri[WS(rs, 5)] = FNMS(KP707106781, T1f, T10);
			      ii[WS(rs, 1)] = FMA(KP707106781, T1A, T1z);
			      ii[WS(rs, 5)] = FNMS(KP707106781, T1A, T1z);
			      ri[WS(rs, 3)] = FMA(KP707106781, T1j, T1g);
			      ii[WS(rs, 3)] = FMA(KP707106781, T1C, T1B);
			 }
			 T1k = Te - Tv;
			 Tw = Te + Tv;
			 T1w = T1t - T1p;
			 T1u = T1p + T1t;
		    }
	       }
	  }
	  {
	       E TT, T1v, T1n, T1o;
	       TT = TJ + TS;
	       T1v = TS - TJ;
	       T1n = T1l - T1m;
	       T1o = T1l + T1m;
	       ii[WS(rs, 2)] = T1v + T1w;
	       ii[WS(rs, 6)] = T1w - T1v;
	       ri[0] = Tw + TT;
	       ri[WS(rs, 4)] = Tw - TT;
	       ii[0] = T1o + T1u;
	       ii[WS(rs, 4)] = T1u - T1o;
	       ri[WS(rs, 2)] = T1k + T1n;
	       ri[WS(rs, 6)] = T1k - T1n;
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_CEXP, 0, 1},
     {TW_CEXP, 0, 3},
     {TW_CEXP, 0, 7},
     {TW_NEXT, 1, 0}
};

static const ct_desc desc = { 8, "t2_8", twinstr, &GENUS, {44, 20, 30, 0}, 0, 0, 0 };

void X(codelet_t2_8) (planner *p) {
     X(kdft_dit_register) (p, t2_8, &desc);
}
#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_twiddle -compact -variables 4 -pipeline-latency 4 -twiddle-log3 -precompute-twiddles -n 8 -name t2_8 -include t.h */

/*
 * This function contains 74 FP additions, 44 FP multiplications,
 * (or, 56 additions, 26 multiplications, 18 fused multiply/add),
 * 42 stack variables, 1 constants, and 32 memory accesses
 */
#include "t.h"

static void t2_8(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DK(KP707106781, +0.707106781186547524400844362104849039284835938);
     INT m;
     for (m = mb, W = W + (mb * 6); m < me; m = m + 1, ri = ri + ms, ii = ii + ms, W = W + 6, MAKE_VOLATILE_STRIDE(rs)) {
	  E T2, T5, T3, T6, T8, Tc, Tg, Ti, Tl, Tm, Tn, Tz, Tp, Tx;
	  {
	       E T4, Tb, T7, Ta;
	       T2 = W[0];
	       T5 = W[1];
	       T3 = W[2];
	       T6 = W[3];
	       T4 = T2 * T3;
	       Tb = T5 * T3;
	       T7 = T5 * T6;
	       Ta = T2 * T6;
	       T8 = T4 - T7;
	       Tc = Ta + Tb;
	       Tg = T4 + T7;
	       Ti = Ta - Tb;
	       Tl = W[4];
	       Tm = W[5];
	       Tn = FMA(T2, Tl, T5 * Tm);
	       Tz = FNMS(Ti, Tl, Tg * Tm);
	       Tp = FNMS(T5, Tl, T2 * Tm);
	       Tx = FMA(Tg, Tl, Ti * Tm);
	  }
	  {
	       E Tf, T1i, TL, T1d, TJ, T17, TV, TY, Ts, T1j, TO, T1a, TC, T16, TQ;
	       E TT;
	       {
		    E T1, T1c, Te, T1b, T9, Td;
		    T1 = ri[0];
		    T1c = ii[0];
		    T9 = ri[WS(rs, 4)];
		    Td = ii[WS(rs, 4)];
		    Te = FMA(T8, T9, Tc * Td);
		    T1b = FNMS(Tc, T9, T8 * Td);
		    Tf = T1 + Te;
		    T1i = T1c - T1b;
		    TL = T1 - Te;
		    T1d = T1b + T1c;
	       }
	       {
		    E TF, TW, TI, TX;
		    {
			 E TD, TE, TG, TH;
			 TD = ri[WS(rs, 7)];
			 TE = ii[WS(rs, 7)];
			 TF = FMA(Tl, TD, Tm * TE);
			 TW = FNMS(Tm, TD, Tl * TE);
			 TG = ri[WS(rs, 3)];
			 TH = ii[WS(rs, 3)];
			 TI = FMA(T3, TG, T6 * TH);
			 TX = FNMS(T6, TG, T3 * TH);
		    }
		    TJ = TF + TI;
		    T17 = TW + TX;
		    TV = TF - TI;
		    TY = TW - TX;
	       }
	       {
		    E Tk, TM, Tr, TN;
		    {
			 E Th, Tj, To, Tq;
			 Th = ri[WS(rs, 2)];
			 Tj = ii[WS(rs, 2)];
			 Tk = FMA(Tg, Th, Ti * Tj);
			 TM = FNMS(Ti, Th, Tg * Tj);
			 To = ri[WS(rs, 6)];
			 Tq = ii[WS(rs, 6)];
			 Tr = FMA(Tn, To, Tp * Tq);
			 TN = FNMS(Tp, To, Tn * Tq);
		    }
		    Ts = Tk + Tr;
		    T1j = Tk - Tr;
		    TO = TM - TN;
		    T1a = TM + TN;
	       }
	       {
		    E Tw, TR, TB, TS;
		    {
			 E Tu, Tv, Ty, TA;
			 Tu = ri[WS(rs, 1)];
			 Tv = ii[WS(rs, 1)];
			 Tw = FMA(T2, Tu, T5 * Tv);
			 TR = FNMS(T5, Tu, T2 * Tv);
			 Ty = ri[WS(rs, 5)];
			 TA = ii[WS(rs, 5)];
			 TB = FMA(Tx, Ty, Tz * TA);
			 TS = FNMS(Tz, Ty, Tx * TA);
		    }
		    TC = Tw + TB;
		    T16 = TR + TS;
		    TQ = Tw - TB;
		    TT = TR - TS;
	       }
	       {
		    E Tt, TK, T1f, T1g;
		    Tt = Tf + Ts;
		    TK = TC + TJ;
		    ri[WS(rs, 4)] = Tt - TK;
		    ri[0] = Tt + TK;
		    {
			 E T19, T1e, T15, T18;
			 T19 = T16 + T17;
			 T1e = T1a + T1d;
			 ii[0] = T19 + T1e;
			 ii[WS(rs, 4)] = T1e - T19;
			 T15 = Tf - Ts;
			 T18 = T16 - T17;
			 ri[WS(rs, 6)] = T15 - T18;
			 ri[WS(rs, 2)] = T15 + T18;
		    }
		    T1f = TJ - TC;
		    T1g = T1d - T1a;
		    ii[WS(rs, 2)] = T1f + T1g;
		    ii[WS(rs, 6)] = T1g - T1f;
		    {
			 E T11, T1k, T14, T1h, T12, T13;
			 T11 = TL - TO;
			 T1k = T1i - T1j;
			 T12 = TT - TQ;
			 T13 = TV + TY;
			 T14 = KP707106781 * (T12 - T13);
			 T1h = KP707106781 * (T12 + T13);
			 ri[WS(rs, 7)] = T11 - T14;
			 ii[WS(rs, 5)] = T1k - T1h;
			 ri[WS(rs, 3)] = T11 + T14;
			 ii[WS(rs, 1)] = T1h + T1k;
		    }
		    {
			 E TP, T1m, T10, T1l, TU, TZ;
			 TP = TL + TO;
			 T1m = T1j + T1i;
			 TU = TQ + TT;
			 TZ = TV - TY;
			 T10 = KP707106781 * (TU + TZ);
			 T1l = KP707106781 * (TZ - TU);
			 ri[WS(rs, 5)] = TP - T10;
			 ii[WS(rs, 7)] = T1m - T1l;
			 ri[WS(rs, 1)] = TP + T10;
			 ii[WS(rs, 3)] = T1l + T1m;
		    }
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_CEXP, 0, 1},
     {TW_CEXP, 0, 3},
     {TW_CEXP, 0, 7},
     {TW_NEXT, 1, 0}
};

static const ct_desc desc = { 8, "t2_8", twinstr, &GENUS, {56, 26, 18, 0}, 0, 0, 0 };

void X(codelet_t2_8) (planner *p) {
     X(kdft_dit_register) (p, t2_8, &desc);
}
#endif				/* HAVE_FMA */
