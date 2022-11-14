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
/* Generated on Mon Nov 10 21:00:08 EST 2008 */

#include "codelet-rdft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_hc2c -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -sign 1 -twiddle-log3 -precompute-twiddles -n 8 -dif -name hc2cb2_8 -include hc2cb.h */

/*
 * This function contains 74 FP additions, 50 FP multiplications,
 * (or, 44 additions, 20 multiplications, 30 fused multiply/add),
 * 64 stack variables, 1 constants, and 32 memory accesses
 */
#include "hc2cb.h"

static void hc2cb2_8(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DK(KP707106781, +0.707106781186547524400844362104849039284835938);
     INT m;
     for (m = mb, W = W + ((mb - 1) * 6); m < me; m = m + 1, Rp = Rp + ms, Ip = Ip + ms, Rm = Rm - ms, Im = Im - ms, W = W + 6, MAKE_VOLATILE_STRIDE(rs)) {
	  E Tf, Ti, TK, Tq, TH, TT, TX, TW, TY, TU, TI;
	  {
	       E Tg, Tl, Tp, Th, T1n, T1t, Tj;
	       Tf = W[0];
	       Tg = W[2];
	       Tl = W[4];
	       Tp = W[5];
	       Ti = W[1];
	       Th = Tf * Tg;
	       T1n = Tf * Tl;
	       T1t = Tf * Tp;
	       Tj = W[3];
	       {
		    E T1o, T1u, Tk, T1b, To, T1e, T13, TP, T1p, T7, T1h, T1v, TZ, Tv, T1i;
		    E TB, TA, TQ, Te, T1w, TE, T1j;
		    {
			 E Tr, T3, Ts, T1f, TO, TL, T6, Tt;
			 {
			      E TM, TN, T4, T5;
			      {
				   E T1, Tn, T2, TJ, Tm;
				   T1 = Rp[0];
				   T1o = FMA(Ti, Tp, T1n);
				   T1u = FNMS(Ti, Tl, T1t);
				   Tk = FMA(Ti, Tj, Th);
				   T1b = FNMS(Ti, Tj, Th);
				   Tn = Tf * Tj;
				   T2 = Rm[WS(rs, 3)];
				   TM = Ip[0];
				   TJ = Tk * Tp;
				   Tm = Tk * Tl;
				   To = FNMS(Ti, Tg, Tn);
				   T1e = FMA(Ti, Tg, Tn);
				   Tr = T1 - T2;
				   T3 = T1 + T2;
				   TK = FNMS(To, Tl, TJ);
				   Tq = FMA(To, Tp, Tm);
				   TN = Im[WS(rs, 3)];
			      }
			      T4 = Rp[WS(rs, 2)];
			      T5 = Rm[WS(rs, 1)];
			      Ts = Ip[WS(rs, 2)];
			      T1f = TM - TN;
			      TO = TM + TN;
			      TL = T4 - T5;
			      T6 = T4 + T5;
			      Tt = Im[WS(rs, 1)];
			 }
			 {
			      E Tw, Ta, TC, Tz, Td, TD;
			      {
				   E Tx, Ty, Tb, Tc;
				   {
					E T8, T1g, Tu, T9;
					T8 = Rp[WS(rs, 1)];
					T13 = TO - TL;
					TP = TL + TO;
					T1p = T3 - T6;
					T7 = T3 + T6;
					T1g = Ts - Tt;
					Tu = Ts + Tt;
					T9 = Rm[WS(rs, 2)];
					Tx = Ip[WS(rs, 1)];
					T1h = T1f + T1g;
					T1v = T1f - T1g;
					TZ = Tr + Tu;
					Tv = Tr - Tu;
					Tw = T8 - T9;
					Ta = T8 + T9;
					Ty = Im[WS(rs, 2)];
				   }
				   Tb = Rm[0];
				   Tc = Rp[WS(rs, 3)];
				   TC = Ip[WS(rs, 3)];
				   T1i = Tx - Ty;
				   Tz = Tx + Ty;
				   TB = Tb - Tc;
				   Td = Tb + Tc;
				   TD = Im[0];
			      }
			      TA = Tw - Tz;
			      TQ = Tw + Tz;
			      Te = Ta + Td;
			      T1w = Ta - Td;
			      TE = TC + TD;
			      T1j = TC - TD;
			 }
		    }
		    {
			 E T1x, T1k, T1r, TG, TS, T19, T15, T17, T11, T16, T12;
			 {
			      E T1B, T1z, T10, T1A, T1C;
			      T1x = T1v - T1w;
			      T1B = T1w + T1v;
			      Rp[0] = T7 + Te;
			      {
				   E T1q, TR, TF, T14;
				   T1k = T1i + T1j;
				   T1q = T1j - T1i;
				   TR = TB + TE;
				   TF = TB - TE;
				   T1r = T1p - T1q;
				   T1z = T1p + T1q;
				   Rm[0] = T1h + T1k;
				   TG = TA + TF;
				   T14 = TA - TF;
				   TS = TQ - TR;
				   T10 = TQ + TR;
				   T1A = Tk * T1z;
				   T19 = FNMS(KP707106781, T14, T13);
				   T15 = FMA(KP707106781, T14, T13);
				   T1C = Tk * T1B;
			      }
			      T17 = FMA(KP707106781, T10, TZ);
			      T11 = FNMS(KP707106781, T10, TZ);
			      Rp[WS(rs, 1)] = FNMS(To, T1B, T1A);
			      T16 = Tg * T15;
			      Rm[WS(rs, 1)] = FMA(To, T1z, T1C);
			 }
			 T12 = Tg * T11;
			 {
			      E T1l, T1a, T1c, T18;
			      Im[WS(rs, 1)] = FMA(Tj, T11, T16);
			      Ip[WS(rs, 1)] = FNMS(Tj, T15, T12);
			      T18 = Tl * T17;
			      T1l = T1h - T1k;
			      T1a = Tl * T19;
			      T1c = T7 - Te;
			      Ip[WS(rs, 3)] = FNMS(Tp, T19, T18);
			      {
				   E T1s, T1m, T1d, T1y, TV;
				   Im[WS(rs, 3)] = FMA(Tp, T17, T1a);
				   T1m = T1e * T1c;
				   T1d = T1b * T1c;
				   T1s = T1o * T1r;
				   Rm[WS(rs, 2)] = FMA(T1b, T1l, T1m);
				   Rp[WS(rs, 2)] = FNMS(T1e, T1l, T1d);
				   Rp[WS(rs, 3)] = FNMS(T1u, T1x, T1s);
				   T1y = T1o * T1x;
				   TV = FMA(KP707106781, TG, Tv);
				   TH = FNMS(KP707106781, TG, Tv);
				   TT = FNMS(KP707106781, TS, TP);
				   TX = FMA(KP707106781, TS, TP);
				   Rm[WS(rs, 3)] = FMA(T1u, T1r, T1y);
				   TW = Tf * TV;
				   TY = Ti * TV;
			      }
			 }
		    }
	       }
	  }
	  Ip[0] = FNMS(Ti, TX, TW);
	  Im[0] = FMA(Tf, TX, TY);
	  TU = TK * TH;
	  TI = Tq * TH;
	  Im[WS(rs, 2)] = FMA(Tq, TT, TU);
	  Ip[WS(rs, 2)] = FNMS(TK, TT, TI);
     }
}

static const tw_instr twinstr[] = {
     {TW_CEXP, 1, 1},
     {TW_CEXP, 1, 3},
     {TW_CEXP, 1, 7},
     {TW_NEXT, 1, 0}
};

static const hc2c_desc desc = { 8, "hc2cb2_8", twinstr, &GENUS, {44, 20, 30, 0} };

void X(codelet_hc2cb2_8) (planner *p) {
     X(khc2c_register) (p, hc2cb2_8, &desc, HC2C_VIA_RDFT);
}
#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_hc2c -compact -variables 4 -pipeline-latency 4 -sign 1 -twiddle-log3 -precompute-twiddles -n 8 -dif -name hc2cb2_8 -include hc2cb.h */

/*
 * This function contains 74 FP additions, 44 FP multiplications,
 * (or, 56 additions, 26 multiplications, 18 fused multiply/add),
 * 46 stack variables, 1 constants, and 32 memory accesses
 */
#include "hc2cb.h"

static void hc2cb2_8(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DK(KP707106781, +0.707106781186547524400844362104849039284835938);
     INT m;
     for (m = mb, W = W + ((mb - 1) * 6); m < me; m = m + 1, Rp = Rp + ms, Ip = Ip + ms, Rm = Rm - ms, Im = Im - ms, W = W + 6, MAKE_VOLATILE_STRIDE(rs)) {
	  E Tf, Ti, Tg, Tj, Tl, Tp, TP, TR, TF, TG, TH, T15, TL, TT;
	  {
	       E Th, To, Tk, Tn;
	       Tf = W[0];
	       Ti = W[1];
	       Tg = W[2];
	       Tj = W[3];
	       Th = Tf * Tg;
	       To = Ti * Tg;
	       Tk = Ti * Tj;
	       Tn = Tf * Tj;
	       Tl = Th - Tk;
	       Tp = Tn + To;
	       TP = Th + Tk;
	       TR = Tn - To;
	       TF = W[4];
	       TG = W[5];
	       TH = FMA(Tf, TF, Ti * TG);
	       T15 = FNMS(TR, TF, TP * TG);
	       TL = FNMS(Ti, TF, Tf * TG);
	       TT = FMA(TP, TF, TR * TG);
	  }
	  {
	       E T7, T1f, T1i, Tw, TI, TW, T18, TM, Te, T19, T1a, TD, TJ, TZ, T12;
	       E TN, Tm, TE;
	       {
		    E T3, TU, Ts, T17, T6, T16, Tv, TV;
		    {
			 E T1, T2, Tq, Tr;
			 T1 = Rp[0];
			 T2 = Rm[WS(rs, 3)];
			 T3 = T1 + T2;
			 TU = T1 - T2;
			 Tq = Ip[0];
			 Tr = Im[WS(rs, 3)];
			 Ts = Tq - Tr;
			 T17 = Tq + Tr;
		    }
		    {
			 E T4, T5, Tt, Tu;
			 T4 = Rp[WS(rs, 2)];
			 T5 = Rm[WS(rs, 1)];
			 T6 = T4 + T5;
			 T16 = T4 - T5;
			 Tt = Ip[WS(rs, 2)];
			 Tu = Im[WS(rs, 1)];
			 Tv = Tt - Tu;
			 TV = Tt + Tu;
		    }
		    T7 = T3 + T6;
		    T1f = TU + TV;
		    T1i = T17 - T16;
		    Tw = Ts + Tv;
		    TI = T3 - T6;
		    TW = TU - TV;
		    T18 = T16 + T17;
		    TM = Ts - Tv;
	       }
	       {
		    E Ta, TX, Tz, TY, Td, T10, TC, T11;
		    {
			 E T8, T9, Tx, Ty;
			 T8 = Rp[WS(rs, 1)];
			 T9 = Rm[WS(rs, 2)];
			 Ta = T8 + T9;
			 TX = T8 - T9;
			 Tx = Ip[WS(rs, 1)];
			 Ty = Im[WS(rs, 2)];
			 Tz = Tx - Ty;
			 TY = Tx + Ty;
		    }
		    {
			 E Tb, Tc, TA, TB;
			 Tb = Rm[0];
			 Tc = Rp[WS(rs, 3)];
			 Td = Tb + Tc;
			 T10 = Tb - Tc;
			 TA = Ip[WS(rs, 3)];
			 TB = Im[0];
			 TC = TA - TB;
			 T11 = TA + TB;
		    }
		    Te = Ta + Td;
		    T19 = TX + TY;
		    T1a = T10 + T11;
		    TD = Tz + TC;
		    TJ = TC - Tz;
		    TZ = TX - TY;
		    T12 = T10 - T11;
		    TN = Ta - Td;
	       }
	       Rp[0] = T7 + Te;
	       Rm[0] = Tw + TD;
	       Tm = T7 - Te;
	       TE = Tw - TD;
	       Rp[WS(rs, 2)] = FNMS(Tp, TE, Tl * Tm);
	       Rm[WS(rs, 2)] = FMA(Tp, Tm, Tl * TE);
	       {
		    E TQ, TS, TK, TO;
		    TQ = TI + TJ;
		    TS = TN + TM;
		    Rp[WS(rs, 1)] = FNMS(TR, TS, TP * TQ);
		    Rm[WS(rs, 1)] = FMA(TP, TS, TR * TQ);
		    TK = TI - TJ;
		    TO = TM - TN;
		    Rp[WS(rs, 3)] = FNMS(TL, TO, TH * TK);
		    Rm[WS(rs, 3)] = FMA(TH, TO, TL * TK);
	       }
	       {
		    E T1h, T1l, T1k, T1m, T1g, T1j;
		    T1g = KP707106781 * (T19 + T1a);
		    T1h = T1f - T1g;
		    T1l = T1f + T1g;
		    T1j = KP707106781 * (TZ - T12);
		    T1k = T1i + T1j;
		    T1m = T1i - T1j;
		    Ip[WS(rs, 1)] = FNMS(Tj, T1k, Tg * T1h);
		    Im[WS(rs, 1)] = FMA(Tg, T1k, Tj * T1h);
		    Ip[WS(rs, 3)] = FNMS(TG, T1m, TF * T1l);
		    Im[WS(rs, 3)] = FMA(TF, T1m, TG * T1l);
	       }
	       {
		    E T14, T1d, T1c, T1e, T13, T1b;
		    T13 = KP707106781 * (TZ + T12);
		    T14 = TW - T13;
		    T1d = TW + T13;
		    T1b = KP707106781 * (T19 - T1a);
		    T1c = T18 - T1b;
		    T1e = T18 + T1b;
		    Ip[WS(rs, 2)] = FNMS(T15, T1c, TT * T14);
		    Im[WS(rs, 2)] = FMA(T15, T14, TT * T1c);
		    Ip[0] = FNMS(Ti, T1e, Tf * T1d);
		    Im[0] = FMA(Ti, T1d, Tf * T1e);
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_CEXP, 1, 1},
     {TW_CEXP, 1, 3},
     {TW_CEXP, 1, 7},
     {TW_NEXT, 1, 0}
};

static const hc2c_desc desc = { 8, "hc2cb2_8", twinstr, &GENUS, {56, 26, 18, 0} };

void X(codelet_hc2cb2_8) (planner *p) {
     X(khc2c_register) (p, hc2cb2_8, &desc, HC2C_VIA_RDFT);
}
#endif				/* HAVE_FMA */
