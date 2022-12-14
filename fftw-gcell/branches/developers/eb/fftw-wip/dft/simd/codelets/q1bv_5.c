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
/* Generated on Mon Nov 10 20:44:15 EST 2008 */

#include "codelet-dft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_twidsq_c -fma -reorder-insns -schedule-for-pipeline -simd -compact -variables 4 -pipeline-latency 8 -n 5 -dif -name q1bv_5 -include q1b.h -sign 1 */

/*
 * This function contains 100 FP additions, 95 FP multiplications,
 * (or, 55 additions, 50 multiplications, 45 fused multiply/add),
 * 69 stack variables, 4 constants, and 50 memory accesses
 */
#include "q1b.h"

static void q1bv_5(R *ri, R *ii, const R *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
     DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
     DVK(KP618033988, +0.618033988749894848204586834365638117720309180);
     DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
     INT m;
     R *x;
     x = ii;
     for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me; m = m + VL, x = x + (VL * ms), W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(vs)) {
	  V Te, T1w, Ty, TS, TW, Tb, T1t, Tv, T1g, T1c, TP, TV, T1f, T19, TY;
	  V TX;
	  {
	       V T1, T1j, Tl, Ti, Ta, T8, T1A, T1q, T1s, T9, TF, T1r, TZ, TR, TL;
	       V TC, Ts, Tu, TQ, TI, T15, T1b, T10, T11, Tt;
	       {
		    V T1n, T1o, T1k, T1l, T7, Td, T4, Tc;
		    {
			 V T5, T6, T2, T3;
			 T1 = LD(&(x[0]), ms, &(x[0]));
			 T5 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
			 T6 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
			 T2 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
			 T3 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
			 T1j = LD(&(x[WS(vs, 4)]), ms, &(x[WS(vs, 4)]));
			 T1n = LD(&(x[WS(vs, 4) + WS(rs, 2)]), ms, &(x[WS(vs, 4)]));
			 T1o = LD(&(x[WS(vs, 4) + WS(rs, 3)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 T1k = LD(&(x[WS(vs, 4) + WS(rs, 1)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 T1l = LD(&(x[WS(vs, 4) + WS(rs, 4)]), ms, &(x[WS(vs, 4)]));
			 T7 = VADD(T5, T6);
			 Td = VSUB(T5, T6);
			 T4 = VADD(T2, T3);
			 Tc = VSUB(T2, T3);
		    }
		    {
			 V Tm, Tn, Tr, Tx, T1v, T1p;
			 Tl = LD(&(x[WS(vs, 1)]), ms, &(x[WS(vs, 1)]));
			 T1v = VSUB(T1n, T1o);
			 T1p = VADD(T1n, T1o);
			 {
			      V T1u, T1m, Tp, Tq;
			      T1u = VSUB(T1k, T1l);
			      T1m = VADD(T1k, T1l);
			      Tp = LD(&(x[WS(vs, 1) + WS(rs, 2)]), ms, &(x[WS(vs, 1)]));
			      Ti = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), Tc, Td));
			      Te = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), Td, Tc));
			      Ta = VSUB(T4, T7);
			      T8 = VADD(T4, T7);
			      Tq = LD(&(x[WS(vs, 1) + WS(rs, 3)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			      T1w = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), T1v, T1u));
			      T1A = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), T1u, T1v));
			      T1q = VADD(T1m, T1p);
			      T1s = VSUB(T1m, T1p);
			      Tm = LD(&(x[WS(vs, 1) + WS(rs, 1)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			      T9 = VFNMS(LDK(KP250000000), T8, T1);
			      Tn = LD(&(x[WS(vs, 1) + WS(rs, 4)]), ms, &(x[WS(vs, 1)]));
			      Tr = VADD(Tp, Tq);
			      Tx = VSUB(Tp, Tq);
			 }
			 {
			      V TJ, TK, TG, Tw, To, TH, T13, T14;
			      TF = LD(&(x[WS(vs, 2)]), ms, &(x[WS(vs, 2)]));
			      T1r = VFNMS(LDK(KP250000000), T1q, T1j);
			      TJ = LD(&(x[WS(vs, 2) + WS(rs, 2)]), ms, &(x[WS(vs, 2)]));
			      TK = LD(&(x[WS(vs, 2) + WS(rs, 3)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			      TG = LD(&(x[WS(vs, 2) + WS(rs, 1)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			      Tw = VSUB(Tm, Tn);
			      To = VADD(Tm, Tn);
			      TH = LD(&(x[WS(vs, 2) + WS(rs, 4)]), ms, &(x[WS(vs, 2)]));
			      TZ = LD(&(x[WS(vs, 3)]), ms, &(x[WS(vs, 3)]));
			      T13 = LD(&(x[WS(vs, 3) + WS(rs, 2)]), ms, &(x[WS(vs, 3)]));
			      T14 = LD(&(x[WS(vs, 3) + WS(rs, 3)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			      TR = VSUB(TJ, TK);
			      TL = VADD(TJ, TK);
			      Ty = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), Tx, Tw));
			      TC = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), Tw, Tx));
			      Ts = VADD(To, Tr);
			      Tu = VSUB(To, Tr);
			      TQ = VSUB(TG, TH);
			      TI = VADD(TG, TH);
			      T15 = VADD(T13, T14);
			      T1b = VSUB(T13, T14);
			      T10 = LD(&(x[WS(vs, 3) + WS(rs, 1)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			      T11 = LD(&(x[WS(vs, 3) + WS(rs, 4)]), ms, &(x[WS(vs, 3)]));
			      Tt = VFNMS(LDK(KP250000000), Ts, Tl);
			 }
		    }
	       }
	       {
		    V TO, T12, T1a, Th, T1z, TN, TM, T18, T17;
		    ST(&(x[0]), VADD(T1, T8), ms, &(x[0]));
		    TS = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), TR, TQ));
		    TW = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), TQ, TR));
		    TM = VADD(TI, TL);
		    TO = VSUB(TI, TL);
		    ST(&(x[WS(rs, 4)]), VADD(T1j, T1q), ms, &(x[0]));
		    T12 = VADD(T10, T11);
		    T1a = VSUB(T10, T11);
		    ST(&(x[WS(rs, 1)]), VADD(Tl, Ts), ms, &(x[WS(rs, 1)]));
		    Th = VFNMS(LDK(KP559016994), Ta, T9);
		    Tb = VFMA(LDK(KP559016994), Ta, T9);
		    T1t = VFMA(LDK(KP559016994), T1s, T1r);
		    T1z = VFNMS(LDK(KP559016994), T1s, T1r);
		    ST(&(x[WS(rs, 2)]), VADD(TF, TM), ms, &(x[0]));
		    TN = VFNMS(LDK(KP250000000), TM, TF);
		    {
			 V T16, Tk, Tj, T1C, T1B, TD, TE, TB;
			 TB = VFNMS(LDK(KP559016994), Tu, Tt);
			 Tv = VFMA(LDK(KP559016994), Tu, Tt);
			 T1g = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), T1a, T1b));
			 T1c = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), T1b, T1a));
			 T18 = VSUB(T12, T15);
			 T16 = VADD(T12, T15);
			 Tk = BYTW(&(W[TWVL * 4]), VFMAI(Ti, Th));
			 Tj = BYTW(&(W[TWVL * 2]), VFNMSI(Ti, Th));
			 T1C = BYTW(&(W[TWVL * 4]), VFMAI(T1A, T1z));
			 T1B = BYTW(&(W[TWVL * 2]), VFNMSI(T1A, T1z));
			 TD = BYTW(&(W[TWVL * 2]), VFNMSI(TC, TB));
			 TE = BYTW(&(W[TWVL * 4]), VFMAI(TC, TB));
			 ST(&(x[WS(rs, 3)]), VADD(TZ, T16), ms, &(x[WS(rs, 1)]));
			 T17 = VFNMS(LDK(KP250000000), T16, TZ);
			 ST(&(x[WS(vs, 3)]), Tk, ms, &(x[WS(vs, 3)]));
			 ST(&(x[WS(vs, 2)]), Tj, ms, &(x[WS(vs, 2)]));
			 ST(&(x[WS(vs, 3) + WS(rs, 4)]), T1C, ms, &(x[WS(vs, 3)]));
			 ST(&(x[WS(vs, 2) + WS(rs, 4)]), T1B, ms, &(x[WS(vs, 2)]));
			 ST(&(x[WS(vs, 2) + WS(rs, 1)]), TD, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 3) + WS(rs, 1)]), TE, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
		    }
		    TP = VFMA(LDK(KP559016994), TO, TN);
		    TV = VFNMS(LDK(KP559016994), TO, TN);
		    T1f = VFNMS(LDK(KP559016994), T18, T17);
		    T19 = VFMA(LDK(KP559016994), T18, T17);
	       }
	  }
	  TY = BYTW(&(W[TWVL * 4]), VFMAI(TW, TV));
	  TX = BYTW(&(W[TWVL * 2]), VFNMSI(TW, TV));
	  {
	       V T1i, T1h, TU, TT;
	       T1i = BYTW(&(W[TWVL * 4]), VFMAI(T1g, T1f));
	       T1h = BYTW(&(W[TWVL * 2]), VFNMSI(T1g, T1f));
	       TU = BYTW(&(W[TWVL * 6]), VFNMSI(TS, TP));
	       TT = BYTW(&(W[0]), VFMAI(TS, TP));
	       {
		    V Tg, Tf, TA, Tz;
		    Tg = BYTW(&(W[TWVL * 6]), VFNMSI(Te, Tb));
		    Tf = BYTW(&(W[0]), VFMAI(Te, Tb));
		    TA = BYTW(&(W[TWVL * 6]), VFNMSI(Ty, Tv));
		    Tz = BYTW(&(W[0]), VFMAI(Ty, Tv));
		    {
			 V T1e, T1d, T1y, T1x;
			 T1e = BYTW(&(W[TWVL * 6]), VFNMSI(T1c, T19));
			 T1d = BYTW(&(W[0]), VFMAI(T1c, T19));
			 T1y = BYTW(&(W[TWVL * 6]), VFNMSI(T1w, T1t));
			 T1x = BYTW(&(W[0]), VFMAI(T1w, T1t));
			 ST(&(x[WS(vs, 3) + WS(rs, 2)]), TY, ms, &(x[WS(vs, 3)]));
			 ST(&(x[WS(vs, 2) + WS(rs, 2)]), TX, ms, &(x[WS(vs, 2)]));
			 ST(&(x[WS(vs, 3) + WS(rs, 3)]), T1i, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 2) + WS(rs, 3)]), T1h, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 4) + WS(rs, 2)]), TU, ms, &(x[WS(vs, 4)]));
			 ST(&(x[WS(vs, 1) + WS(rs, 2)]), TT, ms, &(x[WS(vs, 1)]));
			 ST(&(x[WS(vs, 4)]), Tg, ms, &(x[WS(vs, 4)]));
			 ST(&(x[WS(vs, 1)]), Tf, ms, &(x[WS(vs, 1)]));
			 ST(&(x[WS(vs, 4) + WS(rs, 1)]), TA, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 1) + WS(rs, 1)]), Tz, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 4) + WS(rs, 3)]), T1e, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 1) + WS(rs, 3)]), T1d, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			 ST(&(x[WS(vs, 4) + WS(rs, 4)]), T1y, ms, &(x[WS(vs, 4)]));
			 ST(&(x[WS(vs, 1) + WS(rs, 4)]), T1x, ms, &(x[WS(vs, 1)]));
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
     {TW_NEXT, VL, 0}
};

static const ct_desc desc = { 5, "q1bv_5", twinstr, &GENUS, {55, 50, 45, 0}, 0, 0, 0 };

void X(codelet_q1bv_5) (planner *p) {
     X(kdft_difsq_register) (p, q1bv_5, &desc);
}
#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_twidsq_c -simd -compact -variables 4 -pipeline-latency 8 -n 5 -dif -name q1bv_5 -include q1b.h -sign 1 */

/*
 * This function contains 100 FP additions, 70 FP multiplications,
 * (or, 85 additions, 55 multiplications, 15 fused multiply/add),
 * 44 stack variables, 4 constants, and 50 memory accesses
 */
#include "q1b.h"

static void q1bv_5(R *ri, R *ii, const R *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
     DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
     DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
     DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
     INT m;
     R *x;
     x = ii;
     for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me; m = m + VL, x = x + (VL * ms), W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(vs)) {
	  V Tb, T7, Th, Ta, Tc, Td, T1t, T1p, T1z, T1s, T1u, T1v, Tv, Tr, TB;
	  V Tu, Tw, Tx, TP, TL, TV, TO, TQ, TR, T19, T15, T1f, T18, T1a, T1b;
	  {
	       V T6, T9, T3, T8;
	       Tb = LD(&(x[0]), ms, &(x[0]));
	       {
		    V T4, T5, T1, T2;
		    T4 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
		    T5 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
		    T6 = VSUB(T4, T5);
		    T9 = VADD(T4, T5);
		    T1 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
		    T2 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
		    T3 = VSUB(T1, T2);
		    T8 = VADD(T1, T2);
	       }
	       T7 = VBYI(VFMA(LDK(KP951056516), T3, VMUL(LDK(KP587785252), T6)));
	       Th = VBYI(VFNMS(LDK(KP951056516), T6, VMUL(LDK(KP587785252), T3)));
	       Ta = VMUL(LDK(KP559016994), VSUB(T8, T9));
	       Tc = VADD(T8, T9);
	       Td = VFNMS(LDK(KP250000000), Tc, Tb);
	  }
	  {
	       V T1o, T1r, T1l, T1q;
	       T1t = LD(&(x[WS(vs, 4)]), ms, &(x[WS(vs, 4)]));
	       {
		    V T1m, T1n, T1j, T1k;
		    T1m = LD(&(x[WS(vs, 4) + WS(rs, 2)]), ms, &(x[WS(vs, 4)]));
		    T1n = LD(&(x[WS(vs, 4) + WS(rs, 3)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
		    T1o = VSUB(T1m, T1n);
		    T1r = VADD(T1m, T1n);
		    T1j = LD(&(x[WS(vs, 4) + WS(rs, 1)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
		    T1k = LD(&(x[WS(vs, 4) + WS(rs, 4)]), ms, &(x[WS(vs, 4)]));
		    T1l = VSUB(T1j, T1k);
		    T1q = VADD(T1j, T1k);
	       }
	       T1p = VBYI(VFMA(LDK(KP951056516), T1l, VMUL(LDK(KP587785252), T1o)));
	       T1z = VBYI(VFNMS(LDK(KP951056516), T1o, VMUL(LDK(KP587785252), T1l)));
	       T1s = VMUL(LDK(KP559016994), VSUB(T1q, T1r));
	       T1u = VADD(T1q, T1r);
	       T1v = VFNMS(LDK(KP250000000), T1u, T1t);
	  }
	  {
	       V Tq, Tt, Tn, Ts;
	       Tv = LD(&(x[WS(vs, 1)]), ms, &(x[WS(vs, 1)]));
	       {
		    V To, Tp, Tl, Tm;
		    To = LD(&(x[WS(vs, 1) + WS(rs, 2)]), ms, &(x[WS(vs, 1)]));
		    Tp = LD(&(x[WS(vs, 1) + WS(rs, 3)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
		    Tq = VSUB(To, Tp);
		    Tt = VADD(To, Tp);
		    Tl = LD(&(x[WS(vs, 1) + WS(rs, 1)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
		    Tm = LD(&(x[WS(vs, 1) + WS(rs, 4)]), ms, &(x[WS(vs, 1)]));
		    Tn = VSUB(Tl, Tm);
		    Ts = VADD(Tl, Tm);
	       }
	       Tr = VBYI(VFMA(LDK(KP951056516), Tn, VMUL(LDK(KP587785252), Tq)));
	       TB = VBYI(VFNMS(LDK(KP951056516), Tq, VMUL(LDK(KP587785252), Tn)));
	       Tu = VMUL(LDK(KP559016994), VSUB(Ts, Tt));
	       Tw = VADD(Ts, Tt);
	       Tx = VFNMS(LDK(KP250000000), Tw, Tv);
	  }
	  {
	       V TK, TN, TH, TM;
	       TP = LD(&(x[WS(vs, 2)]), ms, &(x[WS(vs, 2)]));
	       {
		    V TI, TJ, TF, TG;
		    TI = LD(&(x[WS(vs, 2) + WS(rs, 2)]), ms, &(x[WS(vs, 2)]));
		    TJ = LD(&(x[WS(vs, 2) + WS(rs, 3)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
		    TK = VSUB(TI, TJ);
		    TN = VADD(TI, TJ);
		    TF = LD(&(x[WS(vs, 2) + WS(rs, 1)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
		    TG = LD(&(x[WS(vs, 2) + WS(rs, 4)]), ms, &(x[WS(vs, 2)]));
		    TH = VSUB(TF, TG);
		    TM = VADD(TF, TG);
	       }
	       TL = VBYI(VFMA(LDK(KP951056516), TH, VMUL(LDK(KP587785252), TK)));
	       TV = VBYI(VFNMS(LDK(KP951056516), TK, VMUL(LDK(KP587785252), TH)));
	       TO = VMUL(LDK(KP559016994), VSUB(TM, TN));
	       TQ = VADD(TM, TN);
	       TR = VFNMS(LDK(KP250000000), TQ, TP);
	  }
	  {
	       V T14, T17, T11, T16;
	       T19 = LD(&(x[WS(vs, 3)]), ms, &(x[WS(vs, 3)]));
	       {
		    V T12, T13, TZ, T10;
		    T12 = LD(&(x[WS(vs, 3) + WS(rs, 2)]), ms, &(x[WS(vs, 3)]));
		    T13 = LD(&(x[WS(vs, 3) + WS(rs, 3)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
		    T14 = VSUB(T12, T13);
		    T17 = VADD(T12, T13);
		    TZ = LD(&(x[WS(vs, 3) + WS(rs, 1)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
		    T10 = LD(&(x[WS(vs, 3) + WS(rs, 4)]), ms, &(x[WS(vs, 3)]));
		    T11 = VSUB(TZ, T10);
		    T16 = VADD(TZ, T10);
	       }
	       T15 = VBYI(VFMA(LDK(KP951056516), T11, VMUL(LDK(KP587785252), T14)));
	       T1f = VBYI(VFNMS(LDK(KP951056516), T14, VMUL(LDK(KP587785252), T11)));
	       T18 = VMUL(LDK(KP559016994), VSUB(T16, T17));
	       T1a = VADD(T16, T17);
	       T1b = VFNMS(LDK(KP250000000), T1a, T19);
	  }
	  ST(&(x[0]), VADD(Tb, Tc), ms, &(x[0]));
	  ST(&(x[WS(rs, 4)]), VADD(T1t, T1u), ms, &(x[0]));
	  ST(&(x[WS(rs, 2)]), VADD(TP, TQ), ms, &(x[0]));
	  ST(&(x[WS(rs, 3)]), VADD(T19, T1a), ms, &(x[WS(rs, 1)]));
	  ST(&(x[WS(rs, 1)]), VADD(Tv, Tw), ms, &(x[WS(rs, 1)]));
	  {
	       V Tj, Tk, Ti, T1B, T1C, T1A;
	       Ti = VSUB(Td, Ta);
	       Tj = BYTW(&(W[TWVL * 2]), VADD(Th, Ti));
	       Tk = BYTW(&(W[TWVL * 4]), VSUB(Ti, Th));
	       ST(&(x[WS(vs, 2)]), Tj, ms, &(x[WS(vs, 2)]));
	       ST(&(x[WS(vs, 3)]), Tk, ms, &(x[WS(vs, 3)]));
	       T1A = VSUB(T1v, T1s);
	       T1B = BYTW(&(W[TWVL * 2]), VADD(T1z, T1A));
	       T1C = BYTW(&(W[TWVL * 4]), VSUB(T1A, T1z));
	       ST(&(x[WS(vs, 2) + WS(rs, 4)]), T1B, ms, &(x[WS(vs, 2)]));
	       ST(&(x[WS(vs, 3) + WS(rs, 4)]), T1C, ms, &(x[WS(vs, 3)]));
	  }
	  {
	       V T1h, T1i, T1g, TD, TE, TC;
	       T1g = VSUB(T1b, T18);
	       T1h = BYTW(&(W[TWVL * 2]), VADD(T1f, T1g));
	       T1i = BYTW(&(W[TWVL * 4]), VSUB(T1g, T1f));
	       ST(&(x[WS(vs, 2) + WS(rs, 3)]), T1h, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
	       ST(&(x[WS(vs, 3) + WS(rs, 3)]), T1i, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
	       TC = VSUB(Tx, Tu);
	       TD = BYTW(&(W[TWVL * 2]), VADD(TB, TC));
	       TE = BYTW(&(W[TWVL * 4]), VSUB(TC, TB));
	       ST(&(x[WS(vs, 2) + WS(rs, 1)]), TD, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
	       ST(&(x[WS(vs, 3) + WS(rs, 1)]), TE, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
	  }
	  {
	       V TX, TY, TW, TT, TU, TS;
	       TW = VSUB(TR, TO);
	       TX = BYTW(&(W[TWVL * 2]), VADD(TV, TW));
	       TY = BYTW(&(W[TWVL * 4]), VSUB(TW, TV));
	       ST(&(x[WS(vs, 2) + WS(rs, 2)]), TX, ms, &(x[WS(vs, 2)]));
	       ST(&(x[WS(vs, 3) + WS(rs, 2)]), TY, ms, &(x[WS(vs, 3)]));
	       TS = VADD(TO, TR);
	       TT = BYTW(&(W[0]), VADD(TL, TS));
	       TU = BYTW(&(W[TWVL * 6]), VSUB(TS, TL));
	       ST(&(x[WS(vs, 1) + WS(rs, 2)]), TT, ms, &(x[WS(vs, 1)]));
	       ST(&(x[WS(vs, 4) + WS(rs, 2)]), TU, ms, &(x[WS(vs, 4)]));
	  }
	  {
	       V Tf, Tg, Te, Tz, TA, Ty;
	       Te = VADD(Ta, Td);
	       Tf = BYTW(&(W[0]), VADD(T7, Te));
	       Tg = BYTW(&(W[TWVL * 6]), VSUB(Te, T7));
	       ST(&(x[WS(vs, 1)]), Tf, ms, &(x[WS(vs, 1)]));
	       ST(&(x[WS(vs, 4)]), Tg, ms, &(x[WS(vs, 4)]));
	       Ty = VADD(Tu, Tx);
	       Tz = BYTW(&(W[0]), VADD(Tr, Ty));
	       TA = BYTW(&(W[TWVL * 6]), VSUB(Ty, Tr));
	       ST(&(x[WS(vs, 1) + WS(rs, 1)]), Tz, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
	       ST(&(x[WS(vs, 4) + WS(rs, 1)]), TA, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
	  }
	  {
	       V T1d, T1e, T1c, T1x, T1y, T1w;
	       T1c = VADD(T18, T1b);
	       T1d = BYTW(&(W[0]), VADD(T15, T1c));
	       T1e = BYTW(&(W[TWVL * 6]), VSUB(T1c, T15));
	       ST(&(x[WS(vs, 1) + WS(rs, 3)]), T1d, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
	       ST(&(x[WS(vs, 4) + WS(rs, 3)]), T1e, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
	       T1w = VADD(T1s, T1v);
	       T1x = BYTW(&(W[0]), VADD(T1p, T1w));
	       T1y = BYTW(&(W[TWVL * 6]), VSUB(T1w, T1p));
	       ST(&(x[WS(vs, 1) + WS(rs, 4)]), T1x, ms, &(x[WS(vs, 1)]));
	       ST(&(x[WS(vs, 4) + WS(rs, 4)]), T1y, ms, &(x[WS(vs, 4)]));
	  }
     }
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     VTW(0, 4),
     {TW_NEXT, VL, 0}
};

static const ct_desc desc = { 5, "q1bv_5", twinstr, &GENUS, {85, 55, 15, 0}, 0, 0, 0 };

void X(codelet_q1bv_5) (planner *p) {
     X(kdft_difsq_register) (p, q1bv_5, &desc);
}
#endif				/* HAVE_FMA */
