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
/* Generated on Mon Nov 10 20:34:05 EST 2008 */

#include "codelet-dft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_notw_c -fma -reorder-insns -schedule-for-pipeline -simd -compact -variables 4 -pipeline-latency 8 -sign 1 -n 13 -name n1bv_13 -include n1b.h */

/*
 * This function contains 88 FP additions, 63 FP multiplications,
 * (or, 31 additions, 6 multiplications, 57 fused multiply/add),
 * 96 stack variables, 23 constants, and 26 memory accesses
 */
#include "n1b.h"

static void n1bv_13(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs, INT ovs)
{
     DVK(KP904176221, +0.904176221990848204433795481776887926501523162);
     DVK(KP575140729, +0.575140729474003121368385547455453388461001608);
     DVK(KP300462606, +0.300462606288665774426601772289207995520941381);
     DVK(KP516520780, +0.516520780623489722840901288569017135705033622);
     DVK(KP522026385, +0.522026385161275033714027226654165028300441940);
     DVK(KP957805992, +0.957805992594665126462521754605754580515587217);
     DVK(KP600477271, +0.600477271932665282925769253334763009352012849);
     DVK(KP251768516, +0.251768516431883313623436926934233488546674281);
     DVK(KP503537032, +0.503537032863766627246873853868466977093348562);
     DVK(KP769338817, +0.769338817572980603471413688209101117038278899);
     DVK(KP859542535, +0.859542535098774820163672132761689612766401925);
     DVK(KP581704778, +0.581704778510515730456870384989698884939833902);
     DVK(KP853480001, +0.853480001859823990758994934970528322872359049);
     DVK(KP083333333, +0.083333333333333333333333333333333333333333333);
     DVK(KP226109445, +0.226109445035782405468510155372505010481906348);
     DVK(KP301479260, +0.301479260047709873958013540496673347309208464);
     DVK(KP686558370, +0.686558370781754340655719594850823015421401653);
     DVK(KP514918778, +0.514918778086315755491789696138117261566051239);
     DVK(KP038632954, +0.038632954644348171955506895830342264440241080);
     DVK(KP612264650, +0.612264650376756543746494474777125408779395514);
     DVK(KP302775637, +0.302775637731994646559610633735247973125648287);
     DVK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
     INT i;
     const R *xi;
     R *xo;
     xi = ii;
     xo = io;
     for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs), MAKE_VOLATILE_STRIDE(is), MAKE_VOLATILE_STRIDE(os)) {
	  V T1, T7, T2, Tg, Tf, TN, Th, Tq, Ta, Tj, T5, Tr, Tk;
	  T1 = LD(&(xi[0]), ivs, &(xi[0]));
	  {
	       V Td, Te, T8, T9, T3, T4;
	       Td = LD(&(xi[WS(is, 8)]), ivs, &(xi[0]));
	       Te = LD(&(xi[WS(is, 5)]), ivs, &(xi[WS(is, 1)]));
	       T7 = LD(&(xi[WS(is, 12)]), ivs, &(xi[0]));
	       T8 = LD(&(xi[WS(is, 10)]), ivs, &(xi[0]));
	       T9 = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
	       T2 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
	       T3 = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
	       T4 = LD(&(xi[WS(is, 9)]), ivs, &(xi[WS(is, 1)]));
	       Tg = LD(&(xi[WS(is, 11)]), ivs, &(xi[WS(is, 1)]));
	       Tf = VADD(Td, Te);
	       TN = VSUB(Td, Te);
	       Th = LD(&(xi[WS(is, 6)]), ivs, &(xi[0]));
	       Tq = VSUB(T8, T9);
	       Ta = VADD(T8, T9);
	       Tj = LD(&(xi[WS(is, 7)]), ivs, &(xi[WS(is, 1)]));
	       T5 = VADD(T3, T4);
	       Tr = VSUB(T4, T3);
	       Tk = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
	  }
	  {
	       V Tt, Ti, Ty, Tb, Ts, TQ, Tx, T6, Tu, Tl;
	       Tt = VSUB(Tg, Th);
	       Ti = VADD(Tg, Th);
	       Ty = VFMS(LDK(KP500000000), Ta, T7);
	       Tb = VADD(T7, Ta);
	       Ts = VSUB(Tq, Tr);
	       TQ = VADD(Tr, Tq);
	       Tx = VFNMS(LDK(KP500000000), T5, T2);
	       T6 = VADD(T2, T5);
	       Tu = VSUB(Tj, Tk);
	       Tl = VADD(Tj, Tk);
	       {
		    V TK, Tz, Tc, TX, Tv, TO, TL, Tm;
		    TK = VADD(Tx, Ty);
		    Tz = VSUB(Tx, Ty);
		    Tc = VADD(T6, Tb);
		    TX = VSUB(T6, Tb);
		    Tv = VSUB(Tt, Tu);
		    TO = VADD(Tt, Tu);
		    TL = VSUB(Ti, Tl);
		    Tm = VADD(Ti, Tl);
		    {
			 V TF, Tw, TP, TY, TT, TM, TA, Tn;
			 TF = VSUB(Ts, Tv);
			 Tw = VADD(Ts, Tv);
			 TP = VFNMS(LDK(KP500000000), TO, TN);
			 TY = VADD(TN, TO);
			 TT = VFNMS(LDK(KP866025403), TL, TK);
			 TM = VFMA(LDK(KP866025403), TL, TK);
			 TA = VFNMS(LDK(KP500000000), Tm, Tf);
			 Tn = VADD(Tf, Tm);
			 {
			      V T1f, T1n, TI, T18, T1k, T1c, TD, T17, T10, T1m, T16, T1e, TU, TR;
			      TU = VFNMS(LDK(KP866025403), TQ, TP);
			      TR = VFMA(LDK(KP866025403), TQ, TP);
			      {
				   V TZ, T15, TE, TB;
				   TZ = VFMA(LDK(KP302775637), TY, TX);
				   T15 = VFNMS(LDK(KP302775637), TX, TY);
				   TE = VSUB(Tz, TA);
				   TB = VADD(Tz, TA);
				   {
					V TH, To, TV, T13;
					TH = VSUB(Tc, Tn);
					To = VADD(Tc, Tn);
					TV = VFNMS(LDK(KP612264650), TU, TT);
					T13 = VFMA(LDK(KP612264650), TT, TU);
					{
					     V TS, T12, TG, T1b;
					     TS = VFNMS(LDK(KP038632954), TR, TM);
					     T12 = VFMA(LDK(KP038632954), TM, TR);
					     TG = VFNMS(LDK(KP514918778), TF, TE);
					     T1b = VFMA(LDK(KP686558370), TE, TF);
					     {
						  V TC, T1a, Tp, TW, T14;
						  TC = VFMA(LDK(KP301479260), TB, Tw);
						  T1a = VFNMS(LDK(KP226109445), Tw, TB);
						  Tp = VFNMS(LDK(KP083333333), To, T1);
						  ST(&(xo[0]), VADD(T1, To), ovs, &(xo[0]));
						  T1f = VFMA(LDK(KP853480001), TV, TS);
						  TW = VFNMS(LDK(KP853480001), TV, TS);
						  T1n = VFMA(LDK(KP853480001), T13, T12);
						  T14 = VFNMS(LDK(KP853480001), T13, T12);
						  TI = VFMA(LDK(KP581704778), TH, TG);
						  T18 = VFNMS(LDK(KP859542535), TG, TH);
						  T1k = VFMA(LDK(KP769338817), T1b, T1a);
						  T1c = VFNMS(LDK(KP769338817), T1b, T1a);
						  TD = VFMA(LDK(KP503537032), TC, Tp);
						  T17 = VFNMS(LDK(KP251768516), TC, Tp);
						  T10 = VMUL(LDK(KP600477271), VFMA(LDK(KP957805992), TZ, TW));
						  T1m = VFNMS(LDK(KP522026385), TW, TZ);
						  T16 = VMUL(LDK(KP600477271), VFMA(LDK(KP957805992), T15, T14));
						  T1e = VFNMS(LDK(KP522026385), T14, T15);
					     }
					}
				   }
			      }
			      {
				   V T1o, T1q, T1g, T1i, T1d, T1h, T1l, T1p;
				   {
					V T11, TJ, T19, T1j;
					T11 = VFMA(LDK(KP516520780), TI, TD);
					TJ = VFNMS(LDK(KP516520780), TI, TD);
					T19 = VFMA(LDK(KP300462606), T18, T17);
					T1j = VFNMS(LDK(KP300462606), T18, T17);
					T1o = VMUL(LDK(KP575140729), VFNMS(LDK(KP904176221), T1n, T1m));
					T1q = VMUL(LDK(KP575140729), VFMA(LDK(KP904176221), T1n, T1m));
					T1g = VMUL(LDK(KP575140729), VFMA(LDK(KP904176221), T1f, T1e));
					T1i = VMUL(LDK(KP575140729), VFNMS(LDK(KP904176221), T1f, T1e));
					ST(&(xo[WS(os, 12)]), VFMAI(T16, T11), ovs, &(xo[0]));
					ST(&(xo[WS(os, 1)]), VFNMSI(T16, T11), ovs, &(xo[WS(os, 1)]));
					ST(&(xo[WS(os, 8)]), VFNMSI(T10, TJ), ovs, &(xo[0]));
					ST(&(xo[WS(os, 5)]), VFMAI(T10, TJ), ovs, &(xo[WS(os, 1)]));
					T1d = VFNMS(LDK(KP503537032), T1c, T19);
					T1h = VFMA(LDK(KP503537032), T1c, T19);
					T1l = VFNMS(LDK(KP503537032), T1k, T1j);
					T1p = VFMA(LDK(KP503537032), T1k, T1j);
				   }
				   ST(&(xo[WS(os, 9)]), VFNMSI(T1g, T1d), ovs, &(xo[WS(os, 1)]));
				   ST(&(xo[WS(os, 4)]), VFMAI(T1g, T1d), ovs, &(xo[0]));
				   ST(&(xo[WS(os, 10)]), VFMAI(T1i, T1h), ovs, &(xo[0]));
				   ST(&(xo[WS(os, 3)]), VFNMSI(T1i, T1h), ovs, &(xo[WS(os, 1)]));
				   ST(&(xo[WS(os, 7)]), VFNMSI(T1o, T1l), ovs, &(xo[WS(os, 1)]));
				   ST(&(xo[WS(os, 6)]), VFMAI(T1o, T1l), ovs, &(xo[0]));
				   ST(&(xo[WS(os, 11)]), VFNMSI(T1q, T1p), ovs, &(xo[WS(os, 1)]));
				   ST(&(xo[WS(os, 2)]), VFMAI(T1q, T1p), ovs, &(xo[0]));
			      }
			 }
		    }
	       }
	  }
     }
}

static const kdft_desc desc = { 13, "n1bv_13", {31, 6, 57, 0}, &GENUS, 0, 0, 0, 0 };
void X(codelet_n1bv_13) (planner *p) {
     X(kdft_register) (p, n1bv_13, &desc);
}

#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_notw_c -simd -compact -variables 4 -pipeline-latency 8 -sign 1 -n 13 -name n1bv_13 -include n1b.h */

/*
 * This function contains 88 FP additions, 34 FP multiplications,
 * (or, 69 additions, 15 multiplications, 19 fused multiply/add),
 * 60 stack variables, 20 constants, and 26 memory accesses
 */
#include "n1b.h"

static void n1bv_13(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs, INT ovs)
{
     DVK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
     DVK(KP083333333, +0.083333333333333333333333333333333333333333333);
     DVK(KP075902986, +0.075902986037193865983102897245103540356428373);
     DVK(KP251768516, +0.251768516431883313623436926934233488546674281);
     DVK(KP132983124, +0.132983124607418643793760531921092974399165133);
     DVK(KP258260390, +0.258260390311744861420450644284508567852516811);
     DVK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
     DVK(KP300238635, +0.300238635966332641462884626667381504676006424);
     DVK(KP011599105, +0.011599105605768290721655456654083252189827041);
     DVK(KP256247671, +0.256247671582936600958684654061725059144125175);
     DVK(KP156891391, +0.156891391051584611046832726756003269660212636);
     DVK(KP174138601, +0.174138601152135905005660794929264742616964676);
     DVK(KP575140729, +0.575140729474003121368385547455453388461001608);
     DVK(KP503537032, +0.503537032863766627246873853868466977093348562);
     DVK(KP113854479, +0.113854479055790798974654345867655310534642560);
     DVK(KP265966249, +0.265966249214837287587521063842185948798330267);
     DVK(KP387390585, +0.387390585467617292130675966426762851778775217);
     DVK(KP300462606, +0.300462606288665774426601772289207995520941381);
     DVK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
     INT i;
     const R *xi;
     R *xo;
     xi = ii;
     xo = io;
     for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs), MAKE_VOLATILE_STRIDE(is), MAKE_VOLATILE_STRIDE(os)) {
	  V TW, Tb, Tm, Ts, TB, TR, TX, TK, TU, Tz, TC, TN, TT;
	  TW = LD(&(xi[0]), ivs, &(xi[0]));
	  {
	       V Te, TH, Ta, Tu, Tp, T5, Tt, To, Th, Tw, Tk, Tx, Tl, TI, Tc;
	       V Td, Tq, Tr;
	       Tc = LD(&(xi[WS(is, 8)]), ivs, &(xi[0]));
	       Td = LD(&(xi[WS(is, 5)]), ivs, &(xi[WS(is, 1)]));
	       Te = VSUB(Tc, Td);
	       TH = VADD(Tc, Td);
	       {
		    V T6, T7, T8, T9;
		    T6 = LD(&(xi[WS(is, 12)]), ivs, &(xi[0]));
		    T7 = LD(&(xi[WS(is, 10)]), ivs, &(xi[0]));
		    T8 = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
		    T9 = VADD(T7, T8);
		    Ta = VADD(T6, T9);
		    Tu = VFNMS(LDK(KP500000000), T9, T6);
		    Tp = VSUB(T7, T8);
	       }
	       {
		    V T1, T2, T3, T4;
		    T1 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
		    T2 = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
		    T3 = LD(&(xi[WS(is, 9)]), ivs, &(xi[WS(is, 1)]));
		    T4 = VADD(T2, T3);
		    T5 = VADD(T1, T4);
		    Tt = VFNMS(LDK(KP500000000), T4, T1);
		    To = VSUB(T2, T3);
	       }
	       {
		    V Tf, Tg, Ti, Tj;
		    Tf = LD(&(xi[WS(is, 11)]), ivs, &(xi[WS(is, 1)]));
		    Tg = LD(&(xi[WS(is, 6)]), ivs, &(xi[0]));
		    Th = VSUB(Tf, Tg);
		    Tw = VADD(Tf, Tg);
		    Ti = LD(&(xi[WS(is, 7)]), ivs, &(xi[WS(is, 1)]));
		    Tj = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
		    Tk = VSUB(Ti, Tj);
		    Tx = VADD(Ti, Tj);
	       }
	       Tl = VADD(Th, Tk);
	       TI = VADD(Tw, Tx);
	       Tb = VSUB(T5, Ta);
	       Tm = VADD(Te, Tl);
	       Tq = VMUL(LDK(KP866025403), VSUB(To, Tp));
	       Tr = VFNMS(LDK(KP500000000), Tl, Te);
	       Ts = VADD(Tq, Tr);
	       TB = VSUB(Tq, Tr);
	       {
		    V TP, TQ, TG, TJ;
		    TP = VADD(T5, Ta);
		    TQ = VADD(TH, TI);
		    TR = VMUL(LDK(KP300462606), VSUB(TP, TQ));
		    TX = VADD(TP, TQ);
		    TG = VADD(Tt, Tu);
		    TJ = VFNMS(LDK(KP500000000), TI, TH);
		    TK = VSUB(TG, TJ);
		    TU = VADD(TG, TJ);
	       }
	       {
		    V Tv, Ty, TL, TM;
		    Tv = VSUB(Tt, Tu);
		    Ty = VMUL(LDK(KP866025403), VSUB(Tw, Tx));
		    Tz = VSUB(Tv, Ty);
		    TC = VADD(Tv, Ty);
		    TL = VADD(To, Tp);
		    TM = VSUB(Th, Tk);
		    TN = VSUB(TL, TM);
		    TT = VADD(TL, TM);
	       }
	  }
	  ST(&(xo[0]), VADD(TW, TX), ovs, &(xo[0]));
	  {
	       V T1c, T1n, T11, T14, T17, T1k, Tn, TE, T18, T1j, TS, T1m, TZ, T1f, TA;
	       V TD;
	       {
		    V T1a, T1b, T12, T13;
		    T1a = VFMA(LDK(KP387390585), TN, VMUL(LDK(KP265966249), TK));
		    T1b = VFNMS(LDK(KP503537032), TU, VMUL(LDK(KP113854479), TT));
		    T1c = VSUB(T1a, T1b);
		    T1n = VADD(T1a, T1b);
		    T11 = VFMA(LDK(KP575140729), Tb, VMUL(LDK(KP174138601), Tm));
		    T12 = VFNMS(LDK(KP256247671), Tz, VMUL(LDK(KP156891391), Ts));
		    T13 = VFMA(LDK(KP011599105), TB, VMUL(LDK(KP300238635), TC));
		    T14 = VADD(T12, T13);
		    T17 = VSUB(T11, T14);
		    T1k = VMUL(LDK(KP1_732050807), VSUB(T12, T13));
	       }
	       Tn = VFNMS(LDK(KP575140729), Tm, VMUL(LDK(KP174138601), Tb));
	       TA = VFMA(LDK(KP256247671), Ts, VMUL(LDK(KP156891391), Tz));
	       TD = VFNMS(LDK(KP011599105), TC, VMUL(LDK(KP300238635), TB));
	       TE = VADD(TA, TD);
	       T18 = VMUL(LDK(KP1_732050807), VSUB(TD, TA));
	       T1j = VSUB(Tn, TE);
	       {
		    V TO, T1e, TV, TY, T1d;
		    TO = VFNMS(LDK(KP132983124), TN, VMUL(LDK(KP258260390), TK));
		    T1e = VSUB(TR, TO);
		    TV = VFMA(LDK(KP251768516), TT, VMUL(LDK(KP075902986), TU));
		    TY = VFNMS(LDK(KP083333333), TX, TW);
		    T1d = VSUB(TY, TV);
		    TS = VFMA(LDK(KP2_000000000), TO, TR);
		    T1m = VADD(T1e, T1d);
		    TZ = VFMA(LDK(KP2_000000000), TV, TY);
		    T1f = VSUB(T1d, T1e);
	       }
	       {
		    V TF, T10, T1l, T1o;
		    TF = VBYI(VFMA(LDK(KP2_000000000), TE, Tn));
		    T10 = VADD(TS, TZ);
		    ST(&(xo[WS(os, 1)]), VADD(TF, T10), ovs, &(xo[WS(os, 1)]));
		    ST(&(xo[WS(os, 12)]), VSUB(T10, TF), ovs, &(xo[0]));
		    {
			 V T15, T16, T1p, T1q;
			 T15 = VBYI(VFMA(LDK(KP2_000000000), T14, T11));
			 T16 = VSUB(TZ, TS);
			 ST(&(xo[WS(os, 5)]), VADD(T15, T16), ovs, &(xo[WS(os, 1)]));
			 ST(&(xo[WS(os, 8)]), VSUB(T16, T15), ovs, &(xo[0]));
			 T1p = VADD(T1n, T1m);
			 T1q = VBYI(VADD(T1j, T1k));
			 ST(&(xo[WS(os, 4)]), VSUB(T1p, T1q), ovs, &(xo[0]));
			 ST(&(xo[WS(os, 9)]), VADD(T1q, T1p), ovs, &(xo[WS(os, 1)]));
		    }
		    T1l = VBYI(VSUB(T1j, T1k));
		    T1o = VSUB(T1m, T1n);
		    ST(&(xo[WS(os, 3)]), VADD(T1l, T1o), ovs, &(xo[WS(os, 1)]));
		    ST(&(xo[WS(os, 10)]), VSUB(T1o, T1l), ovs, &(xo[0]));
		    {
			 V T1h, T1i, T19, T1g;
			 T1h = VBYI(VADD(T18, T17));
			 T1i = VSUB(T1f, T1c);
			 ST(&(xo[WS(os, 6)]), VADD(T1h, T1i), ovs, &(xo[0]));
			 ST(&(xo[WS(os, 7)]), VSUB(T1i, T1h), ovs, &(xo[WS(os, 1)]));
			 T19 = VBYI(VSUB(T17, T18));
			 T1g = VADD(T1c, T1f);
			 ST(&(xo[WS(os, 2)]), VADD(T19, T1g), ovs, &(xo[0]));
			 ST(&(xo[WS(os, 11)]), VSUB(T1g, T19), ovs, &(xo[WS(os, 1)]));
		    }
	       }
	  }
     }
}

static const kdft_desc desc = { 13, "n1bv_13", {69, 15, 19, 0}, &GENUS, 0, 0, 0, 0 };
void X(codelet_n1bv_13) (planner *p) {
     X(kdft_register) (p, n1bv_13, &desc);
}

#endif				/* HAVE_FMA */
