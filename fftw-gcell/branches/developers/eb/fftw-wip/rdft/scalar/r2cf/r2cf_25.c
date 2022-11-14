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
/* Generated on Mon Nov 10 20:45:34 EST 2008 */

#include "codelet-rdft.h"

#ifdef HAVE_FMA

/* Generated by: ../../../genfft/gen_r2cf -fma -reorder-insns -schedule-for-pipeline -compact -variables 4 -pipeline-latency 4 -n 25 -name r2cf_25 -include r2cf.h */

/*
 * This function contains 200 FP additions, 168 FP multiplications,
 * (or, 44 additions, 12 multiplications, 156 fused multiply/add),
 * 157 stack variables, 66 constants, and 50 memory accesses
 */
#include "r2cf.h"

static void r2cf_25(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
     DK(KP792626838, +0.792626838241819413632131824093538848057784557);
     DK(KP876091699, +0.876091699473550838204498029706869638173524346);
     DK(KP809385824, +0.809385824416008241660603814668679683846476688);
     DK(KP860541664, +0.860541664367944677098261680920518816412804187);
     DK(KP681693190, +0.681693190061530575150324149145440022633095390);
     DK(KP560319534, +0.560319534973832390111614715371676131169633784);
     DK(KP997675361, +0.997675361079556513670859573984492383596555031);
     DK(KP237294955, +0.237294955877110315393888866460840817927895961);
     DK(KP897376177, +0.897376177523557693138608077137219684419427330);
     DK(KP923225144, +0.923225144846402650453449441572664695995209956);
     DK(KP956723877, +0.956723877038460305821989399535483155872969262);
     DK(KP949179823, +0.949179823508441261575555465843363271711583843);
     DK(KP669429328, +0.669429328479476605641803240971985825917022098);
     DK(KP570584518, +0.570584518783621657366766175430996792655723863);
     DK(KP262346850, +0.262346850930607871785420028382979691334784273);
     DK(KP876306680, +0.876306680043863587308115903922062583399064238);
     DK(KP906616052, +0.906616052148196230441134447086066874408359177);
     DK(KP683113946, +0.683113946453479238701949862233725244439656928);
     DK(KP559154169, +0.559154169276087864842202529084232643714075927);
     DK(KP921078979, +0.921078979742360627699756128143719920817673854);
     DK(KP904508497, +0.904508497187473712051146708591409529430077295);
     DK(KP999754674, +0.999754674276473633366203429228112409535557487);
     DK(KP968583161, +0.968583161128631119490168375464735813836012403);
     DK(KP242145790, +0.242145790282157779872542093866183953459003101);
     DK(KP904730450, +0.904730450839922351881287709692877908104763647);
     DK(KP845997307, +0.845997307939530944175097360758058292389769300);
     DK(KP855719849, +0.855719849902058969314654733608091555096772472);
     DK(KP982009705, +0.982009705009746369461829878184175962711969869);
     DK(KP916574801, +0.916574801383451584742370439148878693530976769);
     DK(KP690983005, +0.690983005625052575897706582817180941139845410);
     DK(KP952936919, +0.952936919628306576880750665357914584765951388);
     DK(KP998026728, +0.998026728428271561952336806863450553336905220);
     DK(KP831864738, +0.831864738706457140726048799369896829771167132);
     DK(KP803003575, +0.803003575438660414833440593570376004635464850);
     DK(KP522616830, +0.522616830205754336872861364785224694908468440);
     DK(KP829049696, +0.829049696159252993975487806364305442437946767);
     DK(KP999544308, +0.999544308746292983948881682379742149196758193);
     DK(KP772036680, +0.772036680810363904029489473607579825330539880);
     DK(KP763932022, +0.763932022500210303590826331268723764559381640);
     DK(KP992114701, +0.992114701314477831049793042785778521453036709);
     DK(KP447417479, +0.447417479732227551498980015410057305749330693);
     DK(KP734762448, +0.734762448793050413546343770063151342619912334);
     DK(KP894834959, +0.894834959464455102997960030820114611498661386);
     DK(KP867381224, +0.867381224396525206773171885031575671309956167);
     DK(KP958953096, +0.958953096729998668045963838399037225970891871);
     DK(KP912575812, +0.912575812670962425556968549836277086778922727);
     DK(KP951056516, +0.951056516295153572116439333379382143405698634);
     DK(KP244189809, +0.244189809627953270309879511234821255780225091);
     DK(KP269969613, +0.269969613759572083574752974412347470060951301);
     DK(KP522847744, +0.522847744331509716623755382187077770911012542);
     DK(KP578046249, +0.578046249379945007321754579646815604023525655);
     DK(KP603558818, +0.603558818296015001454675132653458027918768137);
     DK(KP667278218, +0.667278218140296670899089292254759909713898805);
     DK(KP447533225, +0.447533225982656890041886979663652563063114397);
     DK(KP494780565, +0.494780565770515410344588413655324772219443730);
     DK(KP987388751, +0.987388751065621252324603216482382109400433949);
     DK(KP893101515, +0.893101515366181661711202267938416198338079437);
     DK(KP132830569, +0.132830569247582714407653942074819768844536507);
     DK(KP120146378, +0.120146378570687701782758537356596213647956445);
     DK(KP059835404, +0.059835404262124915169548397419498386427871950);
     DK(KP066152395, +0.066152395967733048213034281011006031460903353);
     DK(KP786782374, +0.786782374965295178365099601674911834788448471);
     DK(KP869845200, +0.869845200362138853122720822420327157933056305);
     DK(KP559016994, +0.559016994374947424102293417182819058860154590);
     DK(KP250000000, +0.250000000000000000000000000000000000000000000);
     DK(KP618033988, +0.618033988749894848204586834365638117720309180);
     INT i;
     for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(csr), MAKE_VOLATILE_STRIDE(csi)) {
	  E T2H, T2w, T2x, T2A, T2C, T2v, T2M, T2y, T2B, T2N;
	  {
	       E T2u, TJ, T1O, T39, T2t, TB, T21, T1M, T2e, T26, T1B, T1r, T1k, T1c, T9;
	       E T1X, T1R, T2k, T29, T1z, T1v, T1h, TX, Ti, T13, T2a, T2j, T1U, T1Y, TQ;
	       E T1g, T1u, T1y, T12, Ts, T11, T1I;
	       {
		    E Tt, Tw, T16, Tx, Ty;
		    {
			 E T2p, TG, TH, TD, TE, TI, T2r;
			 T2p = R0[0];
			 TG = R0[WS(rs, 5)];
			 TH = R1[WS(rs, 7)];
			 TD = R1[WS(rs, 2)];
			 TE = R0[WS(rs, 10)];
			 Tt = R1[WS(rs, 1)];
			 TI = TG - TH;
			 T2r = TG + TH;
			 {
			      E TF, T2q, Tu, Tv, T2s;
			      TF = TD - TE;
			      T2q = TD + TE;
			      Tu = R0[WS(rs, 4)];
			      Tv = R1[WS(rs, 11)];
			      T2u = T2q - T2r;
			      T2s = T2q + T2r;
			      TJ = FMA(KP618033988, TI, TF);
			      T1O = FNMS(KP618033988, TF, TI);
			      T39 = T2p + T2s;
			      T2t = FNMS(KP250000000, T2s, T2p);
			      Tw = Tu + Tv;
			      T16 = Tv - Tu;
			      Tx = R1[WS(rs, 6)];
			      Ty = R0[WS(rs, 9)];
			 }
		    }
		    {
			 E T1P, TW, TS, TR;
			 {
			      E T1, T5, T1L, T18, T1a, TA, T4, TU, T6, T19;
			      T1 = R0[WS(rs, 2)];
			      {
				   E T2, T17, Tz, T3;
				   T2 = R1[WS(rs, 4)];
				   T17 = Tx - Ty;
				   Tz = Tx + Ty;
				   T3 = R0[WS(rs, 12)];
				   T5 = R0[WS(rs, 7)];
				   T1L = FMA(KP618033988, T16, T17);
				   T18 = FNMS(KP618033988, T17, T16);
				   T1a = Tz - Tw;
				   TA = Tw + Tz;
				   T4 = T2 + T3;
				   TU = T3 - T2;
				   T6 = R1[WS(rs, 9)];
			      }
			      TB = Tt + TA;
			      T19 = FNMS(KP250000000, TA, Tt);
			      {
				   E T7, TV, T1b, T1K, T8;
				   T7 = T5 + T6;
				   TV = T5 - T6;
				   T1b = FNMS(KP559016994, T1a, T19);
				   T1K = FMA(KP559016994, T1a, T19);
				   T1P = FMA(KP618033988, TU, TV);
				   TW = FNMS(KP618033988, TV, TU);
				   TS = T4 - T7;
				   T8 = T4 + T7;
				   T21 = FMA(KP869845200, T1K, T1L);
				   T1M = FNMS(KP786782374, T1L, T1K);
				   T2e = FMA(KP066152395, T1K, T1L);
				   T26 = FNMS(KP059835404, T1L, T1K);
				   T1B = FMA(KP120146378, T18, T1b);
				   T1r = FNMS(KP132830569, T1b, T18);
				   T1k = FMA(KP893101515, T18, T1b);
				   T1c = FNMS(KP987388751, T1b, T18);
				   T9 = T1 + T8;
				   TR = FMS(KP250000000, T8, T1);
			      }
			 }
			 {
			      E Ta, Te, TK, Td, Tf;
			      Ta = R1[0];
			      {
				   E Tb, Tc, T1Q, TT;
				   Tb = R0[WS(rs, 3)];
				   Tc = R1[WS(rs, 10)];
				   T1Q = FMA(KP559016994, TS, TR);
				   TT = FNMS(KP559016994, TS, TR);
				   Te = R1[WS(rs, 5)];
				   TK = Tb - Tc;
				   Td = Tb + Tc;
				   T1X = FNMS(KP120146378, T1P, T1Q);
				   T1R = FMA(KP132830569, T1Q, T1P);
				   T2k = FMA(KP494780565, T1Q, T1P);
				   T29 = FNMS(KP447533225, T1P, T1Q);
				   T1z = FMA(KP869845200, TT, TW);
				   T1v = FNMS(KP786782374, TW, TT);
				   T1h = FNMS(KP667278218, TT, TW);
				   TX = FMA(KP603558818, TW, TT);
				   Tf = R0[WS(rs, 8)];
			      }
			      {
				   E Tk, T1S, TM, TO, Tn, TZ, TN, T10, Tq, To, Th, Tp, TP, T1T, Tr;
				   Tk = R0[WS(rs, 1)];
				   {
					E Tl, TL, Tg, Tm;
					Tl = R1[WS(rs, 3)];
					TL = Tf - Te;
					Tg = Te + Tf;
					Tm = R0[WS(rs, 11)];
					To = R0[WS(rs, 6)];
					T1S = FMA(KP618033988, TK, TL);
					TM = FNMS(KP618033988, TL, TK);
					TO = Td - Tg;
					Th = Td + Tg;
					Tn = Tl + Tm;
					TZ = Tm - Tl;
					Tp = R1[WS(rs, 8)];
				   }
				   Ti = Ta + Th;
				   TN = FNMS(KP250000000, Th, Ta);
				   T10 = Tp - To;
				   Tq = To + Tp;
				   TP = FMA(KP559016994, TO, TN);
				   T1T = FNMS(KP559016994, TO, TN);
				   Tr = Tn + Tq;
				   T13 = Tn - Tq;
				   T2a = FMA(KP578046249, T1T, T1S);
				   T2j = FNMS(KP522847744, T1S, T1T);
				   T1U = FNMS(KP987388751, T1T, T1S);
				   T1Y = FMA(KP893101515, T1S, T1T);
				   TQ = FMA(KP269969613, TP, TM);
				   T1g = FNMS(KP244189809, TM, TP);
				   T1u = FNMS(KP603558818, TM, TP);
				   T1y = FMA(KP667278218, TP, TM);
				   T12 = FMS(KP250000000, Tr, Tk);
				   Ts = Tk + Tr;
				   T11 = FMA(KP618033988, T10, TZ);
				   T1I = FNMS(KP618033988, TZ, T10);
			      }
			 }
		    }
	       }
	       {
		    E T2f, T27, T1j, T15, T2K, T2J, T2I, T2T, T1Z, T2X, T1N, T1V, T2W, T2U, T22;
		    E T1G;
		    {
			 E T3a, T3b, T20, T1J, T1C, T1s;
			 {
			      E Tj, TC, T1H, T14;
			      T3a = T9 + Ti;
			      Tj = T9 - Ti;
			      TC = Ts - TB;
			      T3b = Ts + TB;
			      T1H = FMA(KP559016994, T13, T12);
			      T14 = FNMS(KP559016994, T13, T12);
			      Ci[WS(csi, 10)] = KP951056516 * (FMA(KP618033988, Tj, TC));
			      Ci[WS(csi, 5)] = KP951056516 * (FNMS(KP618033988, TC, Tj));
			      T20 = FNMS(KP066152395, T1H, T1I);
			      T1J = FMA(KP059835404, T1I, T1H);
			      T2f = FMA(KP667278218, T1H, T1I);
			      T27 = FNMS(KP603558818, T1I, T1H);
			      T1C = FNMS(KP494780565, T14, T11);
			      T1s = FMA(KP447533225, T11, T14);
			      T1j = FNMS(KP522847744, T11, T14);
			      T15 = FMA(KP578046249, T14, T11);
			 }
			 {
			      E T1A, T1t, T1w, T3c, T3e, T1D, T1x, T3d, T1E, T1F;
			      T1A = FNMS(KP912575812, T1z, T1y);
			      T2K = FMA(KP912575812, T1z, T1y);
			      T2J = FNMS(KP958953096, T1s, T1r);
			      T1t = FMA(KP958953096, T1s, T1r);
			      T1w = FMA(KP912575812, T1v, T1u);
			      T2H = FNMS(KP912575812, T1v, T1u);
			      T3c = T3a + T3b;
			      T3e = T3a - T3b;
			      T2I = FMA(KP867381224, T1C, T1B);
			      T1D = FNMS(KP867381224, T1C, T1B);
			      T1x = FNMS(KP894834959, T1w, T1t);
			      T2T = FMA(KP734762448, T1Y, T1X);
			      T1Z = FNMS(KP734762448, T1Y, T1X);
			      T3d = FNMS(KP250000000, T3c, T39);
			      Cr[0] = T3c + T39;
			      T1E = FMA(KP447417479, T1w, T1D);
			      Ci[WS(csi, 4)] = KP951056516 * (FMA(KP992114701, T1x, TJ));
			      Cr[WS(csr, 10)] = FNMS(KP559016994, T3e, T3d);
			      Cr[WS(csr, 5)] = FMA(KP559016994, T3e, T3d);
			      T1F = FMA(KP763932022, T1E, T1t);
			      T2X = FMA(KP772036680, T1M, T1J);
			      T1N = FNMS(KP772036680, T1M, T1J);
			      T1V = FMA(KP734762448, T1U, T1R);
			      T2W = FNMS(KP734762448, T1U, T1R);
			      T2U = FNMS(KP772036680, T21, T20);
			      T22 = FMA(KP772036680, T21, T20);
			      T1G = FMA(KP999544308, T1F, T1A);
			 }
		    }
		    {
			 E T1i, T1l, T2l, T2R, T2g, T2Q, T28, T32, T1f, T1n, T1p, T33, T2b;
			 {
			      E T24, TY, T1d, T1W, T23, T25, T1m, T1e;
			      T2w = FMA(KP829049696, T1h, T1g);
			      T1i = FNMS(KP829049696, T1h, T1g);
			      T1W = FNMS(KP992114701, T1V, T1O);
			      T23 = FNMS(KP522616830, T1V, T22);
			      Ci[WS(csi, 9)] = KP951056516 * (FNMS(KP803003575, T1G, TJ));
			      T2x = FNMS(KP831864738, T1k, T1j);
			      T1l = FMA(KP831864738, T1k, T1j);
			      Ci[WS(csi, 3)] = KP998026728 * (FNMS(KP952936919, T1W, T1N));
			      T24 = FMA(KP690983005, T23, T1N);
			      TY = FNMS(KP916574801, TX, TQ);
			      T2A = FMA(KP916574801, TX, TQ);
			      T2C = FNMS(KP831864738, T1c, T15);
			      T1d = FMA(KP831864738, T1c, T15);
			      T2l = FNMS(KP982009705, T2k, T2j);
			      T2R = FMA(KP982009705, T2k, T2j);
			      T25 = FNMS(KP855719849, T24, T1Z);
			      T2g = FMA(KP845997307, T2f, T2e);
			      T2Q = FNMS(KP845997307, T2f, T2e);
			      T1m = FMA(KP904730450, T1d, TY);
			      T1e = FNMS(KP904730450, T1d, TY);
			      Ci[WS(csi, 8)] = -(KP951056516 * (FNMS(KP992114701, T25, T1O)));
			      T28 = FNMS(KP845997307, T27, T26);
			      T32 = FMA(KP845997307, T27, T26);
			      T1f = FNMS(KP242145790, T1e, TJ);
			      Ci[WS(csi, 1)] = -(KP951056516 * (FMA(KP968583161, T1e, TJ)));
			      T1n = FNMS(KP999754674, T1m, T1l);
			      T1p = FNMS(KP904508497, T1m, T1i);
			      T33 = FMA(KP921078979, T2a, T29);
			      T2b = FNMS(KP921078979, T2a, T29);
			 }
			 {
			      E T2P, T2Z, T2V, T2O;
			      {
				   E T2d, T2n, T2i, T2Y, T2m, T2o;
				   T2P = FNMS(KP559016994, T2u, T2t);
				   T2v = FMA(KP559016994, T2u, T2t);
				   {
					E T1o, T1q, T2h, T2c;
					T1o = FNMS(KP559154169, T1n, T1i);
					T1q = FMA(KP683113946, T1p, T1l);
					T2h = FMA(KP906616052, T2b, T28);
					T2c = FNMS(KP906616052, T2b, T28);
					Ci[WS(csi, 6)] = -(KP951056516 * (FMA(KP968583161, T1o, T1f)));
					Ci[WS(csi, 11)] = -(KP951056516 * (FMA(KP876306680, T1q, T1f)));
					T2d = FMA(KP262346850, T2c, T1O);
					Ci[WS(csi, 2)] = -(KP998026728 * (FNMS(KP952936919, T1O, T2c)));
					T2n = T2g + T2h;
					T2i = FMA(KP618033988, T2h, T2g);
				   }
				   T2m = FMA(KP570584518, T2l, T2i);
				   T2o = FNMS(KP669429328, T2n, T2l);
				   Ci[WS(csi, 12)] = KP951056516 * (FNMS(KP949179823, T2m, T2d));
				   Ci[WS(csi, 7)] = KP951056516 * (FNMS(KP876306680, T2o, T2d));
				   T2V = FMA(KP956723877, T2U, T2T);
				   T2Y = FMA(KP522616830, T2T, T2X);
				   T2Z = FNMS(KP763932022, T2Y, T2U);
			      }
			      Cr[WS(csr, 3)] = FMA(KP992114701, T2V, T2P);
			      {
				   E T30, T34, T2S, T31, T35;
				   T30 = FMA(KP855719849, T2Z, T2W);
				   T34 = FNMS(KP923225144, T2R, T2Q);
				   T2S = FMA(KP923225144, T2R, T2Q);
				   Cr[WS(csr, 8)] = FNMS(KP897376177, T30, T2P);
				   T31 = FNMS(KP237294955, T2S, T2P);
				   Cr[WS(csr, 2)] = FMA(KP949179823, T2S, T2P);
				   T35 = FNMS(KP997675361, T34, T33);
				   {
					E T37, T36, T38, T2L;
					T37 = FNMS(KP904508497, T34, T32);
					T36 = FMA(KP560319534, T35, T32);
					T38 = FNMS(KP681693190, T37, T33);
					Cr[WS(csr, 12)] = FNMS(KP949179823, T36, T31);
					Cr[WS(csr, 7)] = FNMS(KP860541664, T38, T31);
					T2O = FNMS(KP809385824, T2K, T2I);
					T2L = FNMS(KP447417479, T2K, T2J);
					T2M = FNMS(KP690983005, T2L, T2I);
				   }
			      }
			      Cr[WS(csr, 4)] = FNMS(KP992114701, T2O, T2v);
			 }
		    }
	       }
	  }
	  T2y = FNMS(KP904730450, T2x, T2w);
	  T2B = FMA(KP904730450, T2x, T2w);
	  T2N = FNMS(KP999544308, T2M, T2H);
	  {
	       E T2z, T2D, T2F, T2E, T2G;
	       T2z = FNMS(KP242145790, T2y, T2v);
	       Cr[WS(csr, 1)] = FMA(KP968583161, T2y, T2v);
	       T2D = FMA(KP904730450, T2C, T2B);
	       T2F = T2A + T2B;
	       Cr[WS(csr, 9)] = FNMS(KP803003575, T2N, T2v);
	       T2E = FNMS(KP618033988, T2D, T2A);
	       T2G = FMA(KP683113946, T2F, T2C);
	       Cr[WS(csr, 6)] = FNMS(KP876091699, T2E, T2z);
	       Cr[WS(csr, 11)] = FNMS(KP792626838, T2G, T2z);
	  }
     }
}

static const kr2c_desc desc = { 25, "r2cf_25", {44, 12, 156, 0}, &GENUS };

void X(codelet_r2cf_25) (planner *p) {
     X(kr2c_register) (p, r2cf_25, &desc);
}

#else				/* HAVE_FMA */

/* Generated by: ../../../genfft/gen_r2cf -compact -variables 4 -pipeline-latency 4 -n 25 -name r2cf_25 -include r2cf.h */

/*
 * This function contains 200 FP additions, 140 FP multiplications,
 * (or, 117 additions, 57 multiplications, 83 fused multiply/add),
 * 101 stack variables, 40 constants, and 50 memory accesses
 */
#include "r2cf.h"

static void r2cf_25(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
     DK(KP998026728, +0.998026728428271561952336806863450553336905220);
     DK(KP125581039, +0.125581039058626752152356449131262266244969664);
     DK(KP1_996053456, +1.996053456856543123904673613726901106673810439);
     DK(KP062790519, +0.062790519529313376076178224565631133122484832);
     DK(KP809016994, +0.809016994374947424102293417182819058860154590);
     DK(KP309016994, +0.309016994374947424102293417182819058860154590);
     DK(KP1_369094211, +1.369094211857377347464566715242418539779038465);
     DK(KP728968627, +0.728968627421411523146730319055259111372571664);
     DK(KP963507348, +0.963507348203430549974383005744259307057084020);
     DK(KP876306680, +0.876306680043863587308115903922062583399064238);
     DK(KP497379774, +0.497379774329709576484567492012895936835134813);
     DK(KP968583161, +0.968583161128631119490168375464735813836012403);
     DK(KP684547105, +0.684547105928688673732283357621209269889519233);
     DK(KP1_457937254, +1.457937254842823046293460638110518222745143328);
     DK(KP481753674, +0.481753674101715274987191502872129653528542010);
     DK(KP1_752613360, +1.752613360087727174616231807844125166798128477);
     DK(KP248689887, +0.248689887164854788242283746006447968417567406);
     DK(KP1_937166322, +1.937166322257262238980336750929471627672024806);
     DK(KP992114701, +0.992114701314477831049793042785778521453036709);
     DK(KP250666467, +0.250666467128608490746237519633017587885836494);
     DK(KP425779291, +0.425779291565072648862502445744251703979973042);
     DK(KP1_809654104, +1.809654104932039055427337295865395187940827822);
     DK(KP1_274847979, +1.274847979497379420353425623352032390869834596);
     DK(KP770513242, +0.770513242775789230803009636396177847271667672);
     DK(KP844327925, +0.844327925502015078548558063966681505381659241);
     DK(KP1_071653589, +1.071653589957993236542617535735279956127150691);
     DK(KP125333233, +0.125333233564304245373118759816508793942918247);
     DK(KP1_984229402, +1.984229402628955662099586085571557042906073418);
     DK(KP904827052, +0.904827052466019527713668647932697593970413911);
     DK(KP851558583, +0.851558583130145297725004891488503407959946084);
     DK(KP637423989, +0.637423989748689710176712811676016195434917298);
     DK(KP1_541026485, +1.541026485551578461606019272792355694543335344);
     DK(KP535826794, +0.535826794978996618271308767867639978063575346);
     DK(KP1_688655851, +1.688655851004030157097116127933363010763318483);
     DK(KP293892626, +0.293892626146236564584352977319536384298826219);
     DK(KP475528258, +0.475528258147576786058219666689691071702849317);
     DK(KP250000000, +0.250000000000000000000000000000000000000000000);
     DK(KP559016994, +0.559016994374947424102293417182819058860154590);
     DK(KP587785252, +0.587785252292473129168705954639072768597652438);
     DK(KP951056516, +0.951056516295153572116439333379382143405698634);
     INT i;
     for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(rs), MAKE_VOLATILE_STRIDE(csr), MAKE_VOLATILE_STRIDE(csi)) {
	  E T8, T1j, T1V, T1l, T7, T9, Ta, T12, T2u, T1O, T19, T1P, Ti, T2r, T1K;
	  E Tp, T1L, Tx, T2q, T1H, TE, T1I, TN, T2t, T1R, TU, T1S, T6, T1k, T3;
	  E T2s, T2v;
	  T8 = R0[0];
	  {
	       E T4, T5, T1, T2;
	       T4 = R0[WS(rs, 5)];
	       T5 = R1[WS(rs, 7)];
	       T6 = T4 + T5;
	       T1k = T4 - T5;
	       T1 = R1[WS(rs, 2)];
	       T2 = R0[WS(rs, 10)];
	       T3 = T1 + T2;
	       T1j = T1 - T2;
	  }
	  T1V = KP951056516 * T1k;
	  T1l = FMA(KP951056516, T1j, KP587785252 * T1k);
	  T7 = KP559016994 * (T3 - T6);
	  T9 = T3 + T6;
	  Ta = FNMS(KP250000000, T9, T8);
	  {
	       E T16, T13, T14, TY, T17, T11, T15, T18;
	       T16 = R1[WS(rs, 1)];
	       {
		    E TW, TX, TZ, T10;
		    TW = R0[WS(rs, 4)];
		    TX = R1[WS(rs, 11)];
		    T13 = TW + TX;
		    TZ = R1[WS(rs, 6)];
		    T10 = R0[WS(rs, 9)];
		    T14 = TZ + T10;
		    TY = TW - TX;
		    T17 = T13 + T14;
		    T11 = TZ - T10;
	       }
	       T12 = FMA(KP475528258, TY, KP293892626 * T11);
	       T2u = T16 + T17;
	       T1O = FNMS(KP293892626, TY, KP475528258 * T11);
	       T15 = KP559016994 * (T13 - T14);
	       T18 = FNMS(KP250000000, T17, T16);
	       T19 = T15 + T18;
	       T1P = T18 - T15;
	  }
	  {
	       E Tm, Tj, Tk, Te, Tn, Th, Tl, To;
	       Tm = R1[0];
	       {
		    E Tc, Td, Tf, Tg;
		    Tc = R0[WS(rs, 3)];
		    Td = R1[WS(rs, 10)];
		    Tj = Tc + Td;
		    Tf = R1[WS(rs, 5)];
		    Tg = R0[WS(rs, 8)];
		    Tk = Tf + Tg;
		    Te = Tc - Td;
		    Tn = Tj + Tk;
		    Th = Tf - Tg;
	       }
	       Ti = FMA(KP475528258, Te, KP293892626 * Th);
	       T2r = Tm + Tn;
	       T1K = FNMS(KP293892626, Te, KP475528258 * Th);
	       Tl = KP559016994 * (Tj - Tk);
	       To = FNMS(KP250000000, Tn, Tm);
	       Tp = Tl + To;
	       T1L = To - Tl;
	  }
	  {
	       E TB, Ty, Tz, Tt, TC, Tw, TA, TD;
	       TB = R0[WS(rs, 2)];
	       {
		    E Tr, Ts, Tu, Tv;
		    Tr = R1[WS(rs, 4)];
		    Ts = R0[WS(rs, 12)];
		    Ty = Tr + Ts;
		    Tu = R0[WS(rs, 7)];
		    Tv = R1[WS(rs, 9)];
		    Tz = Tu + Tv;
		    Tt = Tr - Ts;
		    TC = Ty + Tz;
		    Tw = Tu - Tv;
	       }
	       Tx = FMA(KP475528258, Tt, KP293892626 * Tw);
	       T2q = TB + TC;
	       T1H = FNMS(KP293892626, Tt, KP475528258 * Tw);
	       TA = KP559016994 * (Ty - Tz);
	       TD = FNMS(KP250000000, TC, TB);
	       TE = TA + TD;
	       T1I = TD - TA;
	  }
	  {
	       E TR, TO, TP, TJ, TS, TM, TQ, TT;
	       TR = R0[WS(rs, 1)];
	       {
		    E TH, TI, TK, TL;
		    TH = R1[WS(rs, 3)];
		    TI = R0[WS(rs, 11)];
		    TO = TH + TI;
		    TK = R0[WS(rs, 6)];
		    TL = R1[WS(rs, 8)];
		    TP = TK + TL;
		    TJ = TH - TI;
		    TS = TO + TP;
		    TM = TK - TL;
	       }
	       TN = FMA(KP475528258, TJ, KP293892626 * TM);
	       T2t = TR + TS;
	       T1R = FNMS(KP293892626, TJ, KP475528258 * TM);
	       TQ = KP559016994 * (TO - TP);
	       TT = FNMS(KP250000000, TS, TR);
	       TU = TQ + TT;
	       T1S = TT - TQ;
	  }
	  T2s = T2q - T2r;
	  T2v = T2t - T2u;
	  Ci[WS(csi, 5)] = FNMS(KP587785252, T2v, KP951056516 * T2s);
	  Ci[WS(csi, 10)] = FMA(KP587785252, T2s, KP951056516 * T2v);
	  {
	       E T2z, T2y, T2A, T2w, T2x, T2B;
	       T2z = T8 + T9;
	       T2w = T2r + T2q;
	       T2x = T2t + T2u;
	       T2y = KP559016994 * (T2w - T2x);
	       T2A = T2w + T2x;
	       Cr[0] = T2z + T2A;
	       T2B = FNMS(KP250000000, T2A, T2z);
	       Cr[WS(csr, 5)] = T2y + T2B;
	       Cr[WS(csr, 10)] = T2B - T2y;
	  }
	  {
	       E Tb, Tq, TF, TG, T1E, T1F, T1G, T1B, T1C, T1D, TV, T1a, T1b, T1o, T1r;
	       E T1s, T1z, T1x, T1e, T1h, T1i, T1u, T1t;
	       Tb = T7 + Ta;
	       Tq = FMA(KP1_688655851, Ti, KP535826794 * Tp);
	       TF = FMA(KP1_541026485, Tx, KP637423989 * TE);
	       TG = Tq - TF;
	       T1E = FMA(KP851558583, TN, KP904827052 * TU);
	       T1F = FMA(KP1_984229402, T12, KP125333233 * T19);
	       T1G = T1E + T1F;
	       T1B = FNMS(KP844327925, Tp, KP1_071653589 * Ti);
	       T1C = FNMS(KP1_274847979, Tx, KP770513242 * TE);
	       T1D = T1B + T1C;
	       TV = FNMS(KP425779291, TU, KP1_809654104 * TN);
	       T1a = FNMS(KP992114701, T19, KP250666467 * T12);
	       T1b = TV + T1a;
	       {
		    E T1m, T1n, T1p, T1q;
		    T1m = FMA(KP1_937166322, Ti, KP248689887 * Tp);
		    T1n = FMA(KP1_071653589, Tx, KP844327925 * TE);
		    T1o = T1m + T1n;
		    T1p = FMA(KP1_752613360, TN, KP481753674 * TU);
		    T1q = FMA(KP1_457937254, T12, KP684547105 * T19);
		    T1r = T1p + T1q;
		    T1s = T1o + T1r;
		    T1z = T1q - T1p;
		    T1x = T1n - T1m;
	       }
	       {
		    E T1c, T1d, T1f, T1g;
		    T1c = FNMS(KP497379774, Ti, KP968583161 * Tp);
		    T1d = FNMS(KP1_688655851, Tx, KP535826794 * TE);
		    T1e = T1c + T1d;
		    T1f = FNMS(KP963507348, TN, KP876306680 * TU);
		    T1g = FNMS(KP1_369094211, T12, KP728968627 * T19);
		    T1h = T1f + T1g;
		    T1i = T1e + T1h;
		    T1u = T1f - T1g;
		    T1t = T1d - T1c;
	       }
	       Cr[WS(csr, 1)] = Tb + T1i;
	       Ci[WS(csi, 1)] = -(T1l + T1s);
	       Cr[WS(csr, 4)] = Tb + TG + T1b;
	       Ci[WS(csi, 4)] = T1l + T1D - T1G;
	       Ci[WS(csi, 9)] = FMA(KP309016994, T1D, T1l) + FMA(KP587785252, T1a - TV, KP809016994 * T1G) - (KP951056516 * (Tq + TF));
	       Cr[WS(csr, 9)] = FMA(KP309016994, TG, Tb) + FMA(KP951056516, T1B - T1C, KP587785252 * (T1F - T1E)) - (KP809016994 * T1b);
	       {
		    E T1v, T1w, T1y, T1A;
		    T1v = FMS(KP250000000, T1s, T1l);
		    T1w = KP559016994 * (T1r - T1o);
		    Ci[WS(csi, 11)] = FMA(KP587785252, T1t, KP951056516 * T1u) + T1v - T1w;
		    Ci[WS(csi, 6)] = FMA(KP951056516, T1t, T1v) + FNMS(KP587785252, T1u, T1w);
		    T1y = FNMS(KP250000000, T1i, Tb);
		    T1A = KP559016994 * (T1e - T1h);
		    Cr[WS(csr, 11)] = FMA(KP587785252, T1x, T1y) + FNMA(KP951056516, T1z, T1A);
		    Cr[WS(csr, 6)] = FMA(KP951056516, T1x, T1A) + FMA(KP587785252, T1z, T1y);
	       }
	  }
	  {
	       E T1W, T1X, T1J, T1M, T1N, T21, T22, T23, T1Q, T1T, T1U, T1Y, T1Z, T20, T26;
	       E T29, T2a, T2k, T2j, T2l, T2m, T2d, T2o, T2i;
	       T1W = FNMS(KP587785252, T1j, T1V);
	       T1X = Ta - T7;
	       T1J = FNMS(KP125333233, T1I, KP1_984229402 * T1H);
	       T1M = FMA(KP1_457937254, T1K, KP684547105 * T1L);
	       T1N = T1J - T1M;
	       T21 = FNMS(KP1_996053456, T1R, KP062790519 * T1S);
	       T22 = FMA(KP1_541026485, T1O, KP637423989 * T1P);
	       T23 = T21 - T22;
	       T1Q = FNMS(KP770513242, T1P, KP1_274847979 * T1O);
	       T1T = FMA(KP125581039, T1R, KP998026728 * T1S);
	       T1U = T1Q - T1T;
	       T1Y = FNMS(KP1_369094211, T1K, KP728968627 * T1L);
	       T1Z = FMA(KP250666467, T1H, KP992114701 * T1I);
	       T20 = T1Y - T1Z;
	       {
		    E T24, T25, T27, T28;
		    T24 = FNMS(KP481753674, T1L, KP1_752613360 * T1K);
		    T25 = FMA(KP851558583, T1H, KP904827052 * T1I);
		    T26 = T24 - T25;
		    T27 = FNMS(KP844327925, T1S, KP1_071653589 * T1R);
		    T28 = FNMS(KP998026728, T1P, KP125581039 * T1O);
		    T29 = T27 + T28;
		    T2a = T26 + T29;
		    T2k = T27 - T28;
		    T2j = T24 + T25;
	       }
	       {
		    E T2b, T2c, T2g, T2h;
		    T2b = FNMS(KP425779291, T1I, KP1_809654104 * T1H);
		    T2c = FMA(KP963507348, T1K, KP876306680 * T1L);
		    T2l = T2c + T2b;
		    T2g = FMA(KP1_688655851, T1R, KP535826794 * T1S);
		    T2h = FMA(KP1_996053456, T1O, KP062790519 * T1P);
		    T2m = T2g + T2h;
		    T2d = T2b - T2c;
		    T2o = T2l + T2m;
		    T2i = T2g - T2h;
	       }
	       Ci[WS(csi, 2)] = T1W + T2a;
	       Cr[WS(csr, 2)] = T1X + T2o;
	       Ci[WS(csi, 3)] = T1N + T1U - T1W;
	       Cr[WS(csr, 3)] = T1X + T20 + T23;
	       Cr[WS(csr, 8)] = FMA(KP309016994, T20, T1X) + FNMA(KP809016994, T23, KP587785252 * (T1T + T1Q)) - (KP951056516 * (T1M + T1J));
	       Ci[WS(csi, 8)] = FNMS(KP587785252, T21 + T22, KP309016994 * T1N) + FNMA(KP809016994, T1U, KP951056516 * (T1Y + T1Z)) - T1W;
	       {
		    E T2e, T2f, T2n, T2p;
		    T2e = KP559016994 * (T26 - T29);
		    T2f = FNMS(KP250000000, T2a, T1W);
		    Ci[WS(csi, 7)] = FMA(KP951056516, T2d, T2e) + FNMS(KP587785252, T2i, T2f);
		    Ci[WS(csi, 12)] = FMA(KP587785252, T2d, T2f) + FMS(KP951056516, T2i, T2e);
		    T2n = KP559016994 * (T2l - T2m);
		    T2p = FNMS(KP250000000, T2o, T1X);
		    Cr[WS(csr, 7)] = FMA(KP951056516, T2j, KP587785252 * T2k) + T2n + T2p;
		    Cr[WS(csr, 12)] = FMA(KP587785252, T2j, T2p) + FNMA(KP951056516, T2k, T2n);
	       }
	  }
     }
}

static const kr2c_desc desc = { 25, "r2cf_25", {117, 57, 83, 0}, &GENUS };

void X(codelet_r2cf_25) (planner *p) {
     X(kr2c_register) (p, r2cf_25, &desc);
}

#endif				/* HAVE_FMA */
