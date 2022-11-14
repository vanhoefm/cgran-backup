/* -*- c++ -*- */
/*
 * Copyright 2004,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
#ifndef INCLUDED_DABP_GALOIS_H
#define INCLUDED_DABP_GALOIS_H

/*! \brief Galois Field
GF(q)=GF(2^m)
Assume 1<=m<=10. 2<=q<=1024
We use lookup tables (LUTs) to translate between power representation and polynomial representation

element				power repres.				poly. repres.
--------------------------------------------------------------------------
0						0							0
alpha^0					1							1
alpha^1					2							2
...
alpha^k					k+1						coefficients of poly.=>integer
...
alpha^(q-2)				q-1
--------------------------------------------------------------------------
alpha^(q-1)=1 [alpha has order q-1]

All functions do not do range check regarding the power and polynomial representations for speed purpose
The user has to make sure the parameters are in range, otherwise weird things may happen

predetermined primitive polynomials to generate GF(q) are as follows:
c.f. Table 2.7, page 29, Error Control Coding: Fundamentals and Applications by Shu Lin and Daniel J. Costello

m		p(octet)	p(X)
-------------------------------------------------
1		03		1+X
2		07		1+X+X^2
3		013		1+X+X^3
4		023		1+X+X^4
5		045		1+X^2+X^5
6		0103	1+X+X^6
7		0211	1+X^3+X^7
8		0435	1+X^2+X^3+X^4+X^8
9		01021	1+X^4+X^9
10		02011	1+X^3+X^10
-------------------------------------------------

*/
class dabp_galois{
public:
	dabp_galois(int m);
	~dabp_galois();

	int add_poly(int a, int b) {return a^b;} // c=a+b, all in polynomial representations
	int add_power(int a, int b); // all in power representations
	int multiply_poly(int a, int b); // a*b
	int multiply_power(int a, int b) {return (a==0 || b==0) ? 0 : (a+b-2)%(d_q-1)+1;}
	int divide_poly(int a, int b); // a/b
	int divide_power(int a, int b);
	int pow_poly(int a, int n); // a^n
	int pow_power(int a, int n) {return (a==0)? 0 : (a-1)*n%(d_q-1)+1;}
	int inverse_poly(int a) {return divide_poly(1,a);} // a^(-1)
	int inverse_power(int a) {return divide_power(1,a);}

	int power2poly(int a) {return lut_power2poly[a];}
	int poly2power(int a) {return lut_poly2power[a];}

	void poly2tuple(int a, unsigned char tuple[]); // convert a polynomial representation to m-tuple representation returned in tuple, lowest degree first. tuple must have size d_m at minimum
	void power2tuple(int a, unsigned char tuple[]) {poly2tuple(lut_power2poly[a],tuple);}

private:
	static const int MAX_M=10, MAX_Q=1<<MAX_M;
	static const int PRIMITIVES[MAX_M]; // the predetermined primitive polynomials

private:
	int d_m, d_q; // m, q=2^m
	int d_p; // primitive polynomial to generate GF(q)
	int lut_power2poly[MAX_Q]; // LUT translating power form to polynomial form
	int lut_poly2power[MAX_Q]; // LUT translating polynomial form to power form
	void init_lut(); // construct the LUTs
	int round_mod(int a, int n) {return (a%n<0)? (a%n+n) : (a%n);} // round mod algorithm, calculate a%n, n>0, a is any integer, a%n>=0
};

#endif // INCLUDED_DABP_GALOIS_H

