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
#ifndef INCLUDED_DABP_RSCODEC_H
#define INCLUDED_DABP_RSCODEC_H

#include "dabp_galois.h"

/*! \brief Reed-Solomon CODEC (Encoder/Decoder)
The generator polynomial of RS code is
g(X)=(X+alpha^j)(X+alpha^(j+1))...(X+alpha^(j+2t-1))
    =g(0)+g(1)X+g(2)X^2+...+g(2t-1)X^(2t-1)+X^(2t)
Note that the starting exponent j does not affect the code's property
Usually, we set j=1.
However, it is also common to use j=0 (such as in ATSC, WiMAX etc.)
Thus, we allow a parameter in the constructor to accomodate different applications

The encoder structure is shown in
Fig. 6.13, page 172, Error Control Coding: Fundamentals and Applications by Shu Lin and Daniel J. Costello, Jr.
Note the order of information and codeword symbols:
information symbol: u(X)=u(k-1)X^(k-1)+...+u(1)X+u(0)
codeword symbol: c(X)=c(n-1)X^(n-1)+...+c(1)X+c(0)
generator: g(X)=X^(2t)+g(2t-1)X^(2t-1)+...+g(1)X+g(0)
remainder: b(X)=b(2t-1)X^(2t-1)+...+b(1)X+b(0)=Remainder[X^(2t)u(X)/g(X)]

information symbol u(k-1) is the first symbol entering the encoder
codeword symbol c(n-1) is the first symbol leaving the encoder
[c(n-1),...,c(n-k)]=[u(k-1),...,u(0)]
[c(n-k-1),...,c(0)]=[b(2t-1),...,b(0)]
So in the codeword, the parity symbols follow the information symbols

All vectors are assumed to have the highest-order coefficient as the first element.
e.g. information symbol polynomial u(X) is represented as a vector u=[u(k-1),...,u(0)], received symbol polynomial r(X) is represented as a vector r=[r(n-1),...,r(0)]

notations for parameters: u (info) -> c (codeword) -> r (received) -> d (decoded)

We use class Galois to perform additions and multiplications

Generally, information and codeword symbols are represented as polynomial form.
For convenience, we provide two types of encoder and decoder that accomodate both power and polynomial forms.

The decoding algorithm is Euclidean algorithm
c.f. p219, Reed-Solomon Codes and Their Applications, edited by Stephen B. Wicker and Vijay K. Bhargava
*/
class dabp_rscodec{
public:
    dabp_rscodec(int m=8, int t=5, int start_j=0); // GF(2^m), error correction capability t, start_j is the starting exponent in the generator polynomial
	~dabp_rscodec();

	void set_t(int t); // set error correction capability t
	void enc_power(const int * u, int * c); // encode, both u and c are in power representation
	void enc_poly(const int * u, int * c); // u and c are in polynomial representation. This is more common in practice
    
    // encode in poly form specifically for m=8
	void enc(const unsigned char * u, unsigned char * c, int cutlen=135); // encode shortened code, cutlen bytes were shortened
    
    int dec_power(const int * r, int * d); // decode, both r and d are in power representation. return the number of errors (only count information symbols)
	int dec_poly(const int * r, int * d); // r and d are in polynomial representation. This is more common in practice
    
    // decode in poly form specifically for m=8
    int dec(const unsigned char * r, unsigned char * d, int cutlen=135); // decode shortened code
private:
	int d_m; // GF(2^m)
	dabp_galois d_gf; // Galois field
	int * d_g; // generator g. d_g[0] is the lowest exponent coefficient, d_g[2t-1] is the highest exponent coefficient, d_g[2t]=1 is not included
	int d_t, d_k, d_n; // error correcting capability t, info length k, codeword length n, all measured in unit of GF(2^m) symbols
	int d_j; // starting exponent for generator polynomial, j

	int * d_reg; // registers for encoding
	int * d_synd; // syndrome
	int * d_euc[2]; // data structure for Euclidean computation

	void init_g(int start_j); // initialize the generator polynomial g
};

#endif // INCLUDED_DABP_RSCODEC_H
