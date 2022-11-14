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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dabp_rscodec.h"
#include <cassert>
#include <cstring>

dabp_rscodec::dabp_rscodec(int m, int t, int start_j):d_m(m),d_gf(m),d_t(t),d_g(0)
{
	d_n=(1<<m)-1; // non-shortened. n=2^m-1
	d_k=d_n-2*t;
	assert(start_j==0||start_j==1);
	d_j=start_j;
	init_g(start_j); // construct the generator polynomial
	d_reg=new int[2*t];
	d_synd=new int[2*t];
	d_euc[0]=new int[2*t+2];
	d_euc[1]=new int[2*t+2];
}

void dabp_rscodec::set_t(int t)
{
	d_t=t;
	d_k=d_n-2*t;
	init_g(d_j);
    delete [] d_reg;
    delete [] d_synd;
    delete [] d_euc[0];
    delete [] d_euc[1];
	d_reg=new int[2*t];
	d_synd=new int[2*t];
	d_euc[0]=new int[2*t+2];
	d_euc[1]=new int[2*t+2];
}

dabp_rscodec::~dabp_rscodec()
{
    delete [] d_g;
    delete [] d_reg;
    delete [] d_synd;
    delete [] d_euc[0];
    delete [] d_euc[1];
}

void dabp_rscodec::init_g(int start_j)
{
	int i;
	int tmp_g, tmp_g2;
	if(d_t<=0) return; // no error correcting
	if(d_g)
        delete [] d_g;
    d_g=new int[2*d_t]; // g has degree 2t, g(0),g(1),...,g(2t-1) are its coefficients, g(2t)=1 is not included
	d_g[2*d_t-1]=start_j+1; // power representation
	for(int k=1;k<2*d_t;k++){
		tmp_g=d_g[2*d_t-1]; // preserve for later use
		d_g[2*d_t-1]=d_gf.add_power(d_g[2*d_t-1],start_j+k+1);
		for(i=2*d_t-2;i>=2*d_t-k;i--){
			tmp_g2=d_g[i];
			d_g[i]=d_gf.add_power(d_gf.multiply_power(tmp_g,start_j+k+1),d_g[i]);
			tmp_g=tmp_g2;
		}
		d_g[2*d_t-k-1]=d_gf.multiply_power(tmp_g,start_j+k+1);
	}
}

void dabp_rscodec::enc_power(const int * u, int * c)
{
	//d_reg.clear();
    memset(d_reg,0,2*d_t*sizeof(d_reg[0]));
	int i,j;
	int fb;
	for(i=0;i<d_k;i++){ // shift in u
		c[i]=u[i]; // k systematic symbols first
		if(d_t>0){
			fb=d_gf.add_power(u[i],d_reg[2*d_t-1]);
			for(j=2*d_t-1;j>0;j--){
				d_reg[j]=d_gf.add_power(d_reg[j-1],d_gf.multiply_power(fb,d_g[j]));
			}
			d_reg[0]=d_gf.multiply_power(fb,d_g[0]);
		}
	}
	// n-k parity symbols follow systematic symbols
	for(i=d_k;i<d_n;i++){
		c[i]=d_reg[2*d_t-1-i+d_k];
	}
}

void dabp_rscodec::enc_poly(const int * u, int * c)
{
	//d_reg.clear();
    memset(d_reg,0,2*d_t*sizeof(d_reg[0]));
	int i,j;
	int fb;
	for(i=0;i<d_k;i++){ // shift in u
		c[i]=u[i]; // k systematic symbols
		if(d_t>0){
			fb=d_gf.add_poly(u[i],d_reg[2*d_t-1]);
			for(j=2*d_t-1;j>0;j--){
				d_reg[j]=d_gf.add_poly(d_reg[j-1],d_gf.multiply_poly(fb,d_gf.power2poly(d_g[j])));
			}
			d_reg[0]=d_gf.multiply_poly(fb,d_gf.power2poly(d_g[0]));
		}
	}
	// n-k parity symbols
	for(i=d_k;i<d_n;i++){
		c[i]=d_reg[2*d_t-1-i+d_k];
	}
}

void dabp_rscodec::enc(const unsigned char * u, unsigned char * c, int cutlen)
{
    assert(cutlen>=0);
    int uf[d_k]; // full length info bytes
    int cf[d_n]; // full length code bytes
    int i;
    for(i=0;i<cutlen;i++)
        uf[i]=0;
    for(;i<d_k;i++)
        uf[i]=(int)u[i-cutlen];
    enc_poly(uf,cf);
    for(i=cutlen;i<d_n;i++)
        c[i-cutlen]=(unsigned char)cf[i];
}

int dabp_rscodec::dec_power(const int * r, int * d)
{
	// calculate the syndrome, d_synd[0] is the highest degree coefficient, d_synd[2t-1] is the lowest degree coefficient
	// i.e. S(X)=d_synd[0]X^(2t-1)+d_synd[1]X^(2t-2)+...+d_synd[2t-1]
	int i,j;
	if(d_t<=0){ // no error correcting
		memcpy(d,r,d_k*sizeof(r[0]));
		return 0;
	}
	
	for(i=0;i<2*d_t;i++){
		d_synd[2*d_t-1-i]=r[0]; // syndrome also in power representation
		for(j=1;j<d_n;j++){
			d_synd[2*d_t-1-i]=d_gf.add_power(r[j],d_gf.multiply_power(d_synd[2*d_t-1-i],d_j+i+1));
		}
	}

	// Euclidean algorithm
	// step 1: initialize
	int top=0; // index to 'top', index to 'bottom' is !top
	int deg[2]={2*d_t,2*d_t-1}; // top and bottom relaxed degree
	//d_euc[top].clear();
	//d_euc[!top].clear();
    memset(d_euc[top],0,(2*d_t+2)*sizeof(d_euc[0][0]));
    memset(d_euc[!top],0,(2*d_t+2)*sizeof(d_euc[0][0]));
	d_euc[top][0]=1;
	d_euc[top][2*d_t+1]=1;
	//d_euc[!top].set_subvector(0,d_synd);
    memcpy(d_euc[!top],d_synd,2*d_t*sizeof(d_synd[0]));
    
	// step 2: repeat 2t times
	int mu[2];
	for(i=0;i<2*d_t;i++){
		// step 2.a
		mu[top]=d_euc[top][0];
		mu[!top]=d_euc[!top][0];
		// step 2.b
		if(mu[!top]!=0 && deg[!top]<deg[top]){
			top=!top; // swap 'top' and 'bottom'
		}
		// step 2.c
		if(mu[!top]!=0){
			for(j=1;j<=deg[top];j++){
				d_euc[!top][j]=d_gf.add_power(d_gf.multiply_power(mu[top],d_euc[!top][j]),d_gf.multiply_power(mu[!top],d_euc[top][j]));
			}
			for(;j<=deg[!top];j++){
				d_euc[!top][j]=d_gf.multiply_power(mu[top],d_euc[!top][j]);
			}

			for(j=2*d_t+1;j>deg[!top];j--){
				d_euc[top][j]=d_gf.add_power(d_gf.multiply_power(mu[top],d_euc[top][j]),d_gf.multiply_power(mu[!top],d_euc[!top][j]));
			}
			for(;j>deg[top];j--){
				d_euc[top][j]=d_gf.multiply_power(mu[top],d_euc[top][j]);
			}
		}
		// step 2.d
		//d_euc[!top].shift_left(0);
        memmove(d_euc[!top],d_euc[!top]+1,(2*d_t+1)*sizeof(d_euc[0][0]));
        d_euc[!top][2*d_t+1]=0;
		deg[!top]--;
	}
	// step 3: output, evaluator=d_euc[!top][0..deg[!top]], locator=d_euc[top][2t+1..deg[top]+1], highest degree coefficient first
	// if deg[top]>deg[!top], then error correctable; otherwise uncorrectable error pattern
	if(deg[top]<=deg[!top]){
		//d=r.left(d_k); // no correction attempt if uncorrectable error pattern
        memcpy(d,r,d_k*sizeof(r[0]));
		return -1;
	}
	if(deg[top]==2*d_t){ // no error
		//d=r.left(d_k);
        memcpy(d,r,d_k*sizeof(r[0]));
		return 0;
	}

	/*int check=0;
	for(i=deg[top]+2;i<=2*d_t+1;i+=2){
		check+=d_euc[top][i];
	}
	if(check==0){ //sigma_odd = 0, error not correctable!
		d=r.left(d_k);
		return -1;
	}
	*/
	//int len_loc=2*d_t+1-deg[top]; // length of locator polynomial
	//int len_eval=deg[!top]+1; // length of evaluator polynomial

	// Chien's Search
	int x,x2,y;
	int sig_high,sig_low; // sigma_high, sigma_low (temporary)
	int sig_even,sig_odd; // sigma_even, sigma_odd: error locator value
	int omega; // error evaluator value
	int err_cnt=0; // error symbol counter

	for(i=d_n-1;i>=2*d_t;i--){ // for each information symbol position i, i.e. alpha^(-i). i represents the exponent of the received polynomial
		x=d_gf.inverse_power(i+1); // alpha^(-i). can be optimized
		x2=d_gf.multiply_power(x,x); // x^2
		// calculate locator_even(x^2)
		// calculate locator_odd(x^2)
		sig_high=d_euc[top][2*d_t+1];
		sig_low=d_euc[top][2*d_t];
		for(j=2*d_t-1;j>deg[top];j-=2){
			sig_high=d_gf.add_power(d_gf.multiply_power(sig_high,x2),d_euc[top][j]);
		}
		for(j=2*d_t-2;j>deg[top];j-=2){
			sig_low=d_gf.add_power(d_gf.multiply_power(sig_low,x2),d_euc[top][j]);
		}
		if(j==deg[top]){ // the last j is deg[top]+2, then sig_low is sig_odd
			sig_odd=sig_low;
			sig_even=sig_high;
		}else{
			sig_odd=sig_high;
			sig_even=sig_low;
		}

		// calculate locator and judge if it is an error location
		if(d_gf.add_power(sig_even,d_gf.multiply_power(x,sig_odd))==0){ // located an error
			if(sig_odd==0){ // non-correctable error
				//d=r.left(d_k);
                memcpy(d,r,d_k*sizeof(r[0]));
				return -1;
			}
			// calculate error evaluator
			omega=d_euc[!top][0];
			for(j=1;j<=deg[!top];j++){
				omega=d_gf.add_power(d_gf.multiply_power(omega,x),d_euc[!top][j]);
			}
			// error value. Forney
			if(d_j>=1){
				y=d_gf.multiply_power(d_gf.divide_power(omega,sig_odd),d_gf.pow_power(x,d_j-1));
			}else{
				y=d_gf.divide_power(d_gf.divide_power(omega,sig_odd),d_gf.pow_power(x,1-d_j));
			}
			// error correction
			d[d_n-1-i]=d_gf.add_power(r[d_n-1-i],y);
			err_cnt++; // count the errors
		}else{
			d[d_n-1-i]=r[d_n-1-i]; // not an error symbol
		}

	}

	return err_cnt;
}

int dabp_rscodec::dec_poly(const int * r, int * d)
{
	// calculate the syndrome, d_synd[0] is the highest degree coefficient, d_synd[2t-1] is the lowest degree coefficient
	// i.e. S(X)=d_synd[0]X^(2t-1)+d_synd[1]X^(2t-2)+...+d_synd[2t-1]
	int i,j;
	if(d_t<=0){ // no error correcting
		memcpy(d,r,d_k*sizeof(r[0]));
		return 0;
	}
	
	for(i=0;i<2*d_t;i++){
		d_synd[2*d_t-1-i]=r[0]; // syndrome also in polynomial representation
		for(j=1;j<d_n;j++){
			d_synd[2*d_t-1-i]=d_gf.add_poly(r[j],d_gf.multiply_poly(d_synd[2*d_t-1-i],d_gf.power2poly(d_j+i+1)));
		}
	}

	// Euclidean algorithm
	// step 1: initialize
	int top=0; // index to 'top', index to 'bottom' is !top
	int deg[2]={2*d_t,2*d_t-1}; // top and bottom relaxed degree
	//d_euc[top].clear();
	//d_euc[!top].clear();
    memset(d_euc[top],0,(2*d_t+2)*sizeof(d_euc[0][0]));
    memset(d_euc[!top],0,(2*d_t+2)*sizeof(d_euc[0][0]));
	d_euc[top][0]=1;
	d_euc[top][2*d_t+1]=1;
	//d_euc[!top].set_subvector(0,d_synd);
    memcpy(d_euc[!top],d_synd,2*d_t*sizeof(d_synd[0]));
    
	// step 2: repeat 2t times
	int mu[2];
	for(i=0;i<2*d_t;i++){
		// step 2.a
		mu[top]=d_euc[top][0];
		mu[!top]=d_euc[!top][0];
		// step 2.b
		if(mu[!top]!=0 && deg[!top]<deg[top]){
			top=!top; // swap 'top' and 'bottom'
		}
		// step 2.c
		if(mu[!top]!=0){
			for(j=1;j<=deg[top];j++){
				d_euc[!top][j]=d_gf.add_poly(d_gf.multiply_poly(mu[top],d_euc[!top][j]),d_gf.multiply_poly(mu[!top],d_euc[top][j]));
			}
			for(;j<=deg[!top];j++){
				d_euc[!top][j]=d_gf.multiply_poly(mu[top],d_euc[!top][j]);
			}

			for(j=2*d_t+1;j>deg[!top];j--){
				d_euc[top][j]=d_gf.add_poly(d_gf.multiply_poly(mu[top],d_euc[top][j]),d_gf.multiply_poly(mu[!top],d_euc[!top][j]));
			}
			for(;j>deg[top];j--){
				d_euc[top][j]=d_gf.multiply_poly(mu[top],d_euc[top][j]);
			}
		}
		// step 2.d
		//d_euc[!top].shift_left(0);
        memmove(d_euc[!top],d_euc[!top]+1,(2*d_t+1)*sizeof(d_euc[0][0]));
        d_euc[!top][2*d_t+1]=0;
		deg[!top]--;
	}
	// step 3: output, evaluator=d_euc[!top][0..deg[!top]], locator=d_euc[top][2t+1..deg[top]+1], highest degree coefficient first
	// if deg[top]>deg[!top], then error correctable; otherwise uncorrectable error pattern
	if(deg[top]<=deg[!top]){
		//d=r.left(d_k); // no correction attempt if uncorrectable error pattern
        memcpy(d,r,d_k*sizeof(r[0]));
		return -1;
	}
	if(deg[top]==2*d_t){ // no error
		//d=r.left(d_k);
        memcpy(d,r,d_k*sizeof(r[0]));
		return 0;
	}

	/*int check=0;
	for(i=deg[top]+2;i<=2*d_t+1;i+=2){
		check+=d_euc[top][i];
	}
	if(check==0){ //sigma_odd = 0, error not correctable!
		d=r.left(d_k);
		return -1;
	}
	*/
	//int len_loc=2*d_t+1-deg[top]; // length of locator polynomial
	//int len_eval=deg[!top]+1; // length of evaluator polynomial

	// Chien's Search
	int x,x2,y;
	int sig_high,sig_low; // sigma_high, sigma_low (temporary)
	int sig_even,sig_odd; // sigma_even, sigma_odd: error locator value
	int omega; // error evaluator value
	int err_cnt=0; // error symbol counter

	for(i=d_n-1;i>=2*d_t;i--){ // for each information symbol position i, i.e. alpha^(-i). i represents the exponent of the received polynomial
		x=d_gf.inverse_power(i+1); // alpha^(-i). can be optimized
		x2=d_gf.power2poly(d_gf.multiply_power(x,x)); // x^2
		x=d_gf.power2poly(x);
		// calculate locator_even(x^2)
		// calculate locator_odd(x^2)
		sig_high=d_euc[top][2*d_t+1];
		sig_low=d_euc[top][2*d_t];
		for(j=2*d_t-1;j>deg[top];j-=2){
			sig_high=d_gf.add_poly(d_gf.multiply_poly(sig_high,x2),d_euc[top][j]);
		}
		for(j=2*d_t-2;j>deg[top];j-=2){
			sig_low=d_gf.add_poly(d_gf.multiply_poly(sig_low,x2),d_euc[top][j]);
		}
		if(j==deg[top]){ // the last j is deg[top]+2, then sig_low is sig_odd
			sig_odd=sig_low;
			sig_even=sig_high;
		}else{
			sig_odd=sig_high;
			sig_even=sig_low;
		}
		// calculate locator and judge if it is an error location
		if(d_gf.add_poly(sig_even,d_gf.multiply_poly(x,sig_odd))==0){ // located an error
			if(sig_odd==0){ // non-correctable error
				//d=r.left(d_k);
                memcpy(d,r,d_k*sizeof(r[0]));
				return -1;
			}
			// calculate error evaluator
			omega=d_euc[!top][0];
			for(j=1;j<=deg[!top];j++){
				omega=d_gf.add_poly(d_gf.multiply_poly(omega,x),d_euc[!top][j]);
			}
			// error value. Forney
			if(d_j>=1){
				y=d_gf.multiply_poly(d_gf.divide_poly(omega,sig_odd),d_gf.pow_poly(x,d_j-1));
			}else{
				y=d_gf.divide_poly(d_gf.divide_poly(omega,sig_odd),d_gf.pow_poly(x,1-d_j));
			}
			// error correction
			d[d_n-1-i]=d_gf.add_poly(r[d_n-1-i],y);
			err_cnt++; // count the errors
		}else{
			d[d_n-1-i]=r[d_n-1-i]; // not an error symbol
		}

	}

	return err_cnt;
}

int dabp_rscodec::dec(const unsigned char * r, unsigned char * d, int cutlen)
{
    int rf[d_n];
    int df[d_k];
    int i;
    for(i=0;i<cutlen;i++)
        rf[i]=0;
    for(;i<d_n;i++)
        rf[i]=(int)r[i-cutlen];
    int ret=dec_poly(rf,df);
    for(i=cutlen;i<d_k;i++)
        d[i-cutlen]=(unsigned char)df[i];
    return ret;
}

