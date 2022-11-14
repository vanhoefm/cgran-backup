/* -*- c++ -*- */
/* 
 * Copyright 2010 Communications Engineering Lab, KIT
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */


#include <specesti_esprit_armadillo.h>
#include <specesti_correst.h>

specesti_esprit_armadillo::specesti_esprit_armadillo(unsigned int n, unsigned int m) : d_n(n), d_m(m), d_R(m,m)
{
	if (n > m)
		throw std::invalid_argument("specesti_esprit_armadillo: n cannot exceed m in length.");
}


specesti_esprit_armadillo::~specesti_esprit_armadillo()
{
}

void
specesti_esprit_armadillo::calculate(const gr_complexd* data, unsigned int data_len,
                                     double* omegas)
{
	specesti::correst(data, data_len, d_m, &d_R);

	arma::colvec eigvals;
	arma::cx_mat eigvec;
	arma::eig_sym(eigvals, eigvec, d_R);

	arma::cx_mat S = eigvec.cols(d_m-d_n, d_m-1);
	arma::cx_mat S1 = S.rows(0, d_m - 1 - 1);
	arma::cx_mat S2 = S.rows(1, d_m-1);

	arma::cx_mat Phi_hat = arma::pinv(S2) * S1;
	arma::cx_colvec tmp_omegas_hat(d_m);
	arma::cx_mat tmp_eigvecs;
	arma::eig_gen(tmp_omegas_hat, tmp_eigvecs, Phi_hat);
	for(unsigned int i = 0; i < d_n; i++)
		omegas[i] = arg(tmp_omegas_hat[i]);
}

void
specesti_esprit_armadillo::calculate_pseudospectrum(const gr_complexd* data,
                                                    unsigned int data_len,
                                                    double* pspectrum,
                                                    unsigned int pspectrum_len)
{
	specesti::correst(data, data_len, d_m, &d_R);
	arma::colvec eigvals;
	arma::cx_mat eigvec;
	arma::eig_sym(eigvals, eigvec, d_R);

	arma::cx_mat S = eigvec.cols(d_m-d_n, d_m-1);
	arma::cx_mat S1 = S.rows(0, d_m - 1 - 1);
	arma::cx_mat S2 = S.rows(1, d_m-1);

	arma::cx_mat Phi_hat = arma::pinv(S1) * S2;
	arma::cx_colvec tmp_omegas_hat(d_m);
	arma::cx_mat tmp_eigvecs;
	arma::eig_gen(tmp_omegas_hat, tmp_eigvecs, Phi_hat);

	gr_complexd pspectrum_tmp;
	for(int i = 0; i < pspectrum_len; i++)
	{
		pspectrum_tmp = 1.0;
		double omega = -M_PI + i * 2.0*M_PI/(pspectrum_len -1);
		for(int k = 0; k < d_n; k++)
		{
			pspectrum_tmp /= exp(gr_complexd(0,-omega)) - tmp_omegas_hat[k];
		}
		pspectrum[i] = pow(abs(pspectrum_tmp), 2.0);
	}
}

