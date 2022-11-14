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
#ifndef INCLUDED_SPECEST_ESPRIT_ARMADILLO_H
#define INCLUDED_SPECEST_ESPRIT_ARMADILLO_H

#include <gr_complex.h>
#include <specesti_esprit.h>
#include <armadillo>

class specesti_esprit_armadillo : virtual public specesti_esprit
{
	public:
		/*!
		 * \brief Create an ESPRIT object
		 * \param n dimension of the signal subspace, i.e. how many sinusoids does your model have?
		 * \param m maximum timeshift for your correlation, you want to make 
		 *          this as big as you can afford.
		*/
		specesti_esprit_armadillo(unsigned int n, unsigned int m);
		void calculate(const gr_complexd *data, unsigned int data_len,
		               double* omegas);
		void calculate_pseudospectrum(const gr_complexd* data, unsigned int data_len,
		                              double* pspectrum, unsigned int pspectrum_len);
		~specesti_esprit_armadillo();
	private:
		unsigned int d_n;
		unsigned int d_m;
		arma::cx_mat d_R;
};

#endif /* INCLUDED_SPECEST_ESPRIT_ARMADILLO_H */
