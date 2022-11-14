/* -*- c++ -*- */
/*
 * Copyright 2009 Institut fuer Nachrichtentechnik / Uni Karlsruhe
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

#ifndef INCLUDED_CS_TOEPLITZ_VCCB_H
#define INCLUDED_CS_TOEPLITZ_VCCB_H

#include <gr_sync_block.h>
#include <vector>

class cs_toeplitz_vccb;
typedef boost::shared_ptr<cs_toeplitz_vccb> cs_toeplitz_vccb_sptr;

cs_toeplitz_vccb_sptr
cs_make_toeplitz_vccb (unsigned input_length, unsigned output_length, const std::vector<char> &seq_history);

class cs_toeplitz_vccb : public gr_sync_block
{
	friend cs_toeplitz_vccb_sptr cs_make_toeplitz_vccb (unsigned input_length,
			unsigned output_length,
			const std::vector<char> &seq_history);

 private:
        unsigned d_input_length, d_output_length;
	std::vector<char> d_seq_buf;

	cs_toeplitz_vccb (unsigned input_length, unsigned output_length, const std::vector<char> &seq_history);

	inline char comp_matrix_element(char *sequence, unsigned row, unsigned col);

 public:
	~cs_toeplitz_vccb();

	int work(int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);

	/*!
	 * \brief Compression ratio, in (0,1]
	 */
	float get_compression() { return ((float) d_output_length) / d_input_length; };
};

#endif /* INCLUDED_CS_TOEPLITZ_VCCB_H */

