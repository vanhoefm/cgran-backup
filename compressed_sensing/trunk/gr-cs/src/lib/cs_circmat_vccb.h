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

#ifndef INCLUDED_CS_CIRCMAT_H
#define INCLUDED_CS_CIRCMAT_H

#include <gr_sync_block.h>
#include <vector>

class cs_circmat_vccb;
typedef boost::shared_ptr<cs_circmat_vccb> cs_circmat_vccb_sptr;

cs_circmat_vccb_sptr
cs_make_circmat_vccb (const std::vector<char> &sequence, unsigned output_length, bool translate_zeros = false);

/*!
 * \brief Compress with circular shifted sequence
 *
 *
 * \ingroup compressedsensing
 */
class cs_circmat_vccb : public gr_sync_block
{
	friend cs_circmat_vccb_sptr cs_make_circmat_vccb (const std::vector<char> &sequence,
		        	unsigned output_length, bool translate_zeros);

 private:
        unsigned d_input_length, d_output_length;
	bool d_translate_zeros;
	std::vector<char> d_sequence;

	cs_circmat_vccb(const std::vector<char> &sequence, unsigned output_length, bool translate_zeros);

	void translate_zeros_to_negative();

 public:
	~cs_circmat_vccb();

	int work(int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);


	float get_compression() { return ((float) d_output_length) / d_input_length; };
};
#endif /* INCLUDED_CS_CIRCMAT_H */

