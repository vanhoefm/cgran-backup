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

#ifndef INCLUDED_CS_NUSAIC_H
#define INCLUDED_CS_NUSAIC_H

#include <gr_block.h>

class cs_nusaic_cc;
typedef boost::shared_ptr<cs_nusaic_cc> cs_nusaic_cc_sptr;

cs_nusaic_cc_sptr
cs_make_nusaic_cc (unsigned compression);

/*!
 * \brief Keep one of every N samples as indicated.
 *
 * For a given compression factor N, which must be an integer value, one sample in every consecutive
 * N samples is chosen, the rest is discarded. The first input stream are the complex samples.
 * The second stream is an integer stream in [0;N-1], indicating the offset from every Nth sample to
 * be used. Passing the same value on the second stream all the time is equivalent to regular decimation.
 *
 * \ingroup compressedsensing
 */

class cs_nusaic_cc : public gr_block
{
	friend cs_nusaic_cc_sptr cs_make_nusaic_cc (unsigned compression);

 private:
	unsigned d_compression;

	cs_nusaic_cc(unsigned compression);

 public:
	~cs_nusaic_cc();

	int general_work(int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);

	void forecast(int noutput_items, gr_vector_int &ninput_items_required);

	float get_compression() { return 1.0 / d_compression; };
};
#endif /* INCLUDED_CS_NUSAIC_H */

