/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2004 Free Software Foundation, Inc.
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
// This is a modification of trellis_metrics_f.h from GNU Radio.


#ifndef INCLUDED_FOIMIMO_TRELLIS_METRICS_F_H
#define INCLUDED_FOIMIMO_TRELLIS_METRICS_F_H

#include <gr_block.h>

class foimimo_trellis_metrics_f;
typedef boost::shared_ptr<foimimo_trellis_metrics_f> foimimo_trellis_metrics_f_sptr;

foimimo_trellis_metrics_f_sptr foimimo_make_trellis_metrics_f (int O, int D, const std::vector<float> &REtable,
    const std::vector<float> &IMtable);

/*!
 * \brief Evaluate metrics for use by the Viterbi algorithm.
 *
 * \param O size of coded alphabet
 * \param D dimension of the modulation
 * \param REtable the real coordinates for the modulation
 * \param IMtable the imaginary coordinate for the modulation
 *
 * \comment This block assumes that the modulation type is QPSK. To enable
 *          different modulation types the calc_metric function have to be
 *          modified. This block does support various coding rates.
 */
class foimimo_trellis_metrics_f : public gr_block
{
  int d_O;
  int d_D;
  std::vector<float> d_re;
  std::vector<float> d_im;

  friend foimimo_trellis_metrics_f_sptr foimimo_make_trellis_metrics_f (int O, int D,
      const std::vector<float> &REtable,const std::vector<float> &IMtable);

  foimimo_trellis_metrics_f (int O, int D, const std::vector<float> &REtable,
      const std::vector<float> &IMtable);

  void calc_metric(const float *in,float *metric);

  bool d_real_part;
public:
  int O () const { return d_O; }
  int D () const { return d_D; }

  void forecast (int noutput_items,
		 gr_vector_int &ninput_items_required);
  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};


#endif
