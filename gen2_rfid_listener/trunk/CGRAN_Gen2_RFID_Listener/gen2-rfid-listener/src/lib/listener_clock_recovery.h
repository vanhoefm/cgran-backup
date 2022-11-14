/* -*- c++ -*- */
/*
 * Copyright 2004,2006 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
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


// Acknowledgement: I would like to acknowledge Michael Buettner since this adapted and modified block is inherited from his CGRAN project "Gen2 RFID Reader"

#ifndef INCLUDED_listener_clock_recovery_H
#define INCLUDED_listener_clock_recovery_H

#include <gr_block.h>

class gri_mmse_fir_interpolator;

class listener_clock_recovery;
typedef boost::shared_ptr<listener_clock_recovery> listener_clock_recovery_sptr;

listener_clock_recovery_sptr
listener_make_clock_recovery(int samples_per_pulse, int interp_factor);

class listener_clock_recovery : public gr_block
{  

  friend listener_clock_recovery_sptr
  listener_make_clock_recovery(int samples_per_pulse, int interp_factor);

  public:
  ~listener_clock_recovery();
  int general_work(int noutput_items,
		   gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);
protected:

  listener_clock_recovery(int samples_per_pulse, int interp_factor);
  

private:
  int d_samples_per_pulse;
  int d_interp_factor;
  gri_mmse_fir_interpolator 	*d_interp;
  float * d_interp_buffer;
  int d_last_zc_count;
  float d_pwr;
  int d_avg_window_size;
  bool d_last_was_pos;

  float * d_avg_vec;
  int d_avg_vec_index;
  

  void forecast (int noutput_items, gr_vector_int &ninput_items_required);
  

};

#endif 
