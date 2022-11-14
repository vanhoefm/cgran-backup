/* -*- c++ -*- */
/* 
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


#ifndef INCLUDED_LISTENER_TO_MAG_MUX_H
#define INCLUDED_LISTENER_TO_MAG_MUX_H

#include <gr_block.h>


class listener_to_mag_mux;
typedef boost::shared_ptr<listener_to_mag_mux> listener_to_mag_mux_sptr;


listener_to_mag_mux_sptr 
listener_make_to_mag_mux();

class listener_to_mag_mux : public gr_block 
{
 
 private:
  friend listener_to_mag_mux_sptr
  listener_make_to_mag_mux();
  
  listener_to_mag_mux();
  void forecast (int noutput_items, gr_vector_int &ninput_items_required); 
  

 public:
  ~listener_to_mag_mux();

  int general_work(int noutput_items, 
		   gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);
  
  
};

#endif

