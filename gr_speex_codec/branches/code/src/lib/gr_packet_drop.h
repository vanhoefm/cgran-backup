/* -*- c++ -*- */
/*
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

#ifndef INCLUDED_GR_PACKET_DROP_H
#define INCLUDED_GR_PACKET_DROP_H



#include <gr_random.h>

#include <speex/speex.h>
#include <gr_sync_interpolator.h>

#define FRAME_SIZE 160
#define QUALITY_OPTIONS 11


class gr_packet_drop;
typedef boost::shared_ptr<gr_packet_drop> gr_packet_drop_sptr;

gr_packet_drop_sptr
gr_make_packet_drop (float drop_rate,int quality = 8,int mode = 0);

/*!
* \brief This is the same as the speex decoder, but implements an
*
* additional , packet drop logic. Based on the input packet drop rate
*
* configured, the block drops/allows the speex encoded data packets 
*
* to be decoded or dropped.
*/

class gr_packet_drop : public gr_sync_interpolator
{
	friend gr_packet_drop_sptr gr_make_packet_drop (float drop_rate,int quality,int mode);
 private:
       	gr_packet_drop(float drop_rate,int quality,int mode);
        float d_drop_rate;
        gr_random d_rng;
        int d_quality;
        void* d_state;
        SpeexBits d_bits;
        int d_enhance;           
        int d_mode; 

 public:
      
        ~gr_packet_drop ();
        
        int work(int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items); 	
        
};

#endif /* INCLUDED_GR_PACKET_DROP_H */
