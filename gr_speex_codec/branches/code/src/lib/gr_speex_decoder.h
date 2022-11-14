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

#ifndef INCLUDED_GR_SPEEX_DECODER_H
#define INCLUDED_GR_SPEEX_DECODER_H


#include <speex/speex.h>
#include <gr_sync_interpolator.h>

#define FRAME_SIZE 160
#define QUALITY_OPTIONS 11



class gr_speex_decoder;
typedef boost::shared_ptr<gr_speex_decoder> gr_speex_decoder_sptr;

gr_speex_decoder_sptr
gr_make_speex_decoder (int quality = 8,int mode = 0);

/*!
* \brief Takes the SPEEX encoded bits as input, decodes the data and outputs
* to an audio sink. The output can be a wav file or an audio sink like speakers.
*/

class gr_speex_decoder : public gr_sync_interpolator
{
	friend gr_speex_decoder_sptr gr_make_speex_decoder (int quality,int mode);
 private:
	gr_speex_decoder(int quality,int mode); ///< Function to decode. Quality and mode are the inputs
         
        int d_quality;      ///< Quality parameter (0-10) corresponds to the bit rate
        void* d_state;      ///< Decoder state handler
        SpeexBits d_bits;   ///< Speex bits which contain the encoded bits
        int d_enhance;      ///< Flag to turn on the perceptual enhancement 
        int d_mode;         ///< Mode identifier (narrow/wide/ultrawide band)            
 
	

 public:
              
        ~gr_speex_decoder ();
        int work(int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items); 	

};

#endif /* INCLUDED_GR_SPEEX_DECODER_H */
