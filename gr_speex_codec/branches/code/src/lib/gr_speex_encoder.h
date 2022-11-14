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

#ifndef INCLUDED_GR_SPEEX_ENCODER_H
#define INCLUDED_GR_SPEEX_ENCODER_H


#include <speex/speex.h>
#include <gr_sync_decimator.h>

#define FRAME_SIZE 160
#define QUALITY_OPTIONS 11



class gr_speex_encoder;
typedef boost::shared_ptr<gr_speex_encoder> gr_speex_encoder_sptr;

gr_speex_encoder_sptr
gr_make_speex_encoder (int sampling_rate = 8000, int quality = 8,int mode = 0,int complexity = 3,int vad_enabled =0);

/*!
 * \brief Takes audio samples as input and encodes the input data into SPEEX encoded data stream.
 * The audio input can be a wav file source or microphone input. The output is usually used as the input for
 * the SPEEX decoder
 * \ingroup source
 */

class gr_speex_encoder : public gr_sync_decimator
{
	friend gr_speex_encoder_sptr gr_make_speex_encoder (int sampling_rate, int quality,int mode,int complexity,int vad_enabled);
 private:
       	gr_speex_encoder(int sampling_rate,int quality,int mode,int complexity,int vad_enabled);
        
        int d_sampling_rate; ///< Holds the sampling rate (Hz), as configured by the input 
        void* d_state;       ///< Speex encoder state handler
        SpeexBits d_bits;    ///< Speex bits which contain the encoded data stream
        int d_quality;       ///< Quality parameter(0-10), mapped to the bit rate
        int d_mode;          ///< Mode identifier (narrow/wide/ultrawide band modes)
        int d_vad;           ///< Voice activity detection enable/disable flag
        int d_complexity;    ///< Encoder complexity (0-10)


 public: 
 
        ~gr_speex_encoder ();
         int work(int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items);

};

#endif /* INCLUDED_GR_SPEEX_ENCODER_H */
