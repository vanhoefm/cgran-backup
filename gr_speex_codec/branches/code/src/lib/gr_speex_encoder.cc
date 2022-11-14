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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gr_speex_encoder.h>
#include <gr_io_signature.h>
#include <speex/speex.h>


int speex_enc_frame[QUALITY_OPTIONS] = {6,10,15,20,20,28,28,38,38,46,62}; ///< Array to hold the ouput bytes based on the bit rate
char speexenc_frame[38];		/* 38 * 8 bits	 */
int mode_id[3] = {1,2,4}; ///< Identifier for the mode


/*!
* \brief Constructor for the encoder. Configuration of the encoder based on the
* input values is done here.
* 
* sampling_rate : sets the sampling rate to be used in encoding.
* 
* quality : A parameter of value 0-10, mapped to the bit rate. Ex quality 8 corresponds to 15kbps
* 
* mode : sets the mode for encoding can be a value 0,1 or 2 corresponding to narrowband,wideband,ultrawideband respectively
* 
* complexity : A value from 0-10, sets the compexity for the encoder.
* 
* vad_enabled : A flag to set/reset the Voice Activity Detection.
*
* \ingroup source
*/


gr_speex_encoder_sptr
gr_make_speex_encoder (int sampling_rate,int quality,int mode,int complexity,int vad_enabled)
{
  return gr_speex_encoder_sptr (new gr_speex_encoder(sampling_rate,quality,mode,complexity,vad_enabled));
}

gr_speex_encoder::gr_speex_encoder (int sampling_rate , int quality, int mode,int complexity,int vad_enabled)
  : gr_sync_decimator ("speex_encoder",
		   gr_make_io_signature (1, 1, sizeof(float)),
                   gr_make_io_signature (1, 1, (sizeof(char)* (speex_enc_frame[quality]) *(mode_id[mode]))),FRAME_SIZE*mode_id[mode]),
                   d_sampling_rate(sampling_rate),d_quality(quality),d_mode(mode),d_complexity(complexity),d_vad(vad_enabled)

{ 
    int dtx_enabled = 0;
    
    //Create a new encoder state in narrow/wide/ultrawide mode
    if(d_mode == 0)
    {
        d_state =speex_encoder_init(speex_lib_get_mode(SPEEX_MODEID_NB));
        
    }
    else if(d_mode ==1)
    {
        d_state = speex_encoder_init(speex_lib_get_mode(SPEEX_MODEID_WB));
        
    }
    else if(d_mode == 2)
    {
        d_state = speex_encoder_init(speex_lib_get_mode(SPEEX_MODEID_UWB));
        
    }
    //Set the quality as per the input( ex: 8 => 15 kbps )
    speex_encoder_ctl(d_state, SPEEX_SET_QUALITY, &d_quality);
    
    // Set the sampling rate 
    speex_encoder_ctl(d_state, SPEEX_SET_SAMPLING_RATE, &d_sampling_rate);

    // Set the complexity 
    speex_encoder_ctl(d_state,SPEEX_SET_COMPLEXITY,&d_complexity);

    if(d_vad)
    {
        // Set the VAD- Voice Activity Detection if enabled
        speex_encoder_ctl(d_state,SPEEX_SET_VAD,&d_vad);
        dtx_enabled = 1;
        // Discontinous Transmission DTX is useful option only with VAD.
        speex_encoder_ctl(d_state,SPEEX_SET_DTX,&dtx_enabled);
    }

    //Initialization of the structure that holds the bits
    speex_bits_init(&d_bits);


}

/*!
* \brief Destructor of the speex encoder. 
* 
* Destroys the state handler to the encoder and the speex bits
* 
*/
gr_speex_encoder::~gr_speex_encoder ()
{

    //Destroy the encoder state
    speex_encoder_destroy(d_state);
    //Destroy the bit-packing struct
    speex_bits_destroy(&d_bits);


}

/*!
* \brief Function that implements the actual encoding.
*  
*  Audio samples are first written into speex bits which is then
*  
*  encoded and output into a buffer.The FRAME_SIZE used for encoding is fixed at 
*
*  160, 320, 640 bytes respectively for narrow, wide and ultrawide band modes.
*  
*  The input audio stream is thus, encoded as frames of size FRAME_SIZE.
*  
*  The output encoded bits, is written to buffer of an arbitrary
*  
*  size, but which is based on the bit rate chosen.For example, 
*
*  for a bit rate of 15kbps(quality = 8), the output 
*  
*  data buffer is of the size 38 bytes.
*
*/

int gr_speex_encoder::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{

  float *in = ( float *) input_items[0];
  char *out = (char *) output_items[0];


   int nbBytes;
   int bytes;
   

   for (int i = 0; i < noutput_items; i++)
   {
       speex_bits_reset(&d_bits);

      //Encode the frame
      speex_encode(d_state,in,&d_bits);

       bytes = speex_bits_nbytes(&d_bits);
       //Copy the bits to an array of char that can be written
       nbBytes = speex_bits_write(&d_bits, out, bytes);
      in += (FRAME_SIZE * mode_id[d_mode]);
      out += (sizeof(char)* speex_enc_frame[d_quality] * mode_id[d_mode]);
  

   }

    return noutput_items;

}
