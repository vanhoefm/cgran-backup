/* -*- c++ -*- */
/*
 * Copyright 2004,2006,2007 Free Software Foundation, Inc.
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

#include <gr_speex_decoder.h>
#include <gr_io_signature.h>
#include <speex/speex.h>
#include <assert.h>



int speex_dec_frame[QUALITY_OPTIONS] = {6,10,15,20,20,28,28,38,38,46,62};
char speexdec_frame[38];    /* 38 * 8 bits */
int dec_mode[3] = {1,2,4};


/*!
* \brief Constructor for the decoder. 
*
* Configuration of the decoder based on the input values is done here.
*
* quality : A parameter of value 0-10, mapped to the bit rate. Ex quality 8 corresponds to 15kbps.
* 
* mode : sets the mode for encoding can be a value 0,1 or 2 corresponding to narrowband,wideband,ultrawideband respectively.
*
* \ingroup source
*/
gr_speex_decoder_sptr
gr_make_speex_decoder (int quality,int mode)
{
	return gr_speex_decoder_sptr (new gr_speex_decoder (quality,mode));
}

gr_speex_decoder::gr_speex_decoder(int quality,int mode)
	: gr_sync_interpolator ("speex_decoder",
			  gr_make_io_signature (1, 1, (sizeof(char)* (speex_dec_frame[quality]) * (dec_mode[mode]))),
			  gr_make_io_signature (1, 1, sizeof (float)),
			  FRAME_SIZE*dec_mode[mode]),d_quality(quality),d_mode(mode)	
{

    d_enhance = 1;
    //Create a new decoder state in narrow/wide/ultrawide band mode
    if(d_mode == 0)
    {
        d_state = speex_decoder_init(speex_lib_get_mode(SPEEX_MODEID_NB));
    }
    else if(d_mode == 1)
    {
        d_state = speex_decoder_init(speex_lib_get_mode(SPEEX_MODEID_WB));
    }
    else if(d_mode == 2)
    {
        d_state = speex_decoder_init(speex_lib_get_mode(SPEEX_MODEID_UWB));
    }

   //Set the perceptual enhancement on
   speex_decoder_ctl(d_state, SPEEX_SET_ENH, &d_enhance);


   //Initialization of the structure that holds the bits to be decoded
   speex_bits_init(&d_bits);



}

/*!
* \brief Destructor of the SPEEX decoder. 
*
* Destroys the decoder state handler and the speex bits.
*/
gr_speex_decoder::~gr_speex_decoder ()
{

    //Destroy the decoder state
    speex_decoder_destroy(d_state);

    //Destroy the bit-stream truct
    speex_bits_destroy(&d_bits);

}

/*!
* \brief The main function which implements the actual
* decoding. 
*
* First the encoded bit stream is copied into speex bits 
* 
* and then decoded into audio samples.The input data bits are
*
* collected in buffers of size based on the bit rate(quality) chosen as in
*
* encoding.The bit rate if 15kbps(quality = 8) corresponds to an input buffer
*
* size of 38 bytes. The output audio samples are written into frames of size
*
* 160, 320, 640 bytes respectively for narrow, wide and ultrawide band.
*/

int
gr_speex_decoder::work(int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)

{
  
    int nbBytes;
    
    float *out = (float *) output_items[0];
    char *in = (char *) input_items[0];

     nbBytes = speex_dec_frame[d_quality];     
     assert (noutput_items % (FRAME_SIZE*dec_mode[d_mode]) == 0);


     for (int i = 0; i < noutput_items; i += (FRAME_SIZE*dec_mode[d_mode]))
     {

          //Copy the data into the bit-stream struct
          speex_bits_read_from(&d_bits, in, nbBytes);

          //Decode the data
          speex_decode(d_state, &d_bits, out);
          speex_bits_reset(&d_bits);

          in += (sizeof(char)* speex_dec_frame[d_quality]* dec_mode[d_mode]);
          out += (FRAME_SIZE *dec_mode[d_mode]);

     }


return noutput_items;

}



