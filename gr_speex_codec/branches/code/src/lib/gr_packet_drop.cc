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

#include <gr_packet_drop.h>
#include <gr_io_signature.h>
#include <gr_random.h>
#include <speex/speex.h>
#include <assert.h>


int speex_pdrop_frame[QUALITY_OPTIONS] = {6,10,15,20,20,28,28,38,38,46,62};
char speexpdrop_frame[38];		/* 38 * 8 bits	 */
int pdrop_mode[3] = {1,2,4};



/*!
* \brief Constructor for the packet drop implemented decoder. 
*
* Configuration of the packet drop decoder based on the input values is done here.
*
* drop_rate : A parameter to configure the packet drop rate, could be any value from 0-100%
*
* quality : A parameter of value 0-10, mapped to the bit rate. Ex quality 8 corresponds to 15kbps.
* 
* mode : sets the mode for encoding can be a value 0,1 or 2 corresponding to narrowband,wideband,ultrawideband respectively.
*
* \ingroup source
*/

gr_packet_drop_sptr
gr_make_packet_drop (float drop_rate,int quality,int mode)
{
  return gr_packet_drop_sptr (new gr_packet_drop(drop_rate,quality,mode));

}
gr_packet_drop::gr_packet_drop (float drop_rate,int quality,int mode)
    : gr_sync_interpolator ("packet_drop",
			  gr_make_io_signature (1, 1, (sizeof(char)* speex_pdrop_frame[quality] * pdrop_mode[mode])),
			  gr_make_io_signature (1, 1, sizeof (float)),
			  FRAME_SIZE),d_quality(quality),d_drop_rate(drop_rate),d_rng(2680),d_mode(mode)

{ 
  
    d_enhance = 1;
    /*Create a new decoder state in narrowband mode*/
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

    /*Set the perceptual enhancement on*/
    speex_decoder_ctl(d_state, SPEEX_SET_ENH, &d_enhance);
    /*Initialization of the structure that holds the bits*/
    speex_bits_init(&d_bits);



    

}

/*!
* \brief Destructor of the SPEEX decoder. 
*
* Destroys the decoder state handler and the speex bits.
*/
gr_packet_drop::~gr_packet_drop ()
{
        /*Destroy the decoder state*/
    speex_decoder_destroy(d_state);

    /*Destroy the bit-stream truct*/
    speex_bits_destroy(&d_bits);




}


/*!
* \brief The main function which implements the actual
* decoding. 
*
* The packet drop block implements a simple logic to drop random
*
* packets.A sequence of random numbers between 0 and 1 is generated.
*
* The drop rate is taken from the input to the block, the random number is compared
*
* with the drop rate configured, and decision is made to drop or allow the packet.
*
* In case, the packet needs to be dropped, NULL bits are sent to be decoded for the speex
*
* deocder. If the packet is to be passed, then the encoded bits received are passed to the 
*
* speex decoder. The other features are parameters are same as the speex decoder block.
*
* 
*/

int
gr_packet_drop::work(int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)

{
  
    int nbBytes;
    
    float *out = (float *) output_items[0];
    char *in = (char *) input_items[0];
    int pass = 1;
      
     nbBytes = speex_pdrop_frame[d_quality];     
     assert ((noutput_items % (FRAME_SIZE * pdrop_mode[d_mode])) == 0);


     for (int i = 0; i < noutput_items; i += (FRAME_SIZE * pdrop_mode[d_mode]))
     {

          
        if ((d_rng.ran1()) < (1-d_drop_rate))
        {
	    pass = 1;
        }
        else if ((d_rng.ran1()) >= (1-d_drop_rate))
        {
            pass = 0;
        }

          /*Copy the data into the bit-stream struct*/
          speex_bits_read_from(&d_bits, in, nbBytes);

          /*Decode the data*/
          if(pass ==0)
          {
               speex_decode(d_state, NULL, out);
          }
          else
          {  
              speex_decode(d_state, &d_bits, out);
          }
          speex_bits_reset(&d_bits);

          in += (sizeof(char)* speex_pdrop_frame[d_quality] * pdrop_mode[d_mode]);
          out += (FRAME_SIZE* pdrop_mode[d_mode]);

     }


return noutput_items;

}



