/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2006,2007,2008,2010 Free Software Foundation, Inc.
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
// This is a modification of gr_ofdm_mapper_bcv.cc from GNU Radio.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_ofdm_mapper_bcv.h>
#include <gr_io_signature.h>
#include <stdexcept>
#include <string.h>

foimimo_ofdm_mapper_bcv_sptr
foimimo_make_ofdm_mapper_bcv (const std::vector<gr_complex> &constellation,
    unsigned int occupied_carriers, unsigned int fft_length)
{
  return foimimo_ofdm_mapper_bcv_sptr (new foimimo_ofdm_mapper_bcv(constellation,
      occupied_carriers, fft_length));
}

// Consumes 1 packet and produces as many OFDM symbols of fft_length to hold the full packet
foimimo_ofdm_mapper_bcv::foimimo_ofdm_mapper_bcv (const std::vector<gr_complex> &constellation,
                                        unsigned int occupied_carriers, unsigned int fft_length)
  : gr_block ("mimo_ofdm_mapper_bcv",
                gr_make_io_signature2 (2, 2, sizeof(char),sizeof(char)),
                gr_make_io_signature2 (2, 2, sizeof(gr_complex)*fft_length, sizeof(char))),
    d_constellation(constellation),
    d_occupied_carriers(occupied_carriers),
    d_fft_length(fft_length),
    d_bit_offset(0),
    d_resid(0),
    d_nresid(0)
{
  //std::cout << "Output mult = " << d_output_mult << std::endl;
  if (!(d_occupied_carriers <= d_fft_length)) 
    throw std::invalid_argument("foimimo_ofdm_mapper_bcv: occupied carriers must be <= fft_length");

  // this is not the final form of this solution since we still use the occupied_tones concept,
  // which would get us into trouble if the number of carriers we seek is greater than the occupied carriers.
  // Eventually, we will get rid of the occupied_carriers concept.
  std::string carriers = "FC3F";

  // A bit hacky to fill out carriers to occupied_carriers length
  int diff = (d_occupied_carriers - 4*carriers.length()); 
  while(diff > 7) {
    carriers.insert(0, "f");
    carriers.insert(carriers.length(), "f");
    diff -= 8;
  }

  // if there's extras left to be processed
  // divide remaining to put on either side of current map
  // all of this is done to stick with the concept of a carrier map string that
  // can be later passed by the user, even though it'd be cleaner to just do this
  // on the carrier map itself
  int diff_left=0;
  int diff_right=0;

  // dictionary to convert from integers to ascii hex representation
  char abc[16] = {'0', '1', '2', '3', '4', '5', '6', '7', 
		  '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  if(diff > 0) {
    char c[2] = {0,0};

    diff_left = (int)ceil((float)diff/2.0f);   // number of carriers to put on the left side
    c[0] = abc[(1 << diff_left) - 1];          // convert to bits and move to ASCI integer
    carriers.insert(0, c);
    
    diff_right = diff - diff_left;	       // number of carriers to put on the right side
    c[0] = abc[0xF^((1 << diff_right) - 1)];   // convert to bits and move to ASCI integer
        carriers.insert(carriers.length(), c);
  }
  
  // find out how many zeros to pad on the sides; the difference between the fft length and the subcarrier
  // mapping size in chunks of four. This is the number to pack on the left and this number plus any 
  // residual nulls (if odd) will be packed on the right. 
  diff = (d_fft_length/4 - carriers.length())/2; 

  unsigned int i,j,k;
  for(i = 0; i < carriers.length(); i++) {
    char c = carriers[i];                            // get the current hex character from the string
    for(j = 0; j < 4; j++) {                         // walk through all four bits
      k = (strtol(&c, NULL, 16) >> (3-j)) & 0x1;     // convert to int and extract next bit
      if(k) {                                        // if bit is a 1, 
	d_subcarrier_map.push_back(4*(i+diff) + j);  // use this subcarrier
      }
    }
  }

  // make sure we stay in the limit currently imposed by the occupied_carriers
  if(d_subcarrier_map.size() > d_occupied_carriers) {
    throw std::invalid_argument("foimimo_ofdm_mapper_bcv: subcarriers allocated exceeds size of occupied carriers");
  }
  
  d_nbits = (unsigned long)ceil(log10(d_constellation.size()) / log10(2.0));
}

foimimo_ofdm_mapper_bcv::~foimimo_ofdm_mapper_bcv(void)
{
}

int foimimo_ofdm_mapper_bcv::randsym()
{
  return (rand() % d_constellation.size());
}

void
foimimo_ofdm_mapper_bcv::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  int nreqd  = (int)ceil((d_subcarrier_map.size()*d_nbits)/8.0)*noutput_items;
  unsigned ninputs = ninput_items_required.size ();
  for (unsigned i = 0; i < ninputs; i++)
    ninput_items_required[i] = nreqd;
}

int
foimimo_ofdm_mapper_bcv::general_work(int noutput_items,
                          gr_vector_int &ninput_items,
			  gr_vector_const_void_star &input_items,
			  gr_vector_void_star &output_items)
{
  const unsigned char *in = (const unsigned char *) input_items[0];
  const unsigned char *in_new_pkt = (const unsigned char *) input_items[1];
  gr_complex *out = (gr_complex *)output_items[0];
  int ninput_bytes = std::min(ninput_items[0],ninput_items[1]);

  char *out_flag = (char *) output_items[1];
  out_flag[0] = (char)in_new_pkt[0];

  unsigned int i=0;

  //printf("OFDM BPSK Mapper:  ninput_items: %d   noutput_items: %d\n", ninput_items[0], noutput_items);



  // Build a single symbol:
  // Initialize all bins to 0 to set unused carriers
  memset(out, 0, d_fft_length*sizeof(gr_complex));
  
  i = 0;
  unsigned char bits = 0;
  unsigned int byte_cnt = 0;

  while((byte_cnt < ninput_bytes) && (i < d_subcarrier_map.size())) {

    // need new data to process
    if(d_bit_offset == 0) {
      d_msgbytes = in[byte_cnt];
      //printf("mod message byte: %x\n", d_msgbytes);
    }

    // Modulate the byte
    if(d_nresid > 0) {
      // take the residual bits, fill out nbits with info from the new byte, and put them in the symbol
      d_resid |= (((1 << d_nresid)-1) & d_msgbytes) << (d_nbits - d_nresid);
      bits = d_resid;

      out[d_subcarrier_map[i]] = d_constellation[bits];
      i++;

      d_bit_offset += d_nresid;
      d_nresid = 0;
      d_resid = 0;
    }
    else {
      if((8 - d_bit_offset) >= d_nbits) {  // test to make sure we can fit nbits
	// take the nbits number of bits at a time from the byte to add to the symbol
	bits = ((1 << d_nbits)-1) & (d_msgbytes >> d_bit_offset);
	d_bit_offset += d_nbits;
	
	out[d_subcarrier_map[i]] = d_constellation[bits];
	i++;
      }
      else {  // if we can't fit nbits, store them for the next 
	// saves d_nresid bits of this message where d_nresid < d_nbits
	unsigned int extra = 8-d_bit_offset;
	d_resid = ((1 << extra)-1) & (d_msgbytes >> d_bit_offset);
	d_bit_offset += extra;
	d_nresid = d_nbits - extra;
      }
      
    }
    // End of modulate

    if(d_bit_offset == 8) {
      d_bit_offset = 0;
      byte_cnt++;
    }

    // If we have a new packet comming exit the while loop and add some random
    // constelation points to the end of the OFDM-symbol
    if (byte_cnt > 0 && in_new_pkt[byte_cnt] == 1){
      //printf("foi_ofdm_mapper> found a new_pkt while we didn't expect that.\n");
      break;
    }
  }

  // Ran out of data to put in symbol
  if (in_new_pkt[byte_cnt] == 1 || byte_cnt == ninput_bytes){
    while(i < d_subcarrier_map.size()) {   // finish filling out the symbol
      out[d_subcarrier_map[i]] = d_constellation[randsym()];
      i++;
    }
  }
  if (byte_cnt == ninput_bytes) {
    if(d_nresid > 0) {
      d_resid |= 0x00;
      d_nresid = 0;
      d_resid = 0;
    }
    
    assert(d_bit_offset == 0);
  }
  
  consume_each(byte_cnt);
  return 1;  // produced symbol
}
