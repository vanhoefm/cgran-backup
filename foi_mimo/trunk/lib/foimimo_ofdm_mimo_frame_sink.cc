/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2007,2008,2010 Free Software Foundation, Inc.
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
// This is a modification of gr_ofdm_frame_sink.cc from GNU Radio.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_ofdm_mimo_frame_sink.h>
#include <gr_io_signature.h>
#include <gr_expj.h>
#include <gr_math.h>
#include <math.h>
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <string.h>

#define VERBOSE 0

inline void
foimimo_ofdm_mimo_frame_sink::enter_search()
{
  if (VERBOSE)
    fprintf(stderr, "@ enter_search\n");
  d_state = STATE_SYNC_SEARCH;
}
    
inline void
foimimo_ofdm_mimo_frame_sink::enter_have_sync()
{
  if (VERBOSE)
    fprintf(stderr, "@ enter_have_sync\n");
  d_state = STATE_HAVE_FIRST_PREAMBLE;
}

inline void
foimimo_ofdm_mimo_frame_sink::enter_have_header()
{
  d_state = STATE_HAVE_HEADER;

  // header consists of two 16-bit shorts in network byte order
  // payload length is lower 12 bits
  // whitener offset is upper 4 bits
  d_packetlen =  (d_header >> 16) & 0x0fff;
  d_packet_whitener_offset = (d_header >> 28) & 0x000f;
  d_packetlen_cnt = 0;

  if (VERBOSE)
    fprintf(stderr, "@ enter_have_header (payload_len = %d) (offset = %d)\n", 
	    d_packetlen, d_packet_whitener_offset);
}

inline void
foimimo_ofdm_mimo_frame_sink::have_first_preamble()
{
  if(VERBOSE)
    printf("Entered function have first preamble\n");
  d_state = STATE_HAVE_SYNC;
  // clear state of demapper
  d_byte_offset = 0;
  d_partial_byte = 0;

  d_header = 0;
  d_headerbytelen_cnt = 0;

  // Resetting PLL
  d_freq = 0.0;
  d_phase = 0.0;
  fill(d_dfe.begin(), d_dfe.end(), gr_complex(1.0,0.0));
  if(VERBOSE)
    printf("About to leave function have first preamble\n");
}


unsigned char foimimo_ofdm_mimo_frame_sink::slicer(const gr_complex x, float &out_min_dist)
{
  unsigned int table_size = d_sym_value_out.size();
  unsigned int min_index = 0;
  float min_euclid_dist = norm(x - d_sym_position[0]);
  float euclid_dist = 0;
  
  for (unsigned int j = 1; j < table_size; j++){
    euclid_dist = norm(x - d_sym_position[j]);
    if (euclid_dist < min_euclid_dist){
      min_euclid_dist = euclid_dist;
      min_index = j;
    }
  }

  out_min_dist = min_euclid_dist;

  return d_sym_value_out[min_index];
}

unsigned int foimimo_ofdm_mimo_frame_sink::demapper(const gr_complex *in,
					  unsigned char *out)
{
  unsigned int i=0, bytes_produced=0;
  gr_complex carrier;

  carrier=gr_expj(d_phase);

  float dist, snrest;
  float distsum = 0;

  gr_complex accum_error = 0.0;
  while(i < d_subcarrier_map.size()) {
    if(d_nresid > 0) {
      d_partial_byte |= d_resid;
      d_byte_offset += d_nresid;
      d_nresid = 0;
      d_resid = 0;
    }
    
    while((d_byte_offset < 8) && (i < d_subcarrier_map.size())) {
      gr_complex sigrot = in[d_subcarrier_map[i]]*carrier*d_dfe[i];
      
      if(d_derotated_output != NULL){
    	d_derotated_output[i] = sigrot;
      }
      
      unsigned char bits = slicer(sigrot, dist);
      distsum += 1.0/dist;

      gr_complex closest_sym = d_sym_position[bits];
      
      accum_error += sigrot * conj(closest_sym);

      // FIX THE FOLLOWING STATEMENT
      if (norm(sigrot)> 0.001) d_dfe[i] +=  d_eq_gain*(closest_sym/sigrot-d_dfe[i]);
      
      i++;

      if((8 - d_byte_offset) >= d_nbits) {
    	d_partial_byte |= bits << (d_byte_offset);
	    d_byte_offset += d_nbits;
      }
      else {
	    d_nresid = d_nbits-(8-d_byte_offset);
	    int mask = ((1<<(8-d_byte_offset))-1);
	    d_partial_byte |= (bits & mask) << d_byte_offset;
	    d_resid = bits >> (8-d_byte_offset);
	    d_byte_offset += (d_nbits - d_nresid);
      }
    }

    if(d_byte_offset == 8) {
      out[bytes_produced++] = d_partial_byte;
      d_byte_offset = 0;
      d_partial_byte = 0;
    }
  }
  float angle = arg(accum_error);

  d_freq = d_freq - d_freq_gain*angle;
  d_phase = d_phase + d_freq - d_phase_gain*angle;
  if (d_phase >= 2*M_PI) d_phase -= 2*M_PI;
  if (d_phase <0) d_phase += 2*M_PI;
    
  //if(VERBOSE)
  //  std::cerr << angle << "\t" << d_freq << "\t" << d_phase << "\t" << std::endl;

  snrest = sqrt(distsum / i);
  //std::cerr << "SNR Est: " << 20.0*log10(snrest) << std::endl;
  if(VERBOSE)
    printf("demapper: bytes produced: %i \n", bytes_produced);
  return bytes_produced;
}


foimimo_ofdm_mimo_frame_sink_sptr
foimimo_make_ofdm_mimo_frame_sink(const std::vector<gr_complex> &sym_position, 
			const std::vector<unsigned char> &sym_value_out,
			gr_msg_queue_sptr target_queue, unsigned int occupied_carriers,
			float phase_gain, float freq_gain)
{
  return foimimo_ofdm_mimo_frame_sink_sptr(new foimimo_ofdm_mimo_frame_sink(sym_position, sym_value_out,
							target_queue, occupied_carriers,
							phase_gain, freq_gain));
}


foimimo_ofdm_mimo_frame_sink::foimimo_ofdm_mimo_frame_sink(const std::vector<gr_complex> &sym_position, 
				       const std::vector<unsigned char> &sym_value_out,
				       gr_msg_queue_sptr target_queue, unsigned int occupied_carriers,
				       float phase_gain, float freq_gain)
  : gr_sync_block ("ofdm_mimo_frame_sink",
		   gr_make_io_signature2 (2, 2, sizeof(gr_complex)*occupied_carriers, sizeof(char)),
		   gr_make_io_signature (1, 1, sizeof(gr_complex)*occupied_carriers)),
    d_target_queue(target_queue), d_occupied_carriers(occupied_carriers), 
    d_byte_offset(0), d_partial_byte(0),
    d_resid(0), d_nresid(0),d_phase(0),d_freq(0),d_phase_gain(phase_gain),d_freq_gain(freq_gain),
    d_eq_gain(0.05)
{
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

    diff_left = (int)ceil((float)diff/2.0f);  // number of carriers to put on the left side
    c[0] = abc[(1 << diff_left) - 1];         // convert to bits and move to ASCI integer
    carriers.insert(0, c);
    
    diff_right = diff - diff_left;	      // number of carriers to put on the right side
    c[0] = abc[0xF^((1 << diff_right) - 1)];  // convert to bits and move to ASCI integer
    carriers.insert(carriers.length(), c);
  }

  // It seemed like such a good idea at the time...
  // because we are only dealing with the occupied_carriers
  // at this point, the diff_left in the following compensates
  // for any offset from the 0th carrier introduced
  unsigned int i,j,k;
  for(i = 0; i < (d_occupied_carriers/4)+diff_left; i++) {
    char c = carriers[i];
    for(j = 0; j < 4; j++) {
      k = (strtol(&c, NULL, 16) >> (3-j)) & 0x1;
      if(k) {
	d_subcarrier_map.push_back(4*i + j - diff_left);
      }
    }
  }
  
  // make sure we stay in the limit currently imposed by the occupied_carriers
  if(d_subcarrier_map.size() > d_occupied_carriers) {
    throw std::invalid_argument("foimimo_ofdm_mimo_frame_sink: subcarriers allocated exceeds size of occupied carriers");
  }

  d_bytes_out = new unsigned char[d_occupied_carriers];
  d_dfe.resize(occupied_carriers);
  fill(d_dfe.begin(), d_dfe.end(), gr_complex(1.0,0.0));
  set_sym_value_out(sym_position, sym_value_out);
  
  enter_search();
}

foimimo_ofdm_mimo_frame_sink::~foimimo_ofdm_mimo_frame_sink ()
{
  delete [] d_bytes_out;
}

bool
foimimo_ofdm_mimo_frame_sink::set_sym_value_out(const std::vector<gr_complex> &sym_position, 
				      const std::vector<unsigned char> &sym_value_out)
{
  if (sym_position.size() != sym_value_out.size())
    return false;

  if (sym_position.size()<1)
    return false;

  d_sym_position  = sym_position;
  d_sym_value_out = sym_value_out;
  d_nbits = (unsigned long)ceil(log10(d_sym_value_out.size()) / log10(2.0));

  return true;
}


int
foimimo_ofdm_mimo_frame_sink::work (int noutput_items,
			  gr_vector_const_void_star &input_items,
			  gr_vector_void_star &output_items)
{
  const gr_complex *in = (const gr_complex *) input_items[0];
  const char *sig = (const char *) input_items[1];
  unsigned int j = 0;
  unsigned int bytes=0;
  
  // If the output is connected, send it the derotated symbols
  if(output_items.size() >= 1)
    d_derotated_output = (gr_complex *)output_items[0];
  else
    d_derotated_output = NULL;
  
  if (VERBOSE)
    fprintf(stderr,">>> Entering state machine\n");

  switch(d_state) {
      
  case STATE_SYNC_SEARCH:    // Look for flag indicating beginning of pkt
    if (VERBOSE)
      fprintf(stderr,"SYNC Search, noutput=%d\n", noutput_items);
    
    if (sig[0]) {  // Found it, set up for header decode
      enter_have_sync();
    }
    break;
  case STATE_HAVE_FIRST_PREAMBLE:
    if(VERBOSE)
      printf("Entered state have first preamble\n");
    have_first_preamble();
    break;
  case STATE_HAVE_SYNC:
    // only demod after getting the preamble signal; otherwise, the 
    // equalizer taps will screw with the PLL performance
    if(VERBOSE){
      fprintf(stdout,"Entered state have sync\n");
      fflush(stdout);
    }
    bytes = demapper(&in[0], d_bytes_out);
    
    if (VERBOSE) {
      if(sig[0])
    	printf("ERROR -- Found SYNC in HAVE_SYNC\n");
      fprintf(stderr,"Header Search bitcnt=%d, header=0x%08x\n", d_headerbytelen_cnt, d_header);
    }

    j = 0;
    while(j < bytes) {
      d_header = (d_header << 8) | (d_bytes_out[j] & 0xFF);
      j++;

      if (++d_headerbytelen_cnt == HEADERBYTELEN) {
	
	    if (VERBOSE)
	      fprintf(stderr, "got header: 0x%08x\n", d_header);
	
	    // we have a full header, check to see if it has been received properly
	    if (header_ok()){
	      enter_have_header();
	  
	      if (VERBOSE)
	        printf("Packet Length: %d\n", d_packetlen);
	  
	      while((j < bytes) && (d_packetlen_cnt < d_packetlen)) {
	        d_packet[d_packetlen_cnt++] = d_bytes_out[j++];
	      }
	  
	      if(d_packetlen_cnt == d_packetlen) {
	        gr_message_sptr msg =
	        gr_make_message(0, d_packet_whitener_offset, 0, d_packetlen);
	        memcpy(msg->msg(), d_packet, d_packetlen_cnt);
	        d_target_queue->insert_tail(msg);		// send it
	        msg.reset();  				// free it up
	    
	        enter_search();				
	      }
	    } 
	    else {
	      enter_search();				// bad header
	    }
      }
    }
    break;
      
  case STATE_HAVE_HEADER:
    bytes = demapper(&in[0], d_bytes_out);

    if (VERBOSE) {
      if(sig[0])
    	printf("ERROR -- Found SYNC in HAVE_HEADER at %d, length of %d\n", d_packetlen_cnt, d_packetlen);
      fprintf(stderr,"Packet Build\n");
    }
    
    j = 0;
    while(j < bytes) {
      d_packet[d_packetlen_cnt++] = d_bytes_out[j++];
      
      if (d_packetlen_cnt == d_packetlen){		// packet is filled
	    // build a message
	    // NOTE: passing header field as arg1 is not scalable
	    gr_message_sptr msg = gr_make_message(0, d_packet_whitener_offset, 0, d_packetlen_cnt);
	    memcpy(msg->msg(), d_packet, d_packetlen_cnt);
	
	    d_target_queue->insert_tail(msg);		// send it
	    msg.reset();  				// free it up
	
   	    enter_search();
	    break;
      }
    }
    break;
    
  default:
    assert(0);
  } // switch

  if(VERBOSE)
    printf("Leaving work in frame sink\n\n");
  return 1;
}
