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
// This is a modification of gr_ofdm_frame_sink.cc from GNU Radio

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_ofdm_demapper.h>
#include <gr_io_signature.h>
#include <gr_expj.h>
#include <cstdio>
#include <iostream>

#define VERBOSE 0

inline void
foimimo_ofdm_demapper::enter_search()
{
  d_state = STATE_SYNC_SEARCH;
  d_phase = 0;
  d_freq = 0;
  fill(d_dfe.begin(), d_dfe.end(), gr_complex(1.0,0.0));
  d_packetlen_cnt = 0;
  d_packetlen = 1;
  if (VERBOSE)
    fprintf(stderr, "@ enter_sync_search \n");
}

inline void
foimimo_ofdm_demapper::enter_have_sync()
{
  d_state = STATE_HAVE_SYNC;
  d_packetlen_cnt = 0;
  d_packetlen = 1;
  if (VERBOSE)
    fprintf(stderr, "@ enter_have_sync\n");
}

inline void
foimimo_ofdm_demapper::enter_have_first_symbol()
{
  d_state = STATE_HAVE_FIRST_SYMBOL;
  if (VERBOSE)
    fprintf(stderr, "@ enter_have_first_symbol \n");
}

inline void
foimimo_ofdm_demapper::enter_have_header()
{
  d_state = STATE_HAVE_HEADER;

  // header consists of two 16-bit shorts in network byte order
  // payload length is lower 12 bits
  // 4 upper bit is unused
  d_packetlen =  (d_header >> 16) & 0x0fff;


  if (VERBOSE)
    fprintf(stderr, "@ enter_have_header (raw payload_len = %d)",d_packetlen);

  d_packetlen = ceil((d_packetlen*8.0)/d_code_k);
  d_packetlen = (d_packetlen*d_code_n)/d_nbits;

  if (VERBOSE)
    fprintf(stderr, ", (number of constellation points = %d)\n",d_packetlen);

}

unsigned char foimimo_ofdm_demapper::slicer(const gr_complex x, float &out_min_dist)
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

void
foimimo_ofdm_demapper::demodulate_header(const gr_complex *in){
  unsigned int i=0;
  gr_complex carrier;

  carrier=gr_expj(d_phase);

  float dist, snrest;
  float distsum = 0;

  unsigned int header_offset = 3;
  gr_complex accum_error = 0.0;
   while(i < HEADERCONSTPOINTS) {
    // check if we have some saved bits that we have to put in this byte
    if(d_nresid > 0) {
      d_partial_byte |= d_resid;
      d_byte_offset += d_nresid;
      d_nresid = 0;
      d_resid = 0;
    }

    while((d_byte_offset < 8) && (i < HEADERCONSTPOINTS)) {
      gr_complex sigrot = in[d_subcarrier_map[i]]*carrier*d_dfe[i];
      unsigned char bits = slicer(sigrot, dist);
      distsum += 1.0/dist;

      gr_complex closest_sym = d_sym_position[bits];

      accum_error += sigrot * conj(closest_sym);

      // FIX THE FOLLOWING STATEMENT
      if (norm(sigrot)> 0.001) d_dfe[i] +=  d_eq_gain*(closest_sym/sigrot-d_dfe[i]);

      i++;
      // make up one byte
      if((8 - d_byte_offset) >= d_nbits) {
        d_partial_byte |= bits << (d_byte_offset);
            d_byte_offset += d_nbits;
      }
      else { // there are some bits that doesn't fit in this byte save them.
        d_nresid = d_nbits-(8-d_byte_offset);
        int mask = ((1<<(8-d_byte_offset))-1);
        d_partial_byte |= (bits & mask) << d_byte_offset;
        d_resid = bits >> (8-d_byte_offset);
        d_byte_offset += (d_nbits - d_nresid);
      }
    }
    // There is one full byte. Merge it into the header
    if(d_byte_offset == 8) {
      d_header = (d_header << 8) | (d_partial_byte & 0xFF);
      header_offset--;
      d_byte_offset = 0;
      d_partial_byte = 0;
    }
  }

  float angle = arg(accum_error);

  d_freq = d_freq - d_freq_gain*angle;
  d_phase = d_phase + d_freq - d_phase_gain*angle;
  if (d_phase >= 2*M_PI) d_phase -= 2*M_PI;
  if (d_phase <0) d_phase += 2*M_PI;

  snrest = sqrt(distsum / i);

  if(VERBOSE){
    fprintf(stderr, "angle:%.6f d_freq:%.6f d_phase:%.6f \n", angle, d_freq, d_phase);
    printf("SNR-estimate: %.4f dB\n",10*log10(snrest));
  }
}

bool
foimimo_ofdm_demapper::header_ok()
{
    // confirm that two copies of header info are identical
  if (VERBOSE)
      fprintf(stderr, "Received header: 0x%08x \n",d_header);

  if(((d_header >> 16) ^ (d_header & 0xffff)) == 0 && d_default_packetlen == (d_header & 0xffff))
    return true;
  else if ((((d_header >> 16) ^ (d_header & 0xffff)) == 0) && (d_default_packetlen < (int)(d_header & 0xffff))){
    d_header = (d_default_packetlen & 0x0fff) | ((d_default_packetlen & 0x0fff) << 16);
    if(VERBOSE)
        fprintf(stderr, "Received header correct BUT the packet length is too LARGE, "
            "setting the length to the default value.\n Default packet length:%i\n",d_default_packetlen);
    return true;
  }else if (((d_header >> 16) ^ (d_header & 0xffff)) == 0){
    d_header = (d_default_packetlen & 0x0fff) | ((d_default_packetlen & 0x0fff) << 16);
    if(VERBOSE)
        fprintf(stderr, "Received header correct BUT the packet length is too SMALL, "
            "setting the length to the default value.\n Default packet length:%i\n",d_default_packetlen);
    return true;
  }
  return false;

 }

foimimo_ofdm_demapper_sptr
foimimo_make_ofdm_demapper (const std::vector<gr_complex> &sym_position,
    const std::vector<unsigned char> &sym_value_out,
    unsigned int occupied_carriers,
    unsigned int code_k, unsigned int code_n, int default_packetlen,
    float phase_gain, float freq_gain, gr_msg_queue_sptr bad_header_queue)
{
  return foimimo_ofdm_demapper_sptr(
      new foimimo_ofdm_demapper(sym_position,sym_value_out,occupied_carriers,
                                code_k,code_n,default_packetlen,phase_gain,freq_gain,bad_header_queue));
}

foimimo_ofdm_demapper::foimimo_ofdm_demapper(const std::vector<gr_complex> &sym_position,
    const std::vector<unsigned char> &sym_value_out,
    unsigned int occupied_carriers,
    unsigned int code_k, unsigned int code_n, int default_packetlen,
    float phase_gain, float freq_gain, gr_msg_queue_sptr bad_header_queue)
  :gr_block("foimimo_ofdm_demapper",
            gr_make_io_signature2(2,2,sizeof(gr_complex)*occupied_carriers,sizeof(char)),
            gr_make_io_signature2(2,2,sizeof(gr_complex),sizeof(char))),
            d_occupied_carriers(occupied_carriers),
            d_code_k(code_k),d_code_n(code_n),d_default_packetlen(default_packetlen),
            d_byte_offset(0), d_partial_byte(0),
            d_resid(0), d_nresid(0),d_phase(0),d_freq(0),d_phase_gain(phase_gain),
            d_freq_gain(freq_gain),d_eq_gain(0.05), d_bad_header_queue(bad_header_queue)
{
  std::string carriers = "FC3F";

  set_output_multiple(d_occupied_carriers-4);
  set_relative_rate(d_occupied_carriers-4);
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

    diff_right = diff - diff_left;            // number of carriers to put on the right side
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
    throw std::invalid_argument("foimimo_ofdm_demapper: subcarriers allocated exceeds size of occupied carriers");
  }
  if (!set_sym_value_out(sym_position, sym_value_out))
       throw std::invalid_argument("foimimo_ofdm_demapper: sym_position and sym_value must be of the same length");


  d_dfe.resize(occupied_carriers);
  fill(d_dfe.begin(), d_dfe.end(), gr_complex(1.0,0.0));

  enter_search(); // set default state
}
bool
foimimo_ofdm_demapper::set_sym_value_out(const std::vector<gr_complex> &sym_position,
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
foimimo_ofdm_demapper::~foimimo_ofdm_demapper(){

}

void
foimimo_ofdm_demapper::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  assert (noutput_items % (d_occupied_carriers-4) == 0);
  int input_required = 1;
  unsigned ninputs = ninput_items_required.size();
  for (unsigned int i = 0; i < ninputs; i++) {
    ninput_items_required[i] = input_required;
  }

}

int
foimimo_ofdm_demapper::general_work(int noutput_items,
    gr_vector_int &ninput_items,
    gr_vector_const_void_star &input_items,
    gr_vector_void_star &output_items)
{
  const gr_complex *in = (const gr_complex*) input_items[0];
  const char *in_sync = (const char*) input_items[1];
  gr_complex *out = (gr_complex*) output_items[0];
  char *out_sync = (char*) output_items[1];

  unsigned int subcarrier_index = 0;

  int optr = 0;

  switch(d_state){ // Begin FSM
  case STATE_SYNC_SEARCH: // Search for a new incoming symbol
    if(in_sync[0]){ // Found the sync pulse
      // Don't do anything with the 1st preamble
      enter_have_sync();
    }
    break;

  case STATE_HAVE_SYNC:
    // Don't do anything with the 2nd preamble
    enter_have_first_symbol();
    break;

  case STATE_HAVE_FIRST_SYMBOL: // We have the first symbol
    if(in_sync[0]){
      enter_have_sync();
      consume_each(1);
      return optr;
    }

    // Find out how many constellation points that are going to be extracted
    demodulate_header(&in[0]);

    // Check if we have a correct header
    if (header_ok()){
      enter_have_header();

      subcarrier_index = HEADERBYTELEN*8/2; // The header is QPSK modulated
                                            // therefore divide by 2

      // extract constellation point from the 1st symbol
      while((subcarrier_index < d_subcarrier_map.size() && d_packetlen_cnt < d_packetlen)){
        // Fixme do we have to compensate for a phase drift?
        out_sync[optr] = 0;
        out[optr++] = in[d_subcarrier_map[subcarrier_index++]];
        d_packetlen_cnt++;
      }
      out_sync[0] = 1;
    }
    else{ // The header was bad
      if (VERBOSE)
          fprintf(stderr, "Bad header: 0x%08x \n",d_header);

      gr_message_sptr msg =
          gr_make_message(0,1,0,sizeof(bool));
      d_bad_header_queue->insert_tail(msg);
      msg.reset();
      enter_search();
    }

    break;

  case STATE_HAVE_HEADER:
    if(in_sync[0] == 1){ // Found the sync pulse
      // Don't do anything with the 1st preamble
      if (VERBOSE)
        fprintf(stderr, "Found a synch @HAVE_HEADER optr:%i pktlen_cnt:%i pktlen:%i\n",optr,d_packetlen_cnt, d_packetlen);

      enter_have_sync();
      consume_each(1);
      return optr;
      
    }
    while((subcarrier_index < d_subcarrier_map.size() && d_packetlen_cnt < d_packetlen)){
      // Fixme do we have to compensate for a phase drift?
      out_sync[optr] = 0;
      out[optr++] = in[d_subcarrier_map[subcarrier_index++]];
      d_packetlen_cnt++;
    }
    break;

  default:
    throw std::runtime_error("foimimo_ofdm_demapper entered an unknown state in work");
    break;
  } // end of FSM

  // We are done with this packet
  if (d_packetlen_cnt == d_packetlen && optr > 0){
    enter_search();
  }

  if (!(optr <= noutput_items)){
    printf("%i, %i \n", optr, noutput_items);
  }

  assert (optr <= noutput_items);
  consume_each(1);
  return optr;
}
