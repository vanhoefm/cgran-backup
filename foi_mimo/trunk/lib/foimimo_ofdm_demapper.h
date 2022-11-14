/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 * 
 * Copyright 2007 Free Software Foundation, Inc.
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
// This is a modification of gr_ofdm_frame_sink.h from GNU Radio

#ifndef INCLUDED_FOIMIMO_OFDM_DEMAPPER_H
#define INCLUDED_FOIMIMO_OFDM_DEMAPPER_H

#include <gr_block.h>
#include <foimimo_crc32_checker_sink.h>

class foimimo_ofdm_demapper;
typedef boost::shared_ptr<foimimo_ofdm_demapper> foimimo_ofdm_demapper_sptr;

foimimo_ofdm_demapper_sptr
foimimo_make_ofdm_demapper(const std::vector<gr_complex> &sym_position,
    const std::vector<unsigned char> &sym_value_out,
    unsigned int occupied_carriers,
    unsigned int code_k, unsigned int code_n,int default_packetlen,
    float phase_gain, float freq_gain, gr_msg_queue_sptr bad_header_queue);
/*
 * \brief Takes an OFDM symbol in, removes two preambles before the header is
 * extracted. From the header the number of constellation points are calculated
 * and extracted. When constellation points for an entire OFDM-packet are extracted
 * a complete packet flag is raised for the following block.
 *
 * \param sym_position Coordinates for the header constellation points
 * \param sym_value_out Constellation points values
 * \param occupied_carriers The number of used subcarriers
 * \param code_k The number of useful bits in a codeword
 * \param code_n The total number of bits in a codeword
 *
 * NOTE: This block should be connected to the gr-trellis-metrics block.
 * Code rate = code_k over code_n.
 */

class foimimo_ofdm_demapper : public gr_block
{
  friend foimimo_ofdm_demapper_sptr
  foimimo_make_ofdm_demapper(const std::vector<gr_complex> &sym_position,
      const std::vector<unsigned char> &sym_value_out,
      unsigned int occupied_carriers,
      unsigned int code_k, unsigned int code_n,int default_packetlen,
      float phase_gain, float freq_gain, gr_msg_queue_sptr bad_header_queue);
private:
  enum state_t {STATE_SYNC_SEARCH, STATE_HAVE_SYNC, STATE_HAVE_FIRST_SYMBOL, STATE_HAVE_HEADER};
  state_t d_state;

  static const int MAX_PKT_LEN = 4096;
  static const int HEADERBYTELEN   = 4;
  static const int HEADERCONSTPOINTS = HEADERBYTELEN*8/2; // The header is qpsk modulated

  unsigned int       d_header;                  // header bits
  int                d_headerbytelen_cnt;       // how many so far

  unsigned int       d_occupied_carriers;

  int                d_default_packetlen;       // preknowledge of the packet length
  int                d_packetlen;               // length of packet
  int                d_packetlen_cnt;           // how many so far

  std::vector<int>   d_subcarrier_map;          // utilized subcarriers

  unsigned int       d_code_k;                  // useful bits in codeword
  unsigned int       d_code_n;                  // codeword length

  unsigned int               d_nbits;           // number of bits in header constellation point
  std::vector<gr_complex>    d_sym_position;    // constellation point position
  std::vector<unsigned char> d_sym_value_out;   // constellation point value

  unsigned int       d_byte_offset;
  unsigned int       d_partial_byte;

  std::vector<gr_complex>    d_dfe;

  unsigned char d_resid;
  unsigned int d_nresid;
  float d_phase;
  float d_freq;
  float d_phase_gain;
  float d_freq_gain;
  float d_eq_gain;

  gr_msg_queue_sptr d_bad_header_queue;

  bool header_ok();


protected:
  foimimo_ofdm_demapper(const std::vector<gr_complex> &sym_position,
      const std::vector<unsigned char> &sym_value_out,
      unsigned int occupied_carriers,
      unsigned int code_k, unsigned int code_n, int default_packetlen,
      float phase_gain, float freq_gain, gr_msg_queue_sptr bad_header_queue);

  void enter_search();
  void enter_have_sync();
  void enter_have_header();
  void enter_have_first_symbol();

  unsigned char slicer(const gr_complex x, float &out_min_dist);
  void demodulate_header(const gr_complex *in);
  bool set_sym_value_out(const std::vector<gr_complex> &sym_position,
                         const std::vector<unsigned char> &sym_value_out);

public:
  ~foimimo_ofdm_demapper();
  void forecast (int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

};
#endif
