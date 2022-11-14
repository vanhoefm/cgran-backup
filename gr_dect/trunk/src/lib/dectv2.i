/* -*- c++ -*- */
/*
 * Copyright 2004,2005,2006,2007 Free Software Foundation, Inc.
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
%feature("autodoc", "1");		// generate python docstrings

%include "exception.i"
%import "gnuradio.i"			// the common stuff
%{
#include "gnuradio_swig_bug_workaround.h"	// mandatory bug fix
#include "gr_dect_framer_sink.h"
#include "gr_dect_correlate_access_code.h"
#include "gr_dect_crc_r.h"


#include <stdexcept>
%}

%rename(crc_r) gr_crc_r;

unsigned int gr_crc_r(const std::string buf);



GR_SWIG_BLOCK_MAGIC(dectv1,framer_sink_dect);

dectv1_framer_sink_dect_sptr 
dectv1_make_framer_sink_dect(gr_msg_queue_sptr target_queue);

class dectv1_framer_sink_dect : public gr_sync_block
{
 protected:
  dectv1_framer_sink_dect(gr_msg_queue_sptr target_queue);

 public:
  ~dectv1_framer_sink_dect();
};

GR_SWIG_BLOCK_MAGIC(dectv1,correlate_access_code_dect);

/*!
 * \param access_code is represented with 1 byte per bit, e.g., "010101010111000100"
 * \param threshold maximum number of bits that may be wrong
 */
dectv1_correlate_access_code_dect_sptr 
dectv1_make_correlate_access_code_dect (const std::string &access_code, int threshold) 
  throw(std::out_of_range);

/*!
 * \brief Examine input for specified access code, one bit at a time.
 * \ingroup block
 *
 * input:  stream of bits, 1 bit per input byte (data in LSB)
 * output: stream of bits, 2 bits per output byte (data in LSB, flag in next higher bit)
 *
 * Each output byte contains two valid bits, the data bit, and the
 * flag bit.  The LSB (bit 0) is the data bit, and is the original
 * input data, delayed 64 bits.  Bit 1 is the
 * flag bit and is 1 if the corresponding data bit is the first data
 * bit following the access code. Otherwise the flag bit is 0.
 */
class dectv1_correlate_access_code_dect : public gr_sync_block
{
  friend dectv1_correlate_access_code_dect_sptr 
  dectv1_make_correlate_access_code_dect (const std::string &access_code, int threshold);
 protected:
  dectv1_correlate_access_code_dect(const std::string &access_code, int threshold);

 public:
  ~dectv1_correlate_access_code_dect();

  /*!
   * \param access_code is represented with 1 byte per bit, e.g., "010101010111000100"
   */
  bool set_access_code (const std::string &access_code);
};


