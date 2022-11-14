/* -*- c++ -*- */
/*
 * Copyright 2009 Institut fuer Nachrichtentechnik / Uni Karlsruhe
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

%feature("autodoc", "1");		// generate python docstrings

%include "exception.i"
%import "gnuradio.i"			// the common stuff

%{
#include "cs_nusaic_cc.h"
#include "cs_generic_vccf.h"
#include "cs_circmat_vccb.h"
#include "cs_toeplitz_vccb.h"
#include <stdexcept>
// FIXME Workaround: if I don't include these here, the SWIG code won't compile.
#include <gr_sync_decimator.h>
#include <gr_sync_interpolator.h>
%}

// ----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(cs,nusaic_cc);

cs_nusaic_cc_sptr cs_make_nusaic_cc (unsigned compression);

class cs_nusaic_cc : public gr_block
{
 private:
        cs_nusaic_cc(const unsigned compression);

 public:
        ~cs_nusaic_cc();

        float get_compression();
};

// ----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(cs,generic_vccf);

cs_generic_vccf_sptr
cs_make_generic_vccf (const std::vector<std::vector<float> > &comp_matrix);

class cs_generic_vccf : public gr_sync_block
{
 private:
        cs_generic_vccf(const std::vector<std::vector<float> > &comp_matrix);

 public:
        ~cs_generic_vccf();

        float get_compression();
};

// ----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(cs,toeplitz_vccb);

cs_toeplitz_vccb_sptr
cs_make_toeplitz_vccb (unsigned input_length, unsigned output_length, const std::vector<char> &seq_history)
        throw(std::invalid_argument);

class cs_toeplitz_vccb : public gr_sync_block
{
        friend cs_toeplitz_vccb_sptr cs_make_toeplitz_vccb (unsigned input_length,
                                                          unsigned output_length,
                                                          const std::vector<char> &seq_history);

 private:
        cs_toeplitz_vccb (unsigned input_length, unsigned output_length, std::vector<char> &seq_history);

 public:
        ~cs_toeplitz_vccb();

        float get_compression();
};

// ----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(cs,circmat_vccb);

cs_circmat_vccb_sptr
cs_make_circmat_vccb (const std::vector<char> &sequence, unsigned output_length, bool translate_zeros = false);

class cs_circmat_vccb : public gr_sync_block
{
        friend cs_circmat_vccb_sptr cs_make_circmat_vccb (const std::vector<char> &sequence,
                                unsigned output_length, bool translate_zeros);

 private:
        cs_circmat_vccb(const std::vector<char> &sequence, unsigned output_length, bool translate_zeros);

 public:
        ~cs_circmat_vccb();

        float get_compression();
};

