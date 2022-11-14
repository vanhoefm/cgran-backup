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

%include "gnuradio.i"

%{
#include "specest_welch.h"
%}


// ----------------------------------------------------------------
class specest_welch;
typedef boost::shared_ptr<specest_welch> specest_welch_sptr;
%template(specest_welch_sptr) boost::shared_ptr<specest_welch>;
%rename(welch) specest_make_welch;
%inline {
  gr_hier_block2_sptr specest_welch_block (specest_welch_sptr r)
  {
    return gr_hier_block2_sptr(r);
  }
}

%pythoncode %{
specest_welch_sptr.block = lambda self: specest_welch_block(self)
specest_welch_sptr.__repr__ = lambda self: "<gr_hier_block2 %s (%d)>" % (self.name(), self.unique_id ())
%}

%ignore specest_welch;


specest_welch_sptr
specest_make_welch(unsigned fft_len, int overlap, int ma_len, bool fft_shift, const std::vector<float> &window)
        throw(std::invalid_argument);

specest_welch_sptr
specest_make_welch(unsigned fft_len, int overlap = -1, int ma_len = 8, bool fft_shift = false)
        throw(std::invalid_argument);

class specest_welch : public gr_hier_block2
{
 private:
	specest_welch(unsigned fft_len, int overlap, int ma_len, bool fft_shift, const std::vector<float> &window);

 public:
	bool set_window(const std::vector<float> &window);
	bool set_hamming();
};

