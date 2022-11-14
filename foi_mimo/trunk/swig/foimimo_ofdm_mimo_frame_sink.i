/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 * 
 * This file is part of FOI-MIMO
 * 
 * FOI-MIMO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * FOI-MIMO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FOI-MIMO; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

GR_SWIG_BLOCK_MAGIC(foimimo,ofdm_mimo_frame_sink);

foimimo_ofdm_mimo_frame_sink_sptr 
foimimo_make_ofdm_mimo_frame_sink(const std::vector<gr_complex> &sym_position, 
			const std::vector<unsigned char> &sym_value_out,
			gr_msg_queue_sptr target_queue, unsigned int occupied_tones,
			float phase_gain=0.25, float freq_gain=0.25*0.25/4);

class foimimo_ofdm_mimo_frame_sink : public gr_sync_block
{
 protected:
  foimimo_ofdm_mimo_frame_sink(const std::vector<gr_complex> &sym_position, 
		     const std::vector<unsigned char> &sym_value_out,
		     gr_msg_queue_sptr target_queue, unsigned int occupied_tones,
		     float phase_gain, float freq_gain);

 public:
  ~foimimo_ofdm_mimo_frame_sink();
};
