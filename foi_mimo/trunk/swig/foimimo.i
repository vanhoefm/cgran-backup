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

%include "gnuradio.i"			// the common stuff

%{
#include "foimimo_ofdm_mapper_bcv.h"
#include "foimimo_ofdm_alamouti_frame_acquisition.h"
#include "foimimo_ofdm_mimo_frame_sink.h"
#include "foimimo_ofdm_alamouti_tx_cc.h"
#include "foimimo_crc32_checker_sink.h"
#include "foimimo_crc32_inserter.h"
#include "foimimo_descrambler_bb.h"
#include "foimimo_scrambler_bb.h"
#include "foimimo_trellis_encoder_bb.h"
#include "foimimo_trellis_viterbi_b.h" 
#include "foimimo_ofdm_demapper.h"
#include "foimimo_trellis_metrics_f.h"
#include "foimimo_chunk_2_byte.h"
#include "foimimo_byte_2_chunk.h"
#include "foimimo_chunk_2_byte_skip_head.h"
#include "foimimo_ofdm_mapper_source.h"
#include "foimimo_ofdm_frame_acquisition.h"
%}

%include "foimimo_ofdm_mapper_bcv.i"
%include "foimimo_ofdm_alamouti_frame_acquisition.i"
%include "foimimo_ofdm_mimo_frame_sink.i"
%include "foimimo_ofdm_alamouti_tx_cc.i"
%include "foimimo_crc32_checker_sink.i"
%include "foimimo_crc32_inserter.i"
%include "foimimo_descrambler_bb.i"
%include "foimimo_scrambler_bb.i"
%include "foimimo_trellis_encoder_bb.i"
%include "foimimo_trellis_viterbi_b.i"
%include "foimimo_ofdm_demapper.i"
%include "foimimo_trellis_metrics_f.i"
%include "foimimo_chunk_2_byte.i"
%include "foimimo_byte_2_chunk.i"
%include "foimimo_chunk_2_byte_skip_head.i"
%include "foimimo_ofdm_mapper_source.i"
%include "foimimo_ofdm_frame_acquisition.i"
