/* -*- c++ -*- */
/*
 * Copyright 2011 Anton Blad.
 * 
 * This file is part of OpenRD
 * 
 * OpenRD is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * OpenRD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
/* -*- c++ -*- */

%module pr

%include "gnuradio.i"
%include "python/cstring.i"

%{
#include "stream_meta.h"
#include "modes.h"
#include "coding_type.h"
#include "framing_type.h"
#include "modulation.h"
#include "field_code_type.h"
#include "gsm_frame.h"
#include "pvec.h"
#include "pr_dpacket.h"
#include "pr_file_descriptor_sink.h"
#include "pr_insert_head.h"
#include "pr_pvec_to_stream.h"
#include "pr_stream_to_pvec.h"
#include "pr_pvec_concat.h"
#include "pr_pvec_extract.h"
#include "pr_analyzer_vb.h"
#include "pr_analyzer_none_vb.h"
#include "pr_analyzer_ber_vb.h"
#include "pr_ber_estimate_b.h"
#include "pr_block_coder_vbb.h"
#include "pr_block_coder_none_vbb.h"
#include "pr_block_decoder_vfb.h"
#include "pr_block_decoder_none_vfb.h"
#include "pr_block_merge_vff.h"
#include "pr_block_partition_vbb.h"
#include "pr_const_mapper_vbc.h"
#include "pr_constellation_decoder_cb.h"
#include "pr_constellation_softdecoder_vcf.h"
#include "pr_daf_logic_vbb.h"
#include "pr_data_sink.h"
#include "pr_data_sink_packet.h"
#include "pr_data_source.h"
#include "pr_data_source_zero.h"
#include "pr_data_source_counter.h"
#include "pr_data_source_packet.h"
#include "pr_deframer_vbb.h"
#include "pr_deframer_simple_vbb.h"
#include "pr_deframer_vcc.h"
#include "pr_deframer_none_vcc.h"
#include "pr_deframer_simple_vcc.h"
#include "pr_deframer_gsm_vcc.h"
#include "pr_frame_correlator_bb.h"
#include "pr_frame_correlator_none_bb.h"
#include "pr_frame_correlator_simple_bb.h"
#include "pr_frame_correlator_gsm_bb.h"
#include "pr_frame_sync_bb.h"
#include "pr_frame_sync_cc.h"
#include "pr_framer_vbb.h"
#include "pr_framer_none_vbb.h"
#include "pr_framer_simple_vbb.h"
#include "pr_framer_gsm_vbb.h"
#include "pr_mrc_vcc.h"
#include "pr_packet_sync_vcc.h"
#include "pr_rate_estimate.h"
#include "pr_snr_estimate_c.h"
%}

%cstring_input_binary(char* str, int len);
%cstring_output_allocate_size(char** str, int* len, $1 = $1);

%include <stream_meta.i>
%include <modes.h>
%include <coding_type.h>
%include <framing_type.h>
%include <modulation.i>
%include <field_code_type.h>
%include <gsm_frame.h>
%include <pvec.i>
%include <pr_dpacket.i>
%include <pr_file_descriptor_sink.i>
%include <pr_insert_head.i>
%include <pr_pvec_to_stream.i>
%include <pr_stream_to_pvec.i>
%include <pr_pvec_concat.i>
%include <pr_pvec_extract.i>
%include <pr_analyzer_vb.i>
%include <pr_analyzer_none_vb.i>
%include <pr_analyzer_ber_vb.i>
%include <pr_ber_estimate_b.i>
%include <pr_block_coder_vbb.i>
%include <pr_block_coder_none_vbb.i>
%include <pr_block_decoder_vfb.i>
%include <pr_block_decoder_none_vfb.i>
%include <pr_block_merge_vff.i>
%include <pr_block_partition_vbb.i>
%include <pr_const_mapper_vbc.i>
%include <pr_constellation_decoder_cb.i>
%include <pr_constellation_softdecoder_vcf.i>
%include <pr_daf_logic_vbb.i>
%include <pr_data_sink.i>
%include <pr_data_sink_packet.i>
%include <pr_data_source.i>
%include <pr_data_source_zero.i>
%include <pr_data_source_counter.i>
%include <pr_data_source_packet.i>
%include <pr_deframer_vbb.i>
%include <pr_deframer_simple_vbb.i>
%include <pr_deframer_vcc.i>
%include <pr_deframer_none_vcc.i>
%include <pr_deframer_simple_vcc.i>
%include <pr_deframer_gsm_vcc.i>
%include <pr_frame_correlator_bb.i>
%include <pr_frame_correlator_none_bb.i>
%include <pr_frame_correlator_simple_bb.i>
%include <pr_frame_correlator_gsm_bb.i>
%include <pr_frame_sync_bb.i>
%include <pr_frame_sync_cc.i>
%include <pr_framer_vbb.i>
%include <pr_framer_none_vbb.i>
%include <pr_framer_simple_vbb.i>
%include <pr_framer_gsm_vbb.i>
%include <pr_mrc_vcc.i>
%include <pr_packet_sync_vcc.i>
%include <pr_rate_estimate.i>
%include <pr_snr_estimate_c.i>

