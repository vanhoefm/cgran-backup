include $(top_srcdir)/Makefile.common

# These headers get installed in ${prefix}/include/openrd
grinclude_HEADERS = \
	openrd_debug.h \
	frametime.h \
	modes.h \
	coding_type.h \
	coding.h \
	framing_type.h \
	framing.h \
	modulation_type.h \
	modulation.h \
	field_code_type.h \
	field_coder.h \
	gsm_frame.h \
	golay.h \
	rm.h \
	rm_2_6.h \
	pvec.h \
	snr.h \
	stream_p.h \
	bitpack.h \
	crc16.h \
	pr_dpacket.h \
	pr_file_descriptor_sink.h \
	pr_insert_head.h \
	pr_pvec_to_stream.h \
	pr_stream_to_pvec.h \
	pr_pvec_concat.h \
	pr_pvec_extract.h \
	pr_analyzer_vb.h \
	pr_analyzer_none_vb.h \
	pr_analyzer_ber_vb.h \
	pr_ber_estimate_b.h \
	pr_block_coder_vbb.h \
	pr_block_coder_none_vbb.h \
	pr_block_decoder_vfb.h \
	pr_block_decoder_none_vfb.h \
	pr_block_merge_vff.h \
	pr_block_partition_vbb.h \
	pr_const_mapper_vbc.h \
	pr_constellation_decoder_cb.h \
	pr_constellation_softdecoder_vcf.h \
	pr_daf_logic_vbb.h \
	pr_data_sink.h \
	pr_data_sink_packet.h \
	pr_data_source.h \
	pr_data_source_zero.h \
	pr_data_source_counter.h \
	pr_data_source_packet.h \
	pr_deframer_vbb.h \
	pr_deframer_simple_vbb.h \
	pr_deframer_vcc.h \
	pr_deframer_none_vcc.h \
	pr_deframer_simple_vcc.h \
	pr_deframer_gsm_vcc.h \
	pr_frame_correlator_bb.h \
	pr_frame_correlator_none_bb.h \
	pr_frame_correlator_simple_bb.h \
	pr_frame_correlator_gsm_bb.h \
	pr_frame_sync_bb.h \
	pr_frame_sync_cc.h \
	pr_framer_vbb.h \
	pr_framer_none_vbb.h \
	pr_framer_simple_vbb.h \
	pr_framer_gsm_vbb.h \
	pr_mrc_vcc.h \
	pr_packet_sync_vcc.h \
	pr_rate_estimate.h \
	pr_snr_estimate_c.h

TOP_SWIG_IFILES = pr.i

pr_pythondir_category = openrd
pr_la_swig_sources = \
	openrd_debug.cc \
	frametime.cc \
	modulation.cc \
	field_coder.cc \
	golay.cc \
	golaytable.cc \
	rm.cc \
	rm_2_6.cc \
	pvec.cc \
	snr.cc \
	bitpack.cc \
	crc16.cc \
	pr_dpacket.cc \
	pr_file_descriptor_sink.cc \
	pr_insert_head.cc \
	pr_pvec_to_stream.cc \
	pr_stream_to_pvec.cc \
	pr_pvec_concat.cc \
	pr_pvec_extract.cc \
	pr_analyzer_vb.cc \
	pr_analyzer_none_vb.cc \
	pr_analyzer_ber_vb.cc \
	pr_ber_estimate_b.cc \
	pr_block_coder_vbb.cc \
	pr_block_coder_none_vbb.cc \
	pr_block_decoder_vfb.cc \
	pr_block_decoder_none_vfb.cc \
	pr_block_merge_vff.cc \
	pr_block_partition_vbb.cc \
	pr_const_mapper_vbc.cc \
	pr_constellation_decoder_cb.cc \
	pr_constellation_softdecoder_vcf.cc \
	pr_daf_logic_vbb.cc \
	pr_data_sink.cc \
	pr_data_sink_packet.cc \
	pr_data_source.cc \
	pr_data_source_zero.cc \
	pr_data_source_counter.cc \
	pr_data_source_packet.cc \
	pr_deframer_vbb.cc \
	pr_deframer_simple_vbb.cc \
	pr_deframer_vcc.cc \
	pr_deframer_none_vcc.cc \
	pr_deframer_simple_vcc.cc \
	pr_deframer_gsm_vcc.cc \
	pr_frame_correlator_bb.cc \
	pr_frame_correlator_none_bb.cc \
	pr_frame_correlator_simple_bb.cc \
	pr_frame_correlator_gsm_bb.cc \
	pr_frame_sync_bb.cc \
	pr_frame_sync_cc.cc \
	pr_framer_vbb.cc \
	pr_framer_none_vbb.cc \
	pr_framer_simple_vbb.cc \
	pr_framer_gsm_vbb.cc \
	pr_mrc_vcc.cc \
	pr_packet_sync_vcc.cc \
	pr_rate_estimate.cc \
	pr_snr_estimate_c.cc

include $(top_srcdir)/Makefile.swig

BUILT_SOURCES = $(swig_built_sources)

no_dist_files = $(swig_built_sources)

SUBDIRS = 

