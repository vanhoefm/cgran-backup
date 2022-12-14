/* -*- c++ -*- */

%feature("autodoc", "1");		// generate python docstrings

%include "exception.i"
%{
#include <cstddef>
%}
%import "gnuradio.i"			// the common stuff

%{
#include "gnuradio_swig_bug_workaround.h"	// mandatory bug fix
#include "gr_rds_bpsk_demod.h"
#include "gr_rds_data_decoder.h"
#include "gr_rds_data_encoder.h"
#include "gr_rds_rate_enforcer.h"
#include "gr_rds_freq_divider.h"
#include <stdexcept>
%}

//------------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC (gr_rds, data_decoder);

gr_rds_data_decoder_sptr gr_rds_make_data_decoder(gr_msg_queue_sptr msgq);

class gr_rds_data_decoder: public gr_sync_block
{
private:
	gr_rds_data_decoder(gr_msg_queue_sptr msgq);
public:
	void reset(int x);
};

//------------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC (gr_rds, data_encoder);

gr_rds_data_encoder_sptr gr_rds_make_data_encoder(const char *xmlfile);

class gr_rds_data_encoder: public gr_sync_block
{
private:
	gr_rds_data_encoder(const char *xmlfile);
};

//------------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC (gr_rds, rate_enforcer);

gr_rds_rate_enforcer_sptr gr_rds_make_rate_enforcer(double samp_rate);

class gr_rds_rate_enforcer: public gr_block
{
private:
	gr_rds_rate_enforcer(double samp_rate);
};

// ------------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC (gr_rds, freq_divider);

gr_rds_freq_divider_sptr gr_rds_make_freq_divider (unsigned int divider);

class gr_rds_freq_divider: public gr_sync_block
{
private:
	gr_rds_freq_divider (unsigned int divider);
};

// -----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC (gr_rds, bpsk_demod);

gr_rds_bpsk_demod_sptr gr_rds_make_bpsk_demod (double sampling_rate);

class gr_rds_bpsk_demod: public gr_block
{
private:
	gr_rds_bpsk_demod (double sampling_rate);
public:
	void reset(int r);
};
