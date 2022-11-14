/* -*- c++ -*- */

%feature("autodoc", "1");		// generate python docstrings

%include "exception.i"
%import "gnuradio.i"			// the common stuff

%{
#include "gnuradio_swig_bug_workaround.h"	// mandatory bug fix
#include "tait_c4fm_detect_s.h"
//#include "tait_print_f.h"
//#include "tait_print_i.h"
#include "tait_biquad4_ss.h"
#include "tait_example_ff.h"
//#include "gr_float_to_int16_t.h"
#include "tait_DC_corrector_ss.h"
#include "tait_DC_corrector_ff.h"
#include "tait_socket_encode_fchar.h"
#include "tait_socket_encode_schar.h"
#include "tait_flat_rayleigh_channel_cc.h"
#include <stdexcept>
%}

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait,c4fm_detect_s);
		
tait_c4fm_detect_s_sptr tait_make_c4fm_detect_s (/*uint16_t averageCount*/);

class tait_c4fm_detect_s : public gr_sync_block
{
private:
	tait_c4fm_detect_s (/*uint16_t averageCount*/);

};

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait, biquad4_ss);
		
tait_biquad4_ss_sptr tait_make_biquad4_ss ();

class tait_biquad4_ss : public gr_sync_block
{
private:

	tait_biquad4_ss ();
};

// ----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(tait, example_ff);

tait_example_ff_sptr tait_make_example_ff ();

class tait_example_ff : public gr_block
{
	private:
		tait_example_ff ();
};

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait, DC_corrector_ss);
		
tait_DC_corrector_ss_sptr tait_make_DC_corrector_ss (int dc_offset_remove_const);

class tait_DC_corrector_ss : public gr_sync_block
{
	private:

		tait_DC_corrector_ss (int16_t dc_offset_remove_const);
};

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait, DC_corrector_ff);
		
tait_DC_corrector_ff_sptr tait_make_DC_corrector_ff (float dc_offset_remove_const);

class tait_DC_corrector_ff : public gr_sync_block
{
	private:

		tait_DC_corrector_ff (float dc_offset_remove_const);
};
// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait, socket_encode_fchar);
		
tait_socket_encode_fchar_sptr tait_make_socket_encode_fchar ();

class tait_socket_encode_fchar : public gr_sync_interpolator
{
	private:
		tait_socket_encode_fchar ();
};
// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait, socket_encode_schar);
		
tait_socket_encode_schar_sptr tait_make_socket_encode_schar ();

class tait_socket_encode_schar : public gr_sync_interpolator
{
	private:
		tait_socket_encode_schar ();
};
// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(tait, flat_rayleigh_channel_cc);
		
tait_flat_rayleigh_channel_cc_sptr tait_make_flat_rayleigh_channel_cc (int seeed, float fD, float pwr, bool flag_indep);

class tait_flat_rayleigh_channel_cc : public gr_sync_block
{
	private:

		tait_flat_rayleigh_channel_cc (int seeed, float fD, float pwr, bool flag_indep);
};

// ----------------------------------------------------------------
	
	
