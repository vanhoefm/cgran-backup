/* -*- c++ -*- */

%feature("autodoc", "1");		// generate python docstrings

%include "exception.i"
%import "gnuradio.i"			// the common stuff

%{
#include "gnuradio_swig_bug_workaround.h"	// mandatory bug fix
#include "gr_speex_encoder.h"
#include "gr_speex_decoder.h"
#include "gr_packet_drop.h"
#include <stdexcept>
%}

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(gr,speex_encoder);

gr_speex_encoder_sptr
gr_make_speex_encoder (int sampling_rate,int quality,int mode,int complexity,int vad_enabled);

class gr_speex_encoder : public gr_sync_decimator
{
   private:
	gr_speex_encoder(int sampling_rate,int quality,int mode,int complexity,int vad_enabled);

   public:
        ~gr_speex_encoder();
};

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(gr,speex_decoder);

gr_speex_decoder_sptr
gr_make_speex_decoder (int quality,int mode);

class gr_speex_decoder : public gr_sync_interpolator
{
 protected:
	gr_speex_decoder(int quality,int mode);

 public:
	~gr_speex_decoder ();
};
GR_SWIG_BLOCK_MAGIC(gr,packet_drop);

gr_packet_drop_sptr
gr_make_packet_drop (float drop_rate,int quality,int mode);

class gr_packet_drop : public gr_sync_interpolator
{
   private:
	gr_packet_drop(float drop_rate,int quality,int mode);

   public:
        ~gr_packet_drop();
};

