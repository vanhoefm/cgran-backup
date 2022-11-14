/* -*- c++ -*- */

%include "gnuradio.i"			// the common stuff

%{
#include "dabp_subchannel_selector.h"
#include "dabp_time_deinterleaver.h"
#include "dabp_depuncturer.h"
#include "dabp_vitdec.h"
#include "dabp_scrambler.h"
#include "dabp_super_frame_sync.h"
#include "dabp_super_frame_rsdec.h"
#include "dabp_super_frame_sink.h"
#include "dabp_moving_sum_ff.h"
#include "dabp_ofdm_demod.h"
#include "dabp_fic_msc_demux.h"
#include "dabp_depuncturer_fic.h"
#include "dabp_fib_sink.h"
#include "dabp_parameters.h"
%}

GR_SWIG_BLOCK_MAGIC(dabp,subchannel_selector);

dabp_subchannel_selector_sptr dabp_make_subchannel_selector (int cifsz, int start_addr, int subchsz);

class dabp_subchannel_selector : public gr_block
{
private:
    dabp_subchannel_selector (int cifsz, int start_addr, int subchsz);
public:
    void reset(int start_addr, int subchsz);
};

GR_SWIG_BLOCK_MAGIC(dabp,time_deinterleaver);

dabp_time_deinterleaver_sptr dabp_make_time_deinterleaver (int subchsz);

class dabp_time_deinterleaver : public gr_block
{
private:
    dabp_time_deinterleaver (int subchsz);
public:
    void reset(int subchsz);
};

GR_SWIG_BLOCK_MAGIC(dabp,depuncturer);

dabp_depuncturer_sptr dabp_make_depuncturer (int subchsz, int optprot);

class dabp_depuncturer : public gr_block
{
private:
    dabp_depuncturer (int subchsz, int optprot);
public:
    int getI() const;
    void reset(int subchsz, int optprot);
};

GR_SWIG_BLOCK_MAGIC(dabp,vitdec);

dabp_vitdec_sptr dabp_make_vitdec (int I);

class dabp_vitdec : public gr_block
{
private:
    dabp_vitdec (int I);
public:
    void reset(int I);
};

GR_SWIG_BLOCK_MAGIC(dabp,scrambler);

dabp_scrambler_sptr dabp_make_scrambler (int I);

class dabp_scrambler : public gr_block
{
private:
    dabp_scrambler (int I);
public:
    int get_nbytes() const;
    void reset(int I);
};

GR_SWIG_BLOCK_MAGIC(dabp,super_frame_sync);

dabp_super_frame_sync_sptr dabp_make_super_frame_sync (int len_logfrm);

class dabp_super_frame_sync : public gr_block
{
private:
    dabp_super_frame_sync (int len_logfrm);
public:
    int get_subchidx() const;
    void reset(int len_logfrm);
};

GR_SWIG_BLOCK_MAGIC(dabp,super_frame_rsdec);

dabp_super_frame_rsdec_sptr dabp_make_super_frame_rsdec (int subchidx);

class dabp_super_frame_rsdec : public gr_block
{
private:
    dabp_super_frame_rsdec (int subchidx);
public:
    void reset(int subchidx);
};

GR_SWIG_BLOCK_MAGIC(dabp,super_frame_sink);

dabp_super_frame_sink_sptr dabp_make_super_frame_sink (int subchidx, const char *filename);
dabp_super_frame_sink_sptr dabp_make_super_frame_sink (int subchidx, int filedesc);

class dabp_super_frame_sink : public gr_sync_block
{
private:
    dabp_super_frame_sink (int subchidx, const char *filename);
public:
    void reset(int subchidx);
};

GR_SWIG_BLOCK_MAGIC(dabp,moving_sum_ff);

dabp_moving_sum_ff_sptr dabp_make_moving_sum_ff (int length, double alpha=0.9999);

class dabp_moving_sum_ff : public gr_sync_block
{
private:
  dabp_moving_sum_ff (int length, double alpha=0.9999);
};

GR_SWIG_BLOCK_MAGIC(dabp,ofdm_demod);

dabp_ofdm_demod_sptr dabp_make_ofdm_demod (int mode);

class dabp_ofdm_demod : public gr_block
{
private:
  dabp_ofdm_demod (int mode);
};

GR_SWIG_BLOCK_MAGIC(dabp,fic_msc_demux);

dabp_fic_msc_demux_sptr dabp_make_fic_msc_demux (int sel, int mode=1);

class dabp_fic_msc_demux : public gr_block
{
private:
  dabp_fic_msc_demux (int sel, int mode=1);
};

GR_SWIG_BLOCK_MAGIC(dabp,depuncturer_fic);

dabp_depuncturer_fic_sptr dabp_make_depuncturer_fic (int mode);

class dabp_depuncturer_fic : public gr_block
{
private:
  dabp_depuncturer_fic (int mode);
public:
  int getI() const;
};

GR_SWIG_BLOCK_MAGIC(dabp,fib_sink);

dabp_fib_sink_sptr dabp_make_fib_sink ();

class dabp_fib_sink : public gr_sync_block
{
private:
  dabp_fib_sink ();
public:
  void print_subch();
  void save_subch(const char *filename);
};

GR_SWIG_BLOCK_MAGIC(dabp,parameters);

dabp_parameters_sptr dabp_make_parameters (int mode);

class dabp_parameters
{
public:
  dabp_parameters (int mode);
public:
  float get_f() const;
  float get_fa() const;
  int get_mode() const;
  int get_L() const;
  int get_K() const;
  int get_Tu() const;
  int get_delta() const;
  int get_Tnull() const;
  int get_Tf() const;
  int get_Ts() const;
  int get_Nnull() const;
  int get_Nfft() const;
  int get_Ndel() const;
  int get_Nfrm() const;
  int get_Nficsyms() const;
  int get_Nmscsyms() const;
  int get_cifsz() const;
};
