/* -*- c++ -*- */
 
%include "exception.i"
%import "gnuradio.i"                           
 
%{
#include "gnuradio_swig_bug_workaround.h"       
#include "listener_clock_recovery.h"
#include "listener_tag_monitor.h"
#include "listener_reader_monitor_cmd_gate.h"
#include "listener_find_CW.h"
#include "listener_to_mag_mux.h"
#include <stdexcept>
%}
  

//-----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(listener, clock_recovery);

listener_clock_recovery_sptr 
listener_make_clock_recovery(int samples_per_pulse, int interp_factor);

class listener_clock_recovery: public gr_block{
  listener_clock_recovery(int samples_per_pulse, int interp_factor);

public:
  ~listener_clock_recovery();
  
};


//-----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(listener, tag_monitor);

listener_tag_monitor_sptr
listener_make_tag_monitor (bool real_time, int miller);


class listener_tag_monitor: public gr_block{
 
  listener_tag_monitor (bool real_time, int miller);

public: 
  ~listener_tag_monitor();
  state STATE;
  gr_msg_queue_sptr get_log() const;

};



//-----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(listener, reader_monitor_cmd_gate);

listener_reader_monitor_cmd_gate_sptr 
listener_make_reader_monitor_cmd_gate (float us_per_sample, bool real_time);


class listener_reader_monitor_cmd_gate: public gr_block{
 
  listener_reader_monitor_cmd_gate (float us_per_sample, bool real_time);

public: 
  ~listener_reader_monitor_cmd_gate();
  gr_msg_queue_sptr get_log() const;
  void set_reader_freq(float new_reader_freq);  
 
  bool tag;  
  bool IS_TAG;
  
};



//-----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(listener, find_CW);

listener_find_CW_sptr 
listener_make_find_CW (unsigned int vlen, float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor);

class listener_find_CW: public gr_block{
 
  listener_find_CW (unsigned int vlen, float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor);

public: 
  ~listener_find_CW();
    
};


//-----------------------------------------------------------------
GR_SWIG_BLOCK_MAGIC(listener, to_mag_mux);

listener_to_mag_mux_sptr
listener_make_to_mag_mux ();


class listener_to_mag_mux: public gr_block{
 
  listener_to_mag_mux ();

public: 
  ~listener_to_mag_mux();    
  
};

