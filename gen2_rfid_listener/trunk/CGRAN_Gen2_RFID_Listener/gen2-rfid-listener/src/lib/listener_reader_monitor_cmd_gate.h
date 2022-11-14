/* -*- c++ -*- */
/* 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_LISTENER_READER_MONITOR_CMD_GATE_H
#define INCLUDED_LISTENER_READER_MONITOR_CMD_GATE_H

#include <gr_block.h>
#include <gr_message.h>
#include <gr_msg_queue.h>


class listener_reader_monitor_cmd_gate;
typedef boost::shared_ptr<listener_reader_monitor_cmd_gate> listener_reader_monitor_cmd_gate_sptr;


listener_reader_monitor_cmd_gate_sptr 
listener_make_reader_monitor_cmd_gate (float us_per_sample, bool real_time);

class listener_reader_monitor_cmd_gate : public gr_block 
{
 
 private:
  friend listener_reader_monitor_cmd_gate_sptr
  listener_make_reader_monitor_cmd_gate (float us_per_sample, bool real_time);
  
  double d_us_per_sample;
  int d_delim_width;          
  double d_thresh, mid_pt;
  double pwr_dwn_cnt;
  float cmd_len;
  bool pwrd_dwn;
  bool d_real_time;
  float d_reader_freq;
  bool frame_sync;
  int pwr_gate_cnt;
  
  int state;
  char d_bits[512];
  int d_len_bits;
    
  double pw, tari, rtcal, trcal, inter_arrival;
  
  listener_reader_monitor_cmd_gate(float us_per_sample, bool real_time);
  void forecast (int noutput_items, gr_vector_int &ninput_items_required); 
  int max_min(const float * buffer, int len, double * max, double * min, double* avg );
  int determine_symbol(int high_cnt, int low_cnt, char * bits, double max_RSSI, double avg_RSSI);
  int decode_command(char * bits, int len_bits, float inter_arrival, double max_RSSI, double avg_RSSI, int flag);
  

 public:
  ~listener_reader_monitor_cmd_gate();

  void set_reader_freq(float new_reader_freq);  
 
  gr_msg_queue_sptr log_q;

  gr_msg_queue_sptr get_log() const {return log_q;}

  int general_work(int noutput_items, 
		   gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);
  
  bool tag;  // True if tag response
  bool IS_TAG;
    

};

#endif

