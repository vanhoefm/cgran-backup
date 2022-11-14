/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
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

// @WARNING@

GR_SWIG_BLOCK_MAGIC(gr,@BASE_NAME@)

@SPTR_NAME@ gr_make_@BASE_NAME@ (float threshold_factor_rise = 0.25,
				 float threshold_factor_fall = 0.40, 
				 int look_ahead = 10,
				 float alpha=0.001,
                 int carrier_num = 1);   // linklab, add carrier_num

class @NAME@ : public gr_sync_block
{
 private:
  @NAME@ (float threshold_factor_rise, 
	  float threshold_factor_fall,
	  int look_ahead, float alpha, int carrier_num);  // linklab, add carrier_num

 public:
  float d_sinr;      //linklab, sinr estimation
  float d_sig_power; //linklab, signal power estimation
  void set_threshold_factor_rise(float thr) { d_threshold_factor_rise = thr; }
  void set_threshold_factor_fall(float thr) { d_threshold_factor_fall = thr; }
  void set_look_ahead(int look) { d_look_ahead = look; }
  void set_alpha(int alpha) { d_avg_alpha = alpha; }

  float threshold_factor_rise() { return d_threshold_factor_rise; } 
  float threshold_factor_fall() { return d_threshold_factor_fall; }
  int look_ahead() { return d_look_ahead; }
  float alpha() { return d_avg_alpha; }
  // linklab, add a function to reset carrier)num
  void reset_carrier_map(int carrier_num){d_carrier_num = carrier_num; }
};
