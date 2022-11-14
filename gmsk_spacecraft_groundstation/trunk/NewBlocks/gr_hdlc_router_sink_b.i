/* -*- c++ -*- */
/*
 * Copyright 2008 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
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

GR_SWIG_BLOCK_MAGIC(gr,hdlc_router_sink_b);

gr_hdlc_router_sink_b_sptr gr_make_hdlc_router_sink_b (int dlci);

class gr_hdlc_router_sink_b : public gr_sync_block
{
 private:
  gr_hdlc_router_sink_b (int dlci);

 public:
  int dlci()        const { return d_dlci; }
  int flags()       const { return d_flag_cnt; }
  int frames()      const { return d_good_frame_cnt; }
  int bytes()       const { return d_good_byte_cnt; }
  int dlci_frames() const { return d_good_dlci_cnt; }
  int crc_errs()    const { return d_crc_err_cnt; }
  int aborts()      const { return d_abort_cnt; }
  int seven_ones()  const { return d_seven_ones_cnt; }
  int non_aligned() const { return d_non_aligned_cnt; }
  int giants()      const { return d_giant_cnt; }
  int runts()       const { return d_runt_cnt; }
  void clear_counters()   {
                            d_flag_cnt = 0;
                            d_good_frame_cnt = 0;
                            d_good_byte_cnt = 0;
                            d_good_dlci_cnt = 0;
                            d_crc_err_cnt = 0;
                            d_abort_cnt = 0;
                            d_seven_ones_cnt = 0;
                            d_non_aligned_cnt = 0;
                            d_giant_cnt = 0;
                            d_runt_cnt = 0;
                          }
};
