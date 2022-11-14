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

GR_SWIG_BLOCK_MAGIC(gr,hdlc_router_source_b);

gr_hdlc_router_source_b_sptr gr_make_hdlc_router_source_b (int dlci,
                                                           char* local_addr,
                                                           char* remote_addr);

class gr_hdlc_router_source_b : public gr_sync_block
{
 private:
  gr_hdlc_router_source_b (int   dlci,
                           char* local_addr,
                           char* remote_addr);

 public:
  int dlci()   const    { return d_dlci; }
  int flags()  const    { return d_flag_cnt; }
  int frames() const    { return d_frame_cnt; }
  int bytes()  const    { return d_byte_cnt; }
  void clear_counters() { d_flag_cnt = 0; d_frame_cnt = 0; d_byte_cnt = 0; }
};
