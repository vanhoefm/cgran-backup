/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 * 
 * This file is part of FOI-MIMO
 * 
 * FOI-MIMO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * FOI-MIMO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FOI-MIMO; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
GR_SWIG_BLOCK_MAGIC(foimimo, crc32_checker_sink)

foimimo_crc32_checker_sink_sptr
foimimo_make_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue);

class foimimo_crc32_checker_sink: public gr_block
{
protected:
  foimimo_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue);

public:
    void set_pkt_size(unsigned int pkt_size);
};
