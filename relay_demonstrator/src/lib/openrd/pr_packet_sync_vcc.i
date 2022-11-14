/* -*- c++ -*- */
/*
 * Copyright 2011 Anton Blad.
 * 
 * This file is part of OpenRD
 * 
 * OpenRD is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * OpenRD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
GR_SWIG_BLOCK_MAGIC(pr,packet_sync_vcc);

enum seqpolicy_type { SEQPOLICY_IGNORE, SEQPOLICY_SYNC, SEQPOLICY_SYNCINSERT };

pr_packet_sync_vcc_sptr pr_make_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout = 0);

class pr_packet_sync_vcc : public gr_block
{
private:
    pr_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout);
    int frame_size() const;
};

