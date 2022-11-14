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
GR_SWIG_BLOCK_MAGIC(pr,data_sink_packet);

enum packet_sink_mode { PACKET_SINK_BLOCK };

pr_data_sink_packet_sptr pr_make_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode = PACKET_SINK_BLOCK);

%thread pr_data_source_packet::recv_string;
%thread pr_data_source_packet::recv_list;

class pr_data_sink_packet : public pr_data_sink
{
private:
    pr_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode);

public:
    void recv_string(int proto, char** str, int* len);
    std::vector<unsigned char> recv_list(int proto);
};

