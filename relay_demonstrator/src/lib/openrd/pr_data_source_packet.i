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
GR_SWIG_BLOCK_MAGIC(pr,data_source_packet);

enum packet_source_mode { PACKET_SOURCE_BLOCK };

pr_data_source_packet_sptr pr_make_data_source_packet(int num_proto, int block_size, packet_source_mode mode = PACKET_SOURCE_BLOCK);

%thread pr_data_source_packet::send_string;
%thread pr_data_source_packet::send_list;

class pr_data_source_packet : public pr_data_source
{
private:
    pr_data_source_packet(int num_proto, int block_size, packet_source_mode mode);

public:
    bool send_string(int proto, char* str, int len);
    bool send_list(int proto, const std::vector<unsigned char>& data);
};

