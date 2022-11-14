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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pr_data_sink_packet.h"

#include <gr_io_signature.h>
#include "stream_p.h"
#include "boost/format.hpp"
#include <algorithm>
#include <iostream>
#include "openrd_debug.h"
#include "bitpack.h"
#include "crc16.h"

using namespace std;
using namespace boost;

pr_data_sink_packet_sptr pr_make_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode)
{
	return pr_data_sink_packet_sptr(new pr_data_sink_packet(num_proto, block_size, mode));
}

pr_data_sink_packet::pr_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode) :
	pr_data_sink(block_size),
	d_num_proto(num_proto),
	d_block_size(block_size),
	d_mode(mode),
	d_packet_size(block_size >> 3),
	d_queues(new std::queue<data_t>[num_proto]),
	d_queue_changed(new boost::condition_variable[num_proto]),
	d_seqs(new unsigned int[num_proto]),
	d_current_protocol(127),
	d_current_window(0)
{
	// Initialize packet sequence numbers.
	for(int i = 0; i < num_proto; i++)
		d_seqs[i] = 0;

	d_current.length = 0;
}

pr_data_sink_packet::~pr_data_sink_packet()
{
}

shared_array<unsigned char> pr_data_sink_packet::recv_data(int proto, int* len)
{
	if(proto >= d_num_proto)
	{
		cerr << format("pr_data_sink_packet::recv_data: Invalid protocol %d") % proto << endl;
		throw 0;
	}

	unique_lock<mutex> lock(d_queue_lock);

	// Wait until there are entries in the data queue
	while(d_queues[proto].empty())
	{
		d_queue_changed[proto].wait(lock);
	}

	// Read and pop data from the queue
	shared_array<unsigned char> data(d_queues[proto].front().data);
	*len = d_queues[proto].front().length;
	d_queues[proto].pop();

	return data;
}

vector<unsigned char> pr_data_sink_packet::recv_list(int proto)
{
	int len;
	shared_array<unsigned char> t(recv_data(proto, &len));
	vector<unsigned char> data(&t[0], &t[len]);
	return data;
}

void pr_data_sink_packet::recv_string(int proto, char** str, int* len)
{
	d_string_holder = recv_data(proto, len);
	*str = (char*) &d_string_holder[0];
}

void pr_data_sink_packet::handle_packet(const unsigned char* data)
{
	pr_dpacket pkt(d_packet_size);

	// Pack received bits to packet structure
	bitpack_pack(data, pkt.raw(), d_packet_size);

	// Check CRC of received packet
	if(!pkt.check_crc())
	{
		cerr << "pr_data_sink_packet: Received packet failed CRC, dropping." << endl;
		return;
	}

	if(pkt.protocol() == 127)
	{
		// Null packet, just drop it
		return;
	}

	if(pkt.protocol() >= d_num_proto)
	{
		cerr << format("pr_data_sink_packet: Invalid protocol %x received") % pkt.protocol() << endl;
		return;
	}

	if(pkt.initial_packet())
	{
		// Initial packet received
		if(d_current_window > 0)
		{
			cerr << format("pr_data_sink_packet: Received incomplete data with protocol %x, seq %d.") % 
				d_current_protocol % d_seqs[d_current_protocol] << endl;
		}

		// Initialize received data structures
		d_current_protocol = pkt.protocol();
		d_current_window = 0;
		d_current.length = pkt.length();
		d_current.data.reset(new unsigned char[d_current.length]);

		// Check sequence number of received data
		if(d_seqs[d_current_protocol] != pkt.seq())
		{
			cerr << format("pr_data_sink_packet: Received out-of-sequence packet with protocol %x, seq %d, expected seq %d.") %
				d_current_protocol % pkt.seq() % d_seqs[d_current_protocol] << endl;
			d_seqs[d_current_protocol] = pkt.seq();
		}
	}
	else
	{
		// Data packet received, do checks for protocol, sequence and window
		if(d_current_window == 0)
		{
			cerr << format("pr_data_sink_packet: Received spurious data packet, protocol %x, seq %d, window %d.") %
				(int)pkt.protocol() % pkt.seq() % pkt.window() << endl;
			return;
		}

		if(d_current_protocol != pkt.protocol())
		{
			cerr << format("pr_data_sink_packet: Received out-of-sequence packet, protocol %x (expected %x).") %
				(int)pkt.protocol() % d_current_protocol << endl;
			d_current_window = 0;
			return;
		}

		if(d_seqs[d_current_protocol] != pkt.seq())
		{
			cerr << format("pr_data_sink_packet: Received out-of-sequence packet, protocol %x, seq %d (expected %d).") %
				pkt.protocol() % pkt.seq() % d_seqs[d_current_protocol] << endl;
			d_current_window = 0;
			return;
		}

		if(d_current_window != pkt.window())
		{
			cerr << format("pr_data_sink_packet: Received out-of-sequence packet, protocol %x, seq %d, window %d (expected %d).") %
				pkt.protocol() % pkt.seq() % pkt.window() % d_current_window << endl;
			d_current_window = 0;
			return;
		}
	}

	// Copy data from received packet to buffer
	int len = min(pkt.data_size(), d_current.length - d_current_window);
	copy(pkt.data(), pkt.data()+len, &d_current.data[d_current_window]);
	d_current_window += len;

	// If all data has been received, store it in the read queue, signal 
	// listeners, and clear data structures
	if(d_current_window == d_current.length)
	{
		unique_lock<mutex> lock(d_queue_lock);

		d_queues[d_current_protocol].push(d_current);
		
		lock.unlock();
		d_queue_changed[d_current_protocol].notify_one();

		++d_seqs[d_current_protocol];
		d_current_window = 0;
		d_current_protocol = 127;
		d_current.length = 0;
		d_current.data.reset();
	}
}

