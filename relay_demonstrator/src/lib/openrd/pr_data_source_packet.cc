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

#include "pr_data_source_packet.h"

#include <gr_io_signature.h>
#include <iostream>
#include "stream_p.h"
#include "bitpack.h"
#include "openrd_debug.h"
#include "crc16.h"
#include <algorithm>
#include <boost/format.hpp>

using namespace std;
using namespace boost;

pr_data_source_packet_sptr pr_make_data_source_packet(int num_proto, int block_size, packet_source_mode mode)
{
	return pr_data_source_packet_sptr(new pr_data_source_packet(num_proto, block_size, mode));
}

pr_data_source_packet::pr_data_source_packet(int num_proto, int block_size, packet_source_mode mode) :
	pr_data_source(block_size),
	d_num_proto(num_proto),
	d_block_size(block_size),
	d_mode(mode),
	d_packet_size(block_size >> 3),
	d_seqs(new int[num_proto]),
	d_null_packet(pr_make_dpacket(d_packet_size))
{
	// Initialize packet sequence numbers
	for(int i = 0; i < d_num_proto; i++)
		d_seqs[i] = 0;

	// Initialize null packet structure
	d_null_packet->set_protocol(127);
	d_null_packet->set_initial_packet();
	d_null_packet->set_seq(0);
	d_null_packet->set_length(d_null_packet->data_size());
	fill(d_null_packet->data(), d_null_packet->data() + d_null_packet->data_size(), 0);
	d_null_packet->calculate_crc();
}

pr_data_source_packet::~pr_data_source_packet()
{
}

bool pr_data_source_packet::send_data(int proto, const unsigned char* data, int len)
{
	if(proto >= d_num_proto)
	{
		cerr << "pr_data_source_packet::send_data: invalid proto " << proto << endl;
		return false;
	}

	if(len > 65535)
	{
		cerr << format("pr_data_source_packet::send_data: len is too big (%d)") %
			len << endl;
		return false;
	}

	cerr << format("Sending packet (proto %d, len %d)") % proto % len << endl;

	unique_lock<mutex> lock(d_queue_lock);

	// If blocking mode, wait until queue is empty
	if(d_mode == PACKET_SOURCE_BLOCK)
	{
		while(!d_queue.empty())
			d_queue_avail.wait(lock);
	}

	int window = 0;
	int nbytes;

	// Add initial packet
	pr_dpacket_sptr initpkt(pr_make_dpacket(d_packet_size));
	initpkt->set_protocol(proto);
	initpkt->set_initial_packet();
	initpkt->set_seq(d_seqs[proto]);
	initpkt->set_length(len);
	nbytes = initpkt->data_size();
	if(nbytes > len)
		nbytes = len;
	copy(data, data+nbytes, initpkt->data());
	initpkt->calculate_crc();
	window += nbytes;
	d_queue.push(initpkt);
	
	cerr << format("init: proto=%d, seq=%d, len=%d") % (int)initpkt->protocol() %
		(int)initpkt->seq() % (int)initpkt->length() << endl;

	// Add data packets
	while(window < len)
	{
		pr_dpacket_sptr pkt(pr_make_dpacket(d_packet_size));
		pkt->set_protocol(proto);
		pkt->set_seq(d_seqs[proto]);
		pkt->set_window(window);
		nbytes = pkt->data_size();
		if(window + nbytes > len)
			nbytes = len - window;
		copy(data+window, data+window+nbytes, pkt->data());
		pkt->calculate_crc();
		window += nbytes;
		d_queue.push(pkt);

		cerr << format("data: proto=%d, seq=%d, win=%d") % (int)pkt->protocol() %
			(int)pkt->seq() % (int)pkt->window() << endl;
	}

	d_seqs[proto]++;

	return true;
}

bool pr_data_source_packet::send_list(int proto, const vector<unsigned char>& data)
{
	return send_data(proto, &data.front(), data.size());
}

bool pr_data_source_packet::send_string(int proto, char* str, int len)
{
	return send_data(proto, (unsigned char*) str, len);
}

void pr_data_source_packet::fill_packet(unsigned char* data, unsigned char* valid)
{
	pr_dpacket_sptr pkt;

	{
		unique_lock<mutex> lock(d_queue_lock);

		if(d_queue.empty())
		{
			// Queue is empty, send null packet
			pkt = d_null_packet;
			*valid = 0;
		}
		else
		{
			// Pop a packet from the queue and send it
			pkt = d_queue.front();
			d_queue.pop();
			*valid = 1;

			// If queue is empty, notify a waiter
			if(d_queue.empty())
			{
				lock.unlock();
				d_queue_avail.notify_one();
			}
		}
	}

	// Unpack the data
	bitpack_unpack(pkt->raw(), data, d_packet_size);
}

