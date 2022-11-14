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
#ifndef INCLUDED_PR_DATA_SINK_PACKET_H
#define INCLUDED_PR_DATA_SINK_PACKET_H

#include <pr_data_sink.h>

#include "pr_dpacket.h"
#include <queue>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>

enum packet_sink_mode { PACKET_SINK_BLOCK };

class pr_data_sink_packet;
typedef boost::shared_ptr<pr_data_sink_packet> pr_data_sink_packet_sptr;

/**
 * \brief Public constructor.
 *
 * \param num_proto Number of protocols to use (0 .. num_proto-1)
 * \param block_size Transport block size
 * \param mode Behaviour when recv_pkt is called (currently only blocking mode supported)
 */
pr_data_sink_packet_sptr pr_make_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode = PACKET_SINK_BLOCK);

/**
 * \brief Packet data sink
 *
 * \ingroup sigblk
 * This block receives data packetized using pr_data_source_packet. See 
 * \ref pr_data_source_packet and \ref pr_dpacket for details on transmission
 * of data.
 *
 * The received data blocks are stored in queues that can be read using the
 * recv_data() functions. In the default operation mode (PACKET_SINK_BLOCK),
 * these block if data is not available.
 *
 * Ports
 *  - Input 0: (<b>rxmeta</b>, <b>char</b>[block_size])
 *  - Output: None
 */
class pr_data_sink_packet : public pr_data_sink
{
private:
	friend pr_data_sink_packet_sptr pr_make_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode);

	pr_data_sink_packet(int num_proto, int block_size, packet_sink_mode mode);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~pr_data_sink_packet();

	/**
	 * Receives data using the specified protocol. This function is
	 * thread-safe.
	 *
	 * \param proto The protocol.
	 * \param len The length of the received data is stored here.
	 * \returns the received data.
	 */
	boost::shared_array<unsigned char> recv_data(int proto, int* len);

	/**
	 * Function exported to Python. The global interpreter lock is released 
	 * during the call, enabling other Python threads to run simultaneously.
	 *
	 * \param proto The protocol.
	 * \returns the received data as a Python list.
	 */
	std::vector<unsigned char> recv_list(int proto);

	/**
	 * Function exported to Python. The global interpreter lock is released 
	 * during the call, enabling other Python threads to run simultaneously.
	 *
	 * \param proto The protocol.
	 * \returns the received data as a Python string.
	 *
	 * The returned data is stored in a temporary string which is valid until
	 * the next call to the function.
	 */
	void recv_string(int proto, char** str, int* len);

protected:
	virtual void handle_packet(const unsigned char* data);

private:
	struct data_t
	{
		unsigned int length;
		boost::shared_array<unsigned char> data;
	};

	int d_num_proto;
	int d_block_size;
	packet_sink_mode d_mode;

	int d_packet_size;

	boost::scoped_array<std::queue<data_t> > d_queues;
	boost::mutex d_queue_lock; // Protects d_queues
	boost::scoped_array<boost::condition_variable> d_queue_changed; // Signaled when items are added to the corresponding queue

	boost::scoped_array<unsigned int> d_seqs;

	data_t d_current;
	int d_current_protocol;
	unsigned int d_current_window;
	boost::shared_array<unsigned char> d_string_holder;
};

#endif

