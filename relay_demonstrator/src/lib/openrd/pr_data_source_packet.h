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
#ifndef INCLUDED_PR_DATA_SOURCE_PACKET_H
#define INCLUDED_PR_DATA_SOURCE_PACKET_H

#include "pr_data_source.h"

#include "pr_dpacket.h"
#include <queue>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>

enum packet_source_mode { PACKET_SOURCE_BLOCK };

class pr_data_source_packet;
typedef boost::shared_ptr<pr_data_source_packet> pr_data_source_packet_sptr;

/**
 * \brief Public constructor.
 *
 * \param num_proto Number of protocols to use (0 .. num_proto-1). The maximum
 * supported number of protocols is 127.
 * \param block_size Transport packet size
 * \param mode Behaviour when send_data is called (currently only blocking mode supported)
 */
pr_data_source_packet_sptr pr_make_data_source_packet(int num_proto, int block_size, packet_source_mode mode = PACKET_SOURCE_BLOCK);

/**
 * \brief Packet data source
 *
 * \ingroup sigblk
 * This block packetizes data for transmission using a number of protocols.
 * Each packet contains CRC, protocol, sequence number and window (see the
 * \ref pr_dpacket class), ensuring that each block of data is either 
 * received without errors, or not at all. There is (currently) no 
 * retransmission mechanism, so if lost data can not be tolerated,
 * retransmission must be handled on a higher level.
 *
 * Data is queued for transmission using the send_data() functions, and
 * transmitted at the next data request by the scheduler. If there is no data
 * enqueued, special null packets are transmitted. These are dropped silently
 * by the sink on the other end.
 *
 * Ports
 *  - Input: None
 *  - Output 0: (<b>txmeta</b>, <b>char</b>[block_size])
 */
class pr_data_source_packet : public pr_data_source
{
private:
	friend pr_data_source_packet_sptr pr_make_data_source_packet(int num_proto, int block_size, packet_source_mode mode);

	pr_data_source_packet(int num_proto, int block_size, packet_source_mode mode);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_data_source_packet();

	/**
	 * Transmits data using the specified protocol. This function is thread-
	 * safe.
	 *
	 * \param proto The protocol to use (0 .. \p num_proto-1)
	 * \param data Data to send.
	 * \param len Length of the data.
	 *
	 * Returns true if the data was enqueued successfully.
	 */
	bool send_data(int proto, const unsigned char* data, int len);

	/**
	 * Function exported to Python. The global interpreter lock is released 
	 * during the call, enabling other Python threads to run simultaneously.
	 *
	 * \param proto The protocol to use (0 .. \p num_proto-1)
	 * \param data Data to send. A Python list will work.
	 *
	 * Returns true if the data was enqueued successfully.
	 */
	bool send_list(int proto, const std::vector<unsigned char>& data);

	/**
	 * Function exported to Python. The global interpreter lock is released 
	 * during the call, enabling other Python threads to run simultaneously.
	 *
	 * \param proto The protocol to use (0 .. \p num_proto-1)
	 * \param data Data to send. A Python string will work.
	 *
	 * Returns true if the data was enqueued successfully.
	 */
	bool send_string(int proto, char* str, int len);

protected:
	virtual void fill_packet(unsigned char* data, unsigned char* valid);

private:
	int d_num_proto;
	int d_block_size;
	packet_source_mode d_mode;

	int d_packet_size;

	std::queue<pr_dpacket_sptr> d_queue;
	boost::mutex d_queue_lock; // Protects d_queue
	boost::condition_variable d_queue_avail; // Signaled when d_queue is not full

	boost::scoped_array<int> d_seqs;

	pr_dpacket_sptr d_null_packet;
};

#endif

