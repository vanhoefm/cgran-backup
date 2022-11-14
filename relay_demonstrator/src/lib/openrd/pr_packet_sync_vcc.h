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
#ifndef INCLUDED_PRBLK_PACKET_SYNC_VCC_H
#define INCLUDED_PRBLK_PACKET_SYNC_VCC_H

#include <gr_block.h>
#include <gr_complex.h>

enum seqpolicy_type { SEQPOLICY_IGNORE, SEQPOLICY_SYNC, SEQPOLICY_SYNCINSERT };

class pr_packet_sync_vcc;

typedef boost::shared_ptr<pr_packet_sync_vcc> pr_packet_sync_vcc_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Number of complex symbols in input vectors.
 * \param seqpolicy Determines the synchronization policy. 
 * \param max_timeout Sets the maximum timeout (in ms).
 * \param timeout sets the initial timeout (in ms).
 */
pr_packet_sync_vcc_sptr pr_make_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout = 0);

/**
 * \brief Synchronizes two streams on the frame level, using the pkt_seq
 * and frame_seq fields of the \ref rxframe headers.
 *
 * \ingroup sigblk
 * The block has two inputs and two outputs, and the behaviour is specified 
 * by the \p seqpolicy argument:
 *  
 *  - SEQPOLICY_IGNORE: Ignore sequence numbers in frames. Just feed through
 *  packets on the inputs to the outputs, adding null frames if a stream
 *  reaches timeout.
 *  - SEQPOLICY_SYNC: Synchronize the sequence fields of the frames, such
 *  that the output frames always have the same sequence numbers. If the
 *  input sequence fields do not match, substitute null frames until they
 *  match.
 *
 * If one of the inputs is stalled, the \p timeout specified how long to
 * wait until the frame should be considered lost and having a null frame
 * substituted for it. The timeout can be specified initially, and is
 * updated continually to twice the normal delay between the streams
 * (computed as the time difference of packets with the same sequence
 * numbers). The timeout is updated to at most \p max_timeout. For details,
 * read the code.
 *
 * To determine the arrival time of the frames, the \p stamp fields in the
 * \ref rxframe header are used.
 */
class pr_packet_sync_vcc : public gr_block
{
private:
	friend pr_packet_sync_vcc_sptr pr_make_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout);

	pr_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_packet_sync_vcc();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);
	
	/**
	 * \return the number of symbols per frame
	 */
	int frame_size() const;

	virtual void forecast(int noutput_items, gr_vector_int& ninput_items_required);

private:
	int d_frame_size;
	seqpolicy_type d_seqpolicy;
	unsigned int d_max_timeout;
	unsigned int d_timeout;
	bool d_has_advanced;

	void update_timeout(unsigned int t);
};

#endif

