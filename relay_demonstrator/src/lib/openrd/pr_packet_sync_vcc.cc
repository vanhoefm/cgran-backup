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

#include "openrd_debug.h"

#include "pr_packet_sync_vcc.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"
#include "frametime.h"
#include <time.h>
#include <iostream>

using namespace std;

//#define PACKET_SYNC_DEBUG
//#define PACKET_SYNC_TDEBUG

static const int timeout_step = 100;
static const int timeout_advwait = 100;

pr_packet_sync_vcc_sptr pr_make_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout)
{
	return pr_packet_sync_vcc_sptr(new pr_packet_sync_vcc(frame_size, seqpolicy, max_timeout, timeout));
}

static const int MIN_IN = 2;
static const int MAX_IN = 2;
static const int MIN_OUT = 2;
static const int MAX_OUT = 2;

pr_packet_sync_vcc::pr_packet_sync_vcc(int frame_size, seqpolicy_type seqpolicy, unsigned int max_timeout, unsigned int timeout) :
	gr_block("packet_sync_vcc",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<rxframe,gr_complex>::alloc_size(frame_size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<rxframe,gr_complex>::alloc_size(frame_size))),
	d_frame_size(frame_size),
	d_seqpolicy(seqpolicy),
	d_max_timeout(max_timeout),
	d_timeout(timeout),
	d_has_advanced(false)
{
}

pr_packet_sync_vcc::~pr_packet_sync_vcc()
{
}

#ifdef PACKET_SYNC_DEBUG
static void print_debug_info(const_stream_p<const seqframe,const gr_complex>& in0,
		const_stream_p<const seqframe,const gr_complex>& in1)
{
	cerr << "stream0: ";
	if(in0.atend())
		cerr << "empty";
	else
		cerr << "(" << in0.meta().pkt_seq << "," << in0.meta().frame_seq << "," << in0.meta().stamp << ")";
	cerr << ", ";
	cerr << "stream1: ";
	if(in1.atend())
		cerr << "empty";
	else
		cerr << "(" << in1.meta().pkt_seq << "," << in1.meta().frame_seq << "," << in1.meta().stamp << ")";
	cerr << ": ";
}
#endif

int pr_packet_sync_vcc::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const rxframe,const gr_complex> in0(input_items[0], d_frame_size, ninput_items[0]);
	const_stream_p<const rxframe,const gr_complex> in1(input_items[1], d_frame_size, ninput_items[1]);
	stream_p<rxframe,gr_complex> out0(output_items[0], d_frame_size, noutput_items);
	stream_p<rxframe,gr_complex> out1(output_items[1], d_frame_size, noutput_items);
	const int len = d_frame_size*sizeof(gr_complex);

	// Loop as long as there are outputs available
	while(!out0.atend() && !out1.atend())
	{
		int stream0 = 0;
		int stream1 = 0;
		int nextout = 0;

		if(!in0.atend() && !in1.atend())
		{
			// Packets available on both streams

			const long long int seq0 = ((long long int)in0.meta().pkt_seq << 32) + in0.meta().frame_seq;
			const long long int seq1 = ((long long int)in1.meta().pkt_seq << 32) + in1.meta().frame_seq;
			long long int diff = seq0 - seq1;

			// If policy is to ignore sequence numbers, advance both streams
			if(d_seqpolicy == SEQPOLICY_IGNORE)
				diff = 0;

			// Advance the correct stream depending on the packet difference
			if(diff > 0)
			{
#ifdef PACKET_SYNC_DEBUG
				print_debug_info(in0, in1);
				cerr << "advance 1" << endl;
#endif
				d_has_advanced = true;

				stream0 = 2;
				stream1 = 1;
				nextout = 1;
			}
			else if(diff < 0)
			{
#ifdef PACKET_SYNC_DEBUG
				print_debug_info(in0, in1);
				cerr << "advance 0" << endl;
#endif
				d_has_advanced = true;

				stream0 = 1;
				stream1 = 2;
				nextout = 1;
			}
			else
			{
				// Streams are synchronized, get timeout directly from frames
				int tdiff = in1.meta().stamp >= in0.meta().stamp ? 
					in1.meta().stamp - in0.meta().stamp :
					in0.meta().stamp - in1.meta().stamp;

				update_timeout(2*tdiff);

#ifdef PACKET_SYNC_DEBUG
				print_debug_info(in0, in1);
				cerr << "pass through" << endl;
#endif
				d_has_advanced = false;

				stream0 = 1;
				stream1 = 1;
				nextout = 1;
			}
		}
		else if(in0.atend() && !in1.atend())
		{
			// If stream 1 has reached timeout, insert zero frame for stream 0
			if(frametime_now()-in1.meta().stamp > d_timeout)
			{
				if(d_has_advanced)
				{
					update_timeout(frametime_now()-in1.meta().stamp+timeout_advwait);
					d_has_advanced = false;
				}
#ifdef PACKET_SYNC_DEBUG
				print_debug_info(in0, in1);
				cerr << "inject in 0 " << frametime_now()-in1.meta().stamp << " " << d_timeout << endl;
#endif
				stream0 = 2;
				stream1 = 1;
				nextout = 1;
			}
		}
		else if(!in0.atend() && in1.atend())
		{
			// If stream 0 has reached timeout, insert zero frame for stream 1
			if(frametime_now()-in0.meta().stamp > d_timeout)
			{
				if(d_has_advanced)
				{
					update_timeout(frametime_now()-in0.meta().stamp+timeout_advwait);
					d_has_advanced = false;
				}
#ifdef PACKET_SYNC_DEBUG
				print_debug_info(in0, in1);
				cerr << "inject in 1 " << frametime_now()-in0.meta().stamp << " " << d_timeout << endl;
#endif
				stream0 = 1;
				stream1 = 2;
				nextout = 1;
			}
		}
		else
		{
			// No packet on either input, break
			break;
		}

		if(nextout == 0)
			break;

		if(stream0 == 1)
		{
			out0.meta().power = in0.meta().power;
			out0.meta().pkt_seq = in0.meta().pkt_seq;
			out0.meta().frame_seq = in0.meta().frame_seq;
			std::copy(in0.data(), in0.data() + d_frame_size, out0.data());
		}
		else if(stream0 == 2)
		{
			out0.meta().power = 0;
			out0.meta().pkt_seq = in1.meta().pkt_seq;
			out0.meta().frame_seq = in1.meta().frame_seq;
			std::memset(out0.data(), 0, len);
		}

		if(stream1 == 1)
		{
			out1.meta().power = in1.meta().power;
			out1.meta().pkt_seq = in1.meta().pkt_seq;
			out1.meta().frame_seq = in1.meta().frame_seq;
			std::copy(in1.data(), in1.data() + d_frame_size, out1.data());
		}
		else if(stream1 == 2)
		{
			out1.meta().power = 0;
			out1.meta().pkt_seq = in0.meta().pkt_seq;
			out1.meta().frame_seq = in0.meta().frame_seq;
			std::memset(out1.data(), 0, len);
		}

		if(stream0 == 1)
		{
			in0.next();
		}
		if(stream1 == 1)
		{
			in1.next();
		}

		// Advance the output pointers
		if(nextout == 1)
		{
			out0.next();
			out1.next();
		}
	}

	consume(0, in0.current());
	consume(1, in1.current());

	work_used(this, 0, in0.current());
	work_used(this, 1, in1.current());
	work_exit(this, out0.current());

	if(out0.current() == 0)
	{
		struct timespec tv;
		tv.tv_sec = 0;
		tv.tv_nsec = 10000000;
		nanosleep(&tv, 0);
	}

	return out0.current();
}

int pr_packet_sync_vcc::frame_size() const
{
	return d_frame_size;
}

void pr_packet_sync_vcc::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
	// Do something more sensible here..
	if(d_timeout == 0)
	{
		ninput_items_required[0] = 1;
		ninput_items_required[1] = 1;
	}
	else
	{
		ninput_items_required[0] = 0;
		ninput_items_required[1] = 0;
	}
}

void pr_packet_sync_vcc::update_timeout(unsigned int t)
{
	int d;

	if(t > d_max_timeout)
		t = d_max_timeout;

	d = (int)t-(int)d_timeout;

	if((d >= 0 ? d : -d) >= (int)timeout_step/2)
	{
		d_timeout = t;
#ifdef PACKET_SYNC_TDEBUG
		cerr << "packet_sync_vcc: Updating timeout to " << d_timeout << endl;
#endif
	}
}

