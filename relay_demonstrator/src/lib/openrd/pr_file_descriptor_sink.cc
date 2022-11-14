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

#include "pr_file_descriptor_sink.h"
#include "openrd_debug.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cstdio>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
using namespace std;

#define MTU 32768

pr_file_descriptor_sink_sptr pr_make_file_descriptor_sink(size_t itemsize, int fd)
{
	return pr_file_descriptor_sink_sptr(new pr_file_descriptor_sink(itemsize, fd));
}

pr_file_descriptor_sink::pr_file_descriptor_sink(size_t itemsize, int fd) :
	gr_sync_block("file_descriptor_sink",
			 gr_make_io_signature(1, 1, sizeof(gr_complex)),
			 gr_make_io_signature(0, 0, 0)),
	d_itemsize(itemsize), d_fd(fd)
{
}

pr_file_descriptor_sink::~pr_file_descriptor_sink()
{
	close(d_fd);
}

int pr_file_descriptor_sink::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const char* in = (const char*) input_items[0];
	unsigned long byte_size = noutput_items*d_itemsize;

	while(byte_size > 0)
	{
		ssize_t r;

		r = write(d_fd, in, byte_size > MTU ? MTU : byte_size);
		if(r == -1)
		{
			if(errno == EINTR)
				continue;
			else
			{
				perror("pr_file_descriptor_sink");
				return -1;
			}
		}
		else
		{
			byte_size -= r;
			in += r;
		}
	}
	
	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

