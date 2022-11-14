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
#ifndef INCLUDED_PRBLK_FILE_DESCRIPTOR_SINK_H
#define INCLUDED_PRBLK_FILE_DESCRIPTOR_SINK_H

#include <gr_sync_block.h>

class pr_file_descriptor_sink;

typedef boost::shared_ptr<pr_file_descriptor_sink> pr_file_descriptor_sink_sptr;

/**
 * \brief Public constructor.
 *
 * \param itemsize Size of input items.
 * \param fd File descriptor to write to.
 */
pr_file_descriptor_sink_sptr pr_make_file_descriptor_sink(size_t itemsize, int fd);

/**
 * \brief Writes a stream to an open file descriptor.
 *
 * \ingroup primblk
 * This class is included since the GNU Radio file descriptor sink is buggy
 * and can not be used together with sockets.
 *
 * The specified file descriptor \p fd must be open and writable, and can be
 * of any type. It does not have to be readable. No data alignment assumptions
 * can be made on the receiving side.
 *
 * Ports
 *  - Input 0: <b>char</b>[itemsize]
 */
class pr_file_descriptor_sink : public gr_sync_block
{
private:
	friend pr_file_descriptor_sink_sptr pr_make_file_descriptor_sink(size_t itemsize, int fd);

	pr_file_descriptor_sink(size_t itemsize, int fd);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~pr_file_descriptor_sink();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

private:
	size_t d_itemsize;
	int d_fd;
};

#endif

