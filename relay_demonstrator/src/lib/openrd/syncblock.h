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
#ifndef INCLUDED_%LIBCLASS_H
#define INCLUDED_%LIBCLASS_H

#include <gr_sync_block.h>

class %lib_%class;

typedef boost::shared_ptr<%lib_%class> %lib_%class_sptr;

/**
 * \brief Public constructor.
 */
%lib_%class_sptr %lib_make_%class(%args);

/**
 * \brief 
 *
 * \ingroup sigblk
 *
 * Ports
 *  - Input 0:
 *  - Output 0:
 */
class %lib_%class : public gr_sync_block
{
private:
	friend %lib_%class_sptr %lib_make_%class(%args);

	%lib_%class(%args);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~%lib_%class();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

private:
};

#endif

