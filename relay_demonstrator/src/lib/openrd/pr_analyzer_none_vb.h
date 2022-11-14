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
#ifndef INCLUDED_PRBLK_ANALYZER_NONE_VB_H
#define INCLUDED_PRBLK_ANALYZER_NONE_VB_H

#include "pr_analyzer_vb.h"

#include "stream_p.h"

class pr_analyzer_none_vb;

typedef boost::shared_ptr<pr_analyzer_none_vb> pr_analyzer_none_vb_sptr;

/**
 * Public constructor.
 *
 * \param block_size size of input vectors
 */
pr_analyzer_none_vb_sptr pr_make_analyzer_none_vb(int block_size);

/**
 * \brief Dummy analyzer for data blocks
 *
 * \ingroup sigblk
 * This block accepts two vectors as inputs. It is used as a dummy analyzer.
 *
 * Ports
 *  - Input 0: (<b>txmeta</b>, <b>char</b>[block_size])
 *  - Input 1: (<b>rxmeta</b>, <b>char</b>[block_size])
 */
class pr_analyzer_none_vb : public pr_analyzer_vb
{
private:
	friend pr_analyzer_none_vb_sptr pr_make_analyzer_none_vb(int block_size);

	pr_analyzer_none_vb(int block_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_analyzer_none_vb();

protected:
	virtual void analyze(const txmeta& refmeta, const char* ref,
			const rxmeta& recmeta, const char* rec);
};

#endif

