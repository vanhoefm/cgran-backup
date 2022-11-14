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
#ifndef INCLUDED_PRBLK_ANALYZER_VB_H
#define INCLUDED_PRBLK_ANALYZER_VB_H

#include <gr_sync_block.h>

class txmeta;
class rxmeta;

class pr_analyzer_vb;

typedef boost::shared_ptr<pr_analyzer_vb> pr_analyzer_vb_sptr;

/**
 * \brief Base class for data analyzers
 *
 * \ingroup sigblk
 * This class provides common functionalities for the data analyzers. It has
 * two ports: one for the reference data and one for the received data. For
 * each vector pair, the analyze() function is called. analyze() must be
 * implemented in the subclasses.
 *
 * Ports
 *  - Input 0: (<b>txmeta</b>, <b>char</b>[block_size])
 *  - Input 1: (<b>rxmeta</b>, <b>char</b>[block_size])
 */
class pr_analyzer_vb : public gr_sync_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param block_size size of input
	 */
	pr_analyzer_vb(int block_size);

	/**
	 * Protected constructor.
	 *
	 * \param block_size size of input
	 * \param out output IO signature
	 */
	pr_analyzer_vb(int block_size, gr_io_signature_sptr out);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_analyzer_vb();

	/**
	 * \return the input block size
	 */
	int block_size() const;

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

protected:
	/**
	 * Called for each input. Must be implemented in subclasses to perform 
	 * desired analyses.
	 *
	 * \param refmeta reference meta data
	 * \param ref reference data
	 * \param recmeta received meta data
	 * \param rec received data
	 */
	virtual void analyze(const txmeta& refmeta, const char* ref, 
			const rxmeta& recmeta, const char* rec) = 0;

private:
	int d_block_size;

};

#endif
