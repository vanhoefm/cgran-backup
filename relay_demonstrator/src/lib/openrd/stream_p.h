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
#ifndef INCLUDED_STREAM_P_H
#define INCLUDED_STREAM_P_H

#include <gr_complex.h>
#include "pvec.h"
#include <cstring>
#include "stream_meta.h"

/**
 * \brief Denotes data without meta data
 * \ingroup sigmeta
 */
struct nullmeta { };

/* Template meta-programming used to determine size of meta data structure
 * at run-time. */
template<typename T> struct metasize
{
	enum { RET = sizeof(T) };
};

/* Template specialization for nullmeta is required, as empty structures 
 * have a non-zero size. */
template<> struct metasize<nullmeta>
{
	enum { RET = 0 };
};

template<> struct metasize<const nullmeta>
{
	enum { RET = 0 };
};

/**
 * \brief Base class for const and mutable stream pointers. 
 * \ingroup prim
 *
 * \tparam meta_type type of meta data, use \ref nullmeta if the stream does
 * not contain any meta data.
 * \tparam data_type type of data items
 * \tparam init_type type used for internal representation of the stream
 * \tparam arith_type type used for performing pointer arithmetic
 */
template<typename meta_type, typename data_type, typename init_type, typename arith_type> class base_stream_p
{
protected:
	/**
	 * \brief Protected constructor.
	 *
	 * \param stream pointer to stream
	 * \param data_len number of data items in each stream element
	 * \param num_items number of elements in the stream
	 */
	base_stream_p<meta_type,data_type,init_type,arith_type>(init_type* stream, int data_len, int num_items) :
		d_stream(stream),
		d_data_len(data_len),
		d_num_items(num_items),
		d_current(0),
		d_metasize(metasize<meta_type>::RET),
		d_used(d_metasize + data_len*sizeof(data_type)),
		d_vector_size(pvec_alloc_size(d_used)) { }

public:
	/**
	 * \brief Public destructor.
	 */
	~base_stream_p<meta_type,data_type,init_type,arith_type>() { }

	/**
	 * \returns the number of data items in each stream element
	 */
	int data_len() const { return d_data_len; }
	/**
	 * \returns the number of elements in the stream
	 */
	int num_items() const { return d_num_items; }

	/**
	 * \returns a reference to the meta data of the current stream element
	 */
	meta_type& meta() const { return *static_cast<meta_type*>(d_stream); }
	/**
	 * \returns a pointer to the data of the current stream element
	 */
	data_type* data() const { return static_cast<data_type*>((init_type*)((arith_type*)d_stream + d_metasize)); }

	/**
	 * Advances to the next stream element
	 */
	void next() { d_stream = (init_type*)((arith_type*)d_stream + d_vector_size); ++d_current; }

	/**
	 * \returns the index of the current stream element
	 */
	int current() const { return d_current; }
	/**
	 * \returns true if the stream is at end (past the last item)
	 */
	bool atend() const { return d_current == d_num_items; }

	/**
	 * \brief Determines element allocation size.
	 *
	 * \param data_len number of data items
	 * \returns stream element allocation size
	 */
	static int alloc_size(int data_len) { return pvec_alloc_size(metasize<meta_type>::RET+data_len*sizeof(data_type)); }

protected:
	init_type* d_stream;
	const int d_data_len;
	const int d_num_items;
	int d_current;
	const int d_metasize;
	const int d_used;
	const int d_vector_size;
};

/**
 * \brief Stream pointer for non-mutable (input) streams
 *
 * \ingroup prim
 * \tparam meta_type type of meta data, use \ref nullmeta if the stream does
 * not contain any meta data.
 * \tparam data_type type of data items
 */
template<typename meta_type, typename data_type> class const_stream_p : 
	public base_stream_p<meta_type,data_type,const void,const char>
{
public:
	/**
	 * \brief Public constructor.
	 *
	 * \param stream pointer to stream
	 * \param data_len number of data items in each stream element
	 * \param num_items number of elements in the stream
	 */
	const_stream_p<meta_type,data_type>(const void* stream, int data_len, int num_items) :
		base_stream_p<meta_type,data_type,const void,const char>(stream, data_len, num_items) { }
};

/**
 * \brief Stream pointer for mutable (output) streams
 *
 * \ingroup prim
 * \tparam meta_type type of meta data, use \ref nullmeta if the stream does
 * not contain any meta data.
 * \tparam data_type type of data items
 */
template<typename meta_type, typename data_type> class stream_p :
	public base_stream_p<meta_type,data_type,void,char>
{
public:
	/**
	 * \brief Public constructor.
	 *
	 * \param stream pointer to stream
	 * \param data_len number of data items in each stream element
	 * \param num_items number of elements in the stream
	 */
	stream_p<meta_type,data_type>(void* stream, int data_len, int num_items) :
		base_stream_p<meta_type,data_type,void,char>(stream, data_len, num_items) { }

	/**
	 * \brief Advances to the next stream element.
	 *
	 * This function will pad the unused tail of the current stream element
	 * with zeroes.
	 */
	void next() 
	{
		std::memset((char*)this->d_stream + this->d_used, 0, this->d_vector_size-this->d_used);
		base_stream_p<meta_type,data_type,void,char>::next();
	}
};

#endif

