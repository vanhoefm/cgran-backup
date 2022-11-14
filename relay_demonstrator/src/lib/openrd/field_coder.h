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
#ifndef INCLUDED_FIELD_CODER_H
#define INCLUDED_FIELD_CODER_H

#include "field_code_type.h"

#include <vector>

/**
 * \brief Error correction coding for short messages
 *
 * \ingroup prim
 * This class provides efficient error correction coding for very short 
 * source blocks (max 32 bits). The supported codes are specified by the
 * \ref field_code_type type. There are no restrictions on the lengths of
 * the codewords.
 */
class field_coder
{
public:
	/**
	 * Public constructor
	 *
	 * \param code Specifies coding type
	 */
	field_coder(field_code_type code);
	
	/**
	 * Public destructor
	 */
	~field_coder();

	/**
	 * Returns the length of a codeword
	 *
	 * \returns length of codeword
	 */
	int codeword_length() const;

	/**
	 * Static version of codeword_length()
	 *
	 * \param code code type
	 */
	static int codeword_length(field_code_type code);

	/**
	 * Codes val to codeword using the specified code
	 *
	 * \param val source message
	 * \param codeword encoded word
	 */
	void code(unsigned int val, char* codeword);

	/**
	 * Does soft decoding of a codeword
	 *
	 * \param codeword noisy codeword
	 * \param val pointer to decoded word
	 * \returns 0 if success, non-zero if failure
	 */
	int decode(const float* codeword, unsigned int* val);

private:
	field_code_type d_code;
};

#endif

