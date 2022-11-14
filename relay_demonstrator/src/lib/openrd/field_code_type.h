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
#ifndef INCLUDED_FIELD_CODE_TYPE_H
#define INCLUDED_FIELD_CODE_TYPE_H

/**
 * Specifies coding for fields in frame structures
 *
 * \ingroup prim
 */
enum field_code_type
{
	FIELD_CODE_32_78,          /*!< Dummy code with 32 bit data and 78 bit codeword */
	FIELD_CODE_REPEAT3_16_78,  /*!< Repetition code with 16 bit data and 78 bit codeword */
	FIELD_CODE_REPEAT7_11_78,  /*!< Repetition code with 11 bit data and 78 bit codeword */
	FIELD_CODE_GOLAY_12_78,    /*!< Binary Golay code expanded to 78 bits */
	FIELD_CODE_GOLAY_R3_12_78, /*!< Binary Golay code in repeat-3 code */
	FIELD_CODE_GOLAY_R2_12_78, /*!< Binary Golay code in repeat-2 code */
	FIELD_CODE_R2_RM_11_78     /*!< Repeat-2 code (for detection) in inner Reed-Muller(2,6) code */
};

#endif

