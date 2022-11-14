/* -*- c++ -*- */
/*
 * Copyright 2004,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
#ifndef INCLUDED_DABP_FIB_CONSTANTS_H
#define INCLUDED_DABP_FIB_CONSTANTS_H

#define FIB_LENGTH 32
#define FIB_CRC_LENGTH 2
#define FIB_CRC_POLY 0x1021
#define FIB_CRC_INITSTATE 0xffff
#define FIB_ENDMARKER 0xff
#define FIB_PADDING 0x00
#define FIB_FIG_TYPE_MCI    0
#define FIB_FIG_TYPE_LABEL1 1
#define FIB_FIG_TYPE_LABEL2 2
#define FIB_FIG_TYPE_FIDC   5
#define FIB_FIG_TYPE_CA     6

#define FIB_FIDC_EXTENSION_PAGING 0
#define FIB_FIDC_EXTENSION_TMC    1
#define FIB_FIDC_EXTENSION_EWS    2

#define MAX_NUM_SUBCH 64

#define MAX_DUPLICATE 8

#endif //INCLUDED_DABP_FIB_CONSTANTS_H
