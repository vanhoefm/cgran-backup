/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
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
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef INCLUDED_GMSK_SYMBOLS_H
#define INCLUDED_GMSK_SYMBOLS_H

#include <pmt.h>

// TX
static pmt_t s_cmd_mod = pmt_intern("cmd-mod");
static pmt_t s_response_mod = pmt_intern("response-mod");
static pmt_t s_cmd_demod = pmt_intern("cmd-demod");
static pmt_t s_response_demod = pmt_intern("response-demod");

// RX

#endif // INCLUDED_GMSK_SYMBOLS_H
