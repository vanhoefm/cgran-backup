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
#ifndef INCLUDED_FRAMING_TYPE_H
#define INCLUDED_FRAMING_TYPE_H

/**
 * Specifies the framing type, see the technical documentation for details.
 * \ingroup mode
 */
enum framing_type
{
	/**
	 * Dummy framing, handled by the \ref pr_framer_none_vbb and
	 * \ref pr_deframer_none_vcc classes.
	 */
	FRAMING_NONE,

	/**
	 * Simple framing, handled by the \ref pr_framer_simple_vbb and
	 * \ref pr_deframer_simple_vcc classes.
	 */
	FRAMING_SIMPLE,

	/**
	 * GSM framing, handled by the \ref pr_framer_gsm_vbb and
	 * \ref pr_deframer_gsm_vcc classes.
	 */
	FRAMING_GSM
};

#endif

