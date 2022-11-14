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
#ifndef INCLUDED_STREAM_META_H
#define INCLUDED_STREAM_META_H

/**
 * \brief Meta data used in RX processing chain
 * \ingroup sigmeta
 */
struct rxframe
{
	float power; /*!< Estimated SNR of frame */
	unsigned int stamp; /*!< Time stamp of the receival of the frame */
	unsigned int pkt_seq; /*!< Packet sequence number */
	unsigned int frame_seq; /*!< Frame sequence number */
	unsigned int frame_type; /*!< Frame type (for framing methods that 
							   support multiple frame types */
	unsigned int pad; /*!< Pad the structure length to a power-of-two */
};

/**
 * \brief Meta data of output from RX processing chain
 * \ingroup sigmeta
 */
struct rxmeta
{
	unsigned int pkt_seq; /*!< Packet sequence number */
	unsigned char decoded; /*!< 0 if packet was successfully decoded */
	unsigned char pad[3]; /*!< Pad the structure length to a power-of-two */
};

/**
 * \brief Meta data of input to TX processing chain
 * \ingroup sigmeta
 */
struct txmeta
{
	unsigned int pkt_seq; /*!< Packet sequence number */
	unsigned int frame_seq; /*!< Frame sequence number */
	unsigned char data_valid; /*!< Denotes whether the frame contains useful
								data (1) or not (0) */
	unsigned char pad[7]; /*!< Pad the structure length to a power-of-two */
};

#endif

