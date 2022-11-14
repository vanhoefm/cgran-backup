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
#ifndef INCLUDED_MODES_H
#define INCLUDED_MODES_H

/**
 * Specifies the source data mode, see the technical documentation for details.
 * \ingroup mode
 */
enum source_data_mode
{
	/**
	 * Zero source data, handled by \ref pr_data_source_zero.
	 */
	SRCMODE_ZERO,
	/**
	 * Deterministic source data where all packets are equal. Handled by
	 * \ref pr_data_source_counter.
	 */
	SRCMODE_COUNTER,
	/**
	 * Deterministic pseudo-random data, not implemented.
	 */
	SRCMODE_RANDOM,
	/**
	 * Deterministic data from file, not implemented.
	 */
	SRCMODE_FILE,
	/**
	 * Dynamic application-generated data, handled by
	 * \ref pr_data_source_packet and \ref pr_data_sink_packet.
	 */
	SRCMODE_PACKET
};

/**
 * Specifies the data analysis mode, see the technical documentation for
 * details.
 * \ingroup mode
 */
enum analysis_mode
{
	/**
	 * Dummy analysis, handled by \ref pr_analyzer_none_vb.
	 */
	AMODE_NONE,
	/**
	 * Bit/block error rate analysis, handled by \ref pr_analyzer_ber_vb.
	 */
	AMODE_BER
};

/**
 * Specifies the relaying mode, see the technical documentation for details.
 * \ingroup mode
 */
enum relaying_mode
{
	/**
	 * No relaying.
	 */
	RELAYING_NONE,
	/**
	 * Amplify-and-forward relaying.
	 */
	RELAYING_AAF,
	/**
	 * Decode-and-forward relaying.
	 */
	RELAYING_DAF
};

/**
 * Specifies the simulated channels when the link is socket or wired. See
 * the technical documentation for details.
 * \ingroup mode
 */
enum channel_mode
{
	/**
	 * Ideal noise-less channel.
	 */
	CHMODE_IDEAL, 
	/**
	 * Additive white Gaussian noise channel.
	 */
	CHMODE_AWGN,
	/**
	 * Rayleigh fading channel, not implemented.
	 */
	CHMODE_RAYLEIGH
};

/**
 * Specifies the control mode for the application. See the technical
 * documentation for details.
 * \ingroup mode
 */
enum control_mode
{
	/**
	 * No control.
	 */
	CNTRLMODE_NONE,
	/**
	 * Console control.
	 */
	CNTRLMODE_CONSOLE,
	/**
	 * Remote control, not implemented.
	 */
	CNTRLMODE_REMOTE
};

#endif

