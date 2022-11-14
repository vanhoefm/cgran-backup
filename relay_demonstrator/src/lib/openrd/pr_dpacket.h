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
#ifndef INCLUDED_PR_DPACKET_H
#define INCLUDED_PR_DPACKET_H

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <vector>

class pr_dpacket;
typedef boost::shared_ptr<pr_dpacket> pr_dpacket_sptr;

/**
 * Public constructor.
 *
 * \param size Size of packet.
 * \param raw Specifies if \p size defines size including header or not.
 */
pr_dpacket_sptr pr_make_dpacket(unsigned int size, bool raw = true);

/**
 * \brief Packet structure containing protocol, sequence, CRC, and data.
 *
 * \ingroup primcls
 * The packet contains a six-byte header with the following structure:
 * \verbatim
 * +----------+----------+----------+----------+------- - - --+
 * |   CRC    | Protocol |   Seq    |  Length  |     Data     |
 * +----------+----------+----------+----------+------- - - --+
 *      16          8          8         16
 * \endverbatim
 *
 * The length of each packet is fixed and is specified during allocation.
 *
 * Valid protocols are 0 to 127. The msb of the protocol is used to indicate
 * the start of a transmission consisting of several packets. All packets
 * that are part of a transmission are supposed to have the same sequence
 * number, and their respective positions are determined by the length field.
 *
 * When the initial packet bit (msb of protocol) is set, the length field
 * contains the total length (in bytes) of the transmission, and the field
 * can be accessed through the length() and set_length() methods. When the
 * initial packet bit is cleared, the length field contains the offset of the
 * data in the transmission, and the field can be accessed through the
 * window() and set_window() methods.
 *
 * During transmission, the following should be done:
 *  - Allocate pr_dpacket.
 *  - Set protocol, optionally initial packet, sequence, length/window and 
 *  data.
 *  - Set CRC using calculate_crc().
 *  - Read raw data using raw().
 *
 * During reception, the following should be done:
 *  - Allocate pr_dpacket.
 *  - Set raw data using raw().
 *  - Check CRC using check_crc().
 *  - Read header fields and data.
 */
class pr_dpacket : public boost::noncopyable
{
public:
	/**
	 * Public constructor.
	 *
	 * \param size Size of packet.
	 * \param raw Specifies if \p size defines size including header or not.
	 */
	pr_dpacket(unsigned int size, bool raw = true);

	/**
	 * Public destructor.
	 */
	~pr_dpacket();

	/**
	 * Returns the raw size of the packet (including header).
	 */
	unsigned int raw_size() const;

	/**
	 * Returns a pointer to the raw data of the packet.
	 */
	unsigned char* raw();

	/**
	 * Returns the data size of the packet.
	 */
	unsigned int data_size() const;

	/**
	 * Returns a pointer to the data of the packet.
	 */
	unsigned char* data();

	/**
	 * Returns protocol.
	 */
	unsigned char protocol() const;

	/**
	 * Sets protocol.
	 *
	 * \param protocol Protocol (valid range is 0 .. 127).
	 */
	void set_protocol(unsigned char protocol);

	/**
	 * Returns whether the initial packet bit is set.
	 */
	bool initial_packet() const;

	/**
	 * Sets the initial packet bit.
	 */
	void set_initial_packet();

	/**
	 * Returns the length field. Can only be called when initial_packet()
	 * returns true.
	 */
	unsigned int length() const;

	/**
	 * Sets the length field. Can only be called when initial_packet() 
	 * returns true.
	 *
	 * \param length Length (valid range is 0 .. 65535).
	 */
	void set_length(unsigned int length);

	/**
	 * Returns the length field. Can only be called when initial_packet()
	 * returns false.
	 */
	unsigned int window() const;

	/**
	 * Sets the length field. Can only be called when initial_packet() 
	 * returns false.
	 *
	 * \param window Window (valid range is 0 .. 65535).
	 */
	void set_window(unsigned int window);

	/**
	 * Returns the sequence field.
	 */
	unsigned int seq() const;

	/**
	 * Sets the sequence field.
	 *
	 * \param seq Sequence number (valid range is 0 .. 255).
	 */
	void set_seq(unsigned int seq);

	/**
	 * Calculates and sets CRC field.
	 */
	void calculate_crc();

	/**
	 * Returns true if the CRC field is correct.
	 */
	bool check_crc() const;

	/**
	 * Python accessor function for setting raw data.
	 *
	 * \param data Raw data vector. Must have size == raw_size().
	 */
	void set_rawv(const std::vector<unsigned char>& data);

	/**
	 * Python accessor function for reading raw data.
	 */
	std::vector<unsigned char> rawv() const;

	/**
	 * Python accessor function for setting data.
	 *
	 * \param data Data vector. Must have size == data_size().
	 */
	void set_datav(const std::vector<unsigned char>& data);

	/**
	 * Python accessor function for reading data.
	 */
	std::vector<unsigned char> datav() const;

private:
	pr_dpacket();

	unsigned int d_size;
	unsigned char* d_data;
};

#endif

