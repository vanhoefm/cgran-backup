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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pr_dpacket.h"
#include <iostream>
#include <boost/format.hpp>
#include "crc16.h"

using namespace std;
using namespace boost;

struct pr_dpacket_header
{
	unsigned short crc;
	unsigned char protocol;
	unsigned char seq;
	unsigned short length;
};

pr_dpacket_sptr pr_make_dpacket(unsigned int size, bool raw)
{
	return pr_dpacket_sptr(new pr_dpacket(size, raw));
}

pr_dpacket::pr_dpacket(unsigned int size, bool raw)
{
	if(raw)
	{
		if(size < sizeof(pr_dpacket_header))
		{
			cerr << format("pr_dpacket: size %d too small when allocating packet") % size << endl;
			throw 0;
		}
		d_size = size;
	}
	else
	{
		d_size = size + sizeof(pr_dpacket_header);
	}

	d_data = new unsigned char[d_size];
	fill(d_data, d_data+sizeof(pr_dpacket_header), 0);
}

pr_dpacket::~pr_dpacket()
{
	delete[] d_data; 
}

unsigned int pr_dpacket::raw_size() const
{
	return d_size;
}

unsigned char* pr_dpacket::raw()
{
	return d_data;
}

unsigned int pr_dpacket::data_size() const
{
	return d_size - sizeof(pr_dpacket_header);
}

unsigned char* pr_dpacket::data()
{
	return d_data + sizeof(pr_dpacket_header);
}

unsigned char pr_dpacket::protocol() const
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	return head->protocol & 0x7f;
}

void pr_dpacket::set_protocol(unsigned char protocol)
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	if(protocol >= 0x80)
	{
		cerr << format("pr_dpacket::set_protocol() : protocol %d is invalid") % protocol << endl;
		throw 0;
	}

	head->protocol = (head->protocol & 0x80) | protocol;
}

bool pr_dpacket::initial_packet() const
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	return (head->protocol & 0x80) != 0;
}

void pr_dpacket::set_initial_packet()
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	head->protocol |= 0x80;
}

unsigned int pr_dpacket::length() const
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	if(!initial_packet())
	{
		cerr << format("pr_dpacket::length() : only valid for initial packets") << endl;
		return 0;
	}

	return head->length;
}

void pr_dpacket::set_length(unsigned int length)
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	if(!initial_packet())
	{
		cerr << format("pr_dpacket::set_length() : only valid for initial packets") << endl;
		return;
	}

	head->length = length;
}

unsigned int pr_dpacket::window() const
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	if(initial_packet())
	{
		cerr << format("pr_dpacket::window() : not valid for initial packets") << endl;
		return 0;
	}

	return head->length;
}

void pr_dpacket::set_window(unsigned int window)
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	if(initial_packet())
	{
		cerr << format("pr_dpacket::set_window() : not valid for initial packets") << endl;
		return;
	}

	head->length = window;
}

unsigned int pr_dpacket::seq() const
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	return head->seq;
}

void pr_dpacket::set_seq(unsigned int seq)
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	head->seq = seq;
}

void pr_dpacket::calculate_crc()
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	head->crc = crc16(d_data+2, d_size-2);
}

bool pr_dpacket::check_crc() const
{
	pr_dpacket_header* head = (pr_dpacket_header*) d_data;

	return head->crc == crc16(d_data+2, d_size-2);
}

void pr_dpacket::set_rawv(const vector<unsigned char>& data)
{
	if(data.size() != raw_size())
	{
		cerr << format("pr_dpacket::set_rawv(): length of data is %d but should be %d") %
			data.size() % raw_size() << endl;

		throw 0;
	}

	for(unsigned int i = 0; i < data.size(); i++)
		d_data[i] = data[i];
}

vector<unsigned char> pr_dpacket::rawv() const
{
	vector<unsigned char> d(raw_size());

	for(unsigned int i = 0; i < raw_size(); i++)
		d[i] = d_data[i];

	return d;
}

void pr_dpacket::set_datav(const vector<unsigned char>& data)
{
	if(data.size() != data_size())
	{
		cerr << format("pr_dpacket::set_datav(): length of data is %d but should be %d") %
			data.size() % data_size() << endl;

		throw 0;
	}

	for(unsigned int i = 0; i < data.size(); i++)
		d_data[sizeof(pr_dpacket_header) + i] = data[i];
}

vector<unsigned char> pr_dpacket::datav() const
{
	vector<unsigned char> d(data_size());

	for(unsigned int i = 0; i < data_size(); i++)
		d[i] = d_data[sizeof(pr_dpacket_header) + i];

	return d;
}

