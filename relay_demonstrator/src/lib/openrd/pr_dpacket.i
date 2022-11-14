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
%include <noncopyable.h>

class pr_dpacket;
typedef boost::shared_ptr<pr_dpacket> pr_dpacket_sptr;
%template(pr_dpacket_sptr) boost::shared_ptr<pr_dpacket>;
%rename(dpacket) pr_make_dpacket;

pr_dpacket_sptr pr_make_dpacket(unsigned int size, bool raw = true);

class pr_dpacket : public boost::noncopyable
{
private:
    friend pr_dpacket_sptr pr_make_dpacket(unsigned int size, bool raw);

    pr_dpacket(unsigned int size, bool raw);

public:
    ~pr_dpacket();

    unsigned int raw_size() const;
    unsigned int data_size() const;

    unsigned char protocol() const;
    void set_protocol(unsigned char protocol);

    bool initial_packet() const;
    void set_initial_packet();

    unsigned int length() const;
    void set_length(unsigned int length);

    unsigned int window() const;
    void set_window(unsigned int window);

    unsigned int seq() const;
    void set_seq(unsigned int seq);

    void calculate_crc();
    bool check_crc() const;

    void set_rawv(const std::vector<unsigned char>& data);
    std::vector<unsigned char> rawv() const;

    void set_datav(const std::vector<unsigned char>& data);
    std::vector<unsigned char> datav() const;
};

