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
#ifndef INCLUDED_DABP_CRC16_H
#define INCLUDED_DABP_CRC16_H

class dabp_crc16
{
    private:
    unsigned short tab[256]; // indices are the last 8 bits of shift registers. table values are the register contents after 8 bit shift
    unsigned short run8(char unsigned regs[]);
    static const unsigned char g[16];
    
    public:
    dabp_crc16();
    ~dabp_crc16();
    bool check(const unsigned char *x, int data_size); // return true if crc check is passed
    void generate(unsigned char *x, int data_size); // x[] must have a minimum length of data_size+2
};
#endif // INCLUDED_DABP_CRC16_H

