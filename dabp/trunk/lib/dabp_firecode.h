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

#ifndef INCLUDED_DABP_FIRECODE_H
#define INCLUDED_DABP_FIRECODE_H

class dabp_firecode
{
    private:
    unsigned short tab[256];
    unsigned short run8(unsigned char regs[]);
    static const unsigned char g[16];
    
    public:
    dabp_firecode();
    ~dabp_firecode();
    // error detection. x[0-1] contains parity, x[2-10] contains data
    bool check(const unsigned char *x); // return true if firecode check is passed
    unsigned short encode(unsigned char *x); // encode x[2-10]. parity will be saved to x[0-1] and be returned as a short
};
#endif // INCLUDED_DABP_FIRECODE_H

