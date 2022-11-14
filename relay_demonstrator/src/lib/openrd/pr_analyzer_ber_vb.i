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
GR_SWIG_BLOCK_MAGIC(pr,analyzer_ber_vb);

pr_analyzer_ber_vb_sptr pr_make_analyzer_ber_vb (int block_size, int avg_blocks);

class pr_analyzer_ber_vb : public pr_analyzer_vb
{
private:
    pr_analyzer_ber_vb(int block_size, int avg_blocks);

public:
    std::vector<float> ber();
    std::vector<float> bler();
};

