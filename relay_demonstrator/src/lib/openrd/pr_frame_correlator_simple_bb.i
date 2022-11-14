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
GR_SWIG_BLOCK_MAGIC(pr,frame_correlator_simple_bb);

pr_frame_correlator_simple_bb_sptr pr_make_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired);

class pr_frame_correlator_simple_bb : public pr_frame_correlator_bb
{
private:
    pr_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired);
};

