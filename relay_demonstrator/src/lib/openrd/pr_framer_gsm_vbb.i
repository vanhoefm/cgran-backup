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
GR_SWIG_BLOCK_MAGIC(pr,framer_gsm_vbb);

pr_framer_gsm_vbb_sptr pr_make_framer_gsm_vbb(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code);

class pr_framer_gsm_vbb : public pr_framer_vbb
{
private:
    pr_framer_gsm_vbb(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code);
};

