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
#ifndef INCLUDED_DABP_SUPER_FRAME_SINK_H
#define INCLUDED_DABP_SUPER_FRAME_SINK_H
#include <gr_sync_block.h>
#include <cstdio>
#include "dabp_firecode.h"
#include "dabp_crc16.h"
#include <boost/thread.hpp>

class dabp_super_frame_sink;

typedef boost::shared_ptr<dabp_super_frame_sink> dabp_super_frame_sink_sptr;

/*! Instantiate DAB+ super frame sink
 * \param subchidx subchannel index = length of logical frame / 24
 * \param filename a pointer to the output file name string. "-" means stdout
 */
dabp_super_frame_sink_sptr dabp_make_super_frame_sink(int subchidx, const char *filename);

/*! Instantiate DAB+ super frame sink
 * \param subchidx subchannel index = length of logical frame / 24
 * \param filedesc the output file descriptor
 */
dabp_super_frame_sink_sptr dabp_make_super_frame_sink(int subchidx, int filedesc);

/*! \brief DAB+ super frame sink
 * \ingroup sink
 * It takes in a super frame of 110*subchidx bytes at a time and is a sync_block
 * The output is in an ADTS file format written to the given file
 */
class dabp_super_frame_sink : public gr_sync_block
{
    private:
    friend dabp_super_frame_sink_sptr dabp_make_super_frame_sink(int subchidx, const char *filename);
    friend dabp_super_frame_sink_sptr dabp_make_super_frame_sink(int subchidx, int filedesc);
    dabp_super_frame_sink(int subchidx, const char *filename);
    dabp_super_frame_sink(int subchidx, int filedesc);
    void init_header();
    void write(unsigned char *in);
    
    int d_subchidx; // subchannel index
    FILE *d_fp;
    dabp_firecode d_firecode;
    dabp_crc16 d_crc16;
    
    int au_start[7];
    int num_aus;
    unsigned char d_header[7]; // adts header
    struct adts_fixed_header {
        unsigned int syncword           :12;
        unsigned int id                 :1;
        unsigned int layer              :2;
        unsigned int protection_absent  :1;
        unsigned int profile_objecttype :2;
        unsigned int sampling_freq_idx  :4;
        unsigned int private_bit        :1;
        unsigned int channel_conf       :3;
        unsigned int original_copy      :1;
        unsigned int home               :1;
    }d_fh;
    struct adts_variable_header {
        unsigned int copyright_id_bit       :1;
        unsigned int copyright_id_start     :1;
        unsigned int aac_frame_length       :13;
        unsigned int adts_buffer_fullness   :11;
        unsigned int no_raw_data_blocks     :2;
    }d_vh;
    
    boost::mutex d_mutex;
    unsigned char * d_ibuf;
    int d_icnt;
    
    public:
    ~dabp_super_frame_sink();
    void set_bits(unsigned char x[], unsigned int bits, int start_position, int num_bits);
	int work (int noutput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
    void reset(int subchidx);
};

#endif // INCLUDED_DABP_SUPER_FRAME_SINK_H

