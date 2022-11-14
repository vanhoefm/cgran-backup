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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dabp_super_frame_sink.h"
#include <gr_io_signature.h>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <gruel/thread.h>

dabp_super_frame_sink_sptr
dabp_make_super_frame_sink (int subchidx, const char *filename)
{
  return dabp_super_frame_sink_sptr (new dabp_super_frame_sink (subchidx, filename));
}
dabp_super_frame_sink_sptr
dabp_make_super_frame_sink (int subchidx, int filedesc)
{
    return dabp_super_frame_sink_sptr (new dabp_super_frame_sink (subchidx, filedesc));
}

dabp_super_frame_sink::dabp_super_frame_sink(int subchidx, const char *filename)
  : gr_sync_block ("super_frame_sink",
		   gr_make_io_signature2(2, 2, sizeof(unsigned char), sizeof(char)),
		   gr_make_io_signature(0, 0, 0)), d_subchidx(subchidx)
{
    if(strcmp(filename,"-")==0) { // stdout
        d_fp=stdout;
    }else{
        d_fp=fopen(filename,"wb");
    }
    if(d_fp==NULL) {
        throw std::runtime_error ("can't open sink file");
    }
    set_output_multiple(110);
    
    d_ibuf = new unsigned char[110*subchidx];
    d_icnt = 0;
    
    init_header();
}

dabp_super_frame_sink::dabp_super_frame_sink(int subchidx, int filedesc)
    : gr_sync_block ("super_frame_sink",
        gr_make_io_signature2(2, 2, sizeof(unsigned char), sizeof(char)),
        gr_make_io_signature(0, 0, 0)), 
    d_subchidx(subchidx)
{
    if(filedesc==1) { // stdout
        d_fp=stdout;
    }else if(filedesc==2) { // stderr
        d_fp=stderr;
    }else if(filedesc==0) { // stdin ERROR!
        d_fp=NULL;
    }else{
        d_fp=fdopen(filedesc,"wb");
    }
    if(d_fp==NULL) {
        throw std::runtime_error ("can't open sink file");
    }
    set_output_multiple(110);
    
    d_ibuf = new unsigned char[110*subchidx];
    d_icnt = 0;
    
    init_header();
}

void dabp_super_frame_sink::init_header()
{
    
    // prepare the header (constant portion)
    d_fh.syncword=0xfff;
    d_fh.id=0;
    d_fh.layer=0;
    d_fh.protection_absent=1;
    d_fh.profile_objecttype=0;
    d_fh.private_bit=0;
    d_fh.original_copy=0;
    d_fh.home=0;
    d_vh.copyright_id_bit=0;
    d_vh.copyright_id_start=0;
    d_vh.adts_buffer_fullness=1999; // ? according to OpenDab
    d_vh.no_raw_data_blocks=0;
    
    d_header[0] = d_fh.syncword >> 4;
	d_header[1] = (d_fh.syncword & 0xf) << 4;
	d_header[1] |= d_fh.id << 3;
	d_header[1] |= d_fh.layer << 1;
	d_header[1] |= d_fh.protection_absent;
    d_header[2] = d_fh.profile_objecttype << 6;
	d_header[2] |= d_fh.private_bit << 1;
	d_header[3] = d_fh.original_copy << 5;
	d_header[3] |= d_fh.home << 4; 
	d_header[3] |= d_vh.copyright_id_bit << 3;
	d_header[3] |= d_vh.copyright_id_start << 2;
    d_header[4] = 0;
	d_header[5] = d_vh.adts_buffer_fullness >> 6;
	d_header[6] = (d_vh.adts_buffer_fullness & 0x3f) << 2;
	d_header[6] |= d_vh.no_raw_data_blocks;
}

dabp_super_frame_sink::~dabp_super_frame_sink ()
{
    if(d_fp!=stdout && d_fp!=stderr && d_fp!=NULL)
        fclose(d_fp);
    delete [] d_ibuf;
}

void dabp_super_frame_sink::reset(int subchidx)
{
    gruel::scoped_lock guard(d_mutex);
    delete [] d_ibuf;
    d_subchidx = subchidx;
    d_ibuf = new unsigned char[d_subchidx*110];
    d_icnt = 0;
}

int dabp_super_frame_sink::work (int noutput_items,
                                 gr_vector_const_void_star &input_items,
                                 gr_vector_void_star &output_items)
{
    assert(noutput_items%110==0);
    
    const unsigned char *in0=(const unsigned char*)input_items[0];
    const char *in1=(const char*)input_items[1];
    
    int nproduced=0; // the number of consumed input items & produced output items
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    const int ilen=d_subchidx*110;
    while(nproduced<noutput_items) {
        // process input from in to ibuf
        assert(d_icnt<ilen);
        if(d_icnt<ilen) { // ibuf not full, fill it
            if(d_icnt==0) { // check indicator
                while(nproduced<noutput_items && *in1 == 0) { // skip to start of super frame
                    in0 ++;
                    in1 ++;
                    nproduced++;
                }
                if(nproduced==noutput_items) { // nothing available from the input
                    break;
                }
            }
            if(d_icnt+noutput_items-nproduced<ilen) { // not enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (noutput_items-nproduced)*sizeof(unsigned char));
                d_icnt += noutput_items-nproduced;
                nproduced = noutput_items;
                break;
            }else { // enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ilen-d_icnt)*sizeof(unsigned char));
                in0 += ilen-d_icnt;
                in1 += ilen-d_icnt;
                nproduced += ilen-d_icnt;
                d_icnt = ilen;
                
                // output ibuf->adts file
                write(d_ibuf);
                d_icnt = 0;
            }
        }
        
        
    }
    return nproduced;
}

void dabp_super_frame_sink::write(unsigned char *in)
{
    unsigned int dac_rate, sbr_flag, aac_channel_mode, ps_flag, mpeg_surround_config;
    int i,j;
    if(d_firecode.check(in)) { // firecode check ok=> update header; otherwise, use old header
        dac_rate=(in[2]>>6)&1;
        sbr_flag=(in[2]>>5)&1;
        aac_channel_mode=(in[2]>>4)&1;
        ps_flag=(in[2]>>3)&1;
        mpeg_surround_config=in[2]&7;
        if(!dac_rate && sbr_flag) {
            num_aus=2;
            au_start[0]=5;
            au_start[1]=in[3]*16+(in[4]>>4);
            au_start[2]=110*d_subchidx;
            d_fh.sampling_freq_idx=8;
        }else if(dac_rate && sbr_flag) {
            num_aus=3;
            au_start[0]=6;
            au_start[1]=in[3]*16+(in[4]>>4);
            au_start[2]=(in[4]&0xf)*256+in[5];
            au_start[3]=110*d_subchidx;
            d_fh.sampling_freq_idx=6;
        }else if(!dac_rate && !sbr_flag) {
            num_aus=4;
            au_start[0]=8;
            au_start[1]=in[3]*16+(in[4]>>4);
            au_start[2]=(in[4]&0xf)*256+in[5];
            au_start[3]=in[6]*16+(in[7]>>4);
            au_start[4]=110*d_subchidx;
            d_fh.sampling_freq_idx=5;
        }else {
            num_aus=6;
            au_start[0]=11;
            au_start[1]=in[3]*16+(in[4]>>4);
            au_start[2]=(in[4]&0xf)*256+in[5];
            au_start[3]=in[6]*16+(in[7]>>4);
            au_start[4]=(in[7]&0xf)*256+in[8];
            au_start[5]=in[9]*16+(in[10]>>4);
            au_start[6]=110*d_subchidx;
            d_fh.sampling_freq_idx=3;
        }
        switch(mpeg_surround_config) {
            case 0:
                if(sbr_flag && !aac_channel_mode && ps_flag)
                    d_fh.channel_conf=2;
                else
                    d_fh.channel_conf=1<<aac_channel_mode;
                break;
            case 1:
                d_fh.channel_conf=6;
                break;
            default:
                std::cerr<<"Unrecognized mpeg_surround_config ignored!"<<std::endl;
                if(sbr_flag && !aac_channel_mode && ps_flag)
                    d_fh.channel_conf=2;
                else
                    d_fh.channel_conf=1<<aac_channel_mode;
        }
        set_bits(&d_header[2],d_fh.sampling_freq_idx,2,4);
        set_bits(&d_header[2],d_fh.channel_conf,7,3);
    }
    for(j=0;j<num_aus;j++) {
        d_vh.aac_frame_length=au_start[j+1]-au_start[j]-2+7; // including adts header, excluding au_crc
        if(d_crc16.check(&in[au_start[j]],d_vh.aac_frame_length-7)) { // au_crc good?
            set_bits(&d_header[3],d_vh.aac_frame_length,6,13);
            fwrite(d_header,1,7,d_fp);
            fwrite(&in[au_start[j]],1,d_vh.aac_frame_length-7,d_fp);
        }else{
            std::cerr<<"AU_CRC error!"<<std::endl;
        }
    }
}

void dabp_super_frame_sink::set_bits(unsigned char x[], unsigned int bits, int start_position, int num_bits)
{
    int i, ibyte, ibit;
    for(i=0;i<num_bits;i++) {
        ibyte=(start_position+i)/8;
        ibit=(start_position+i)%8;
        x[ibyte] = (x[ibyte]&(~(1<<(7-ibit)))) | (((bits>>(num_bits-i-1))&1)<<(7-ibit));
    }
}

