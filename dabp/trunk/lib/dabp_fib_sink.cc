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

#include <cstring>
#include <gr_io_signature.h>
#include <stdexcept>
#include "dabp_fib_sink.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

dabp_fib_sink_sptr dabp_make_fib_sink ()
{
    return dabp_fib_sink_sptr (new dabp_fib_sink ());
}

dabp_fib_sink::dabp_fib_sink()
  : gr_sync_block ("fib_sink",
		   gr_make_io_signature(1, 1, sizeof(unsigned char)),
		   gr_make_io_signature(0, 0, 0))
{
    set_output_multiple(FIB_LENGTH);
	
	for(int i=0;i<MAX_NUM_SUBCH;i++) {
		d_subch[i].subchid=MAX_NUM_SUBCH; // initialize with invalid id
		d_subch[i].subchsz=0; // invalid subchsz
		d_subch[i].label[0]=0; // empty string
		d_duplicate[i]=0;
	}
}
dabp_fib_sink::~dabp_fib_sink()
{
}

void
dabp_fib_sink::dump_fib(const unsigned char *fib) {
  printf("FIB dump: ");
  for (int i=0; i<FIB_LENGTH; i++)
    printf("%.2x ",fib[i]);
  printf("\n");
}

int 
dabp_fib_sink::process_fib(const unsigned char *fib) {
  unsigned char type, length, pos;
  if (!crc.check(fib,FIB_LENGTH-FIB_CRC_LENGTH)) {
    std::cerr<<"FIB CRC error"<<std::endl;
	return 1;
  }
  pos = 0;
  while (pos<FIB_LENGTH-FIB_CRC_LENGTH && fib[pos]!=FIB_ENDMARKER && fib[pos]!=FIB_PADDING) {
    type = fib[pos]>>5;
    length = fib[pos] & 0x1f;
    assert(pos+length<FIB_LENGTH-FIB_CRC_LENGTH);
    assert(length>=1 && length<=29);
    pos++; // now pos points to FIG data field
    if(process_fig(type,&fib[pos],length)<0)
		return -1; // enough duplicates have been obtained, exiting
    pos += length;
  }
  return 0;
}

int
dabp_fib_sink::process_fig(unsigned char type, const unsigned char *data, unsigned char length)
{
	int idx;
	unsigned char label[17];
	unsigned char charset;
	unsigned char oe;
	unsigned char ext;
	unsigned char fidc_ext;
	unsigned short ens_id;
	unsigned int sid;
	unsigned char pd;
	unsigned char scids;
	unsigned short flag;
	unsigned char change_flag;
	unsigned char al_flag;
	unsigned char cif_count_hi, cif_count_lo;
	unsigned short cif_count;
	unsigned char occ_change;
	unsigned char cn;
	int subch_cnt;
	unsigned char subchid;
	unsigned short start_addr;
	unsigned char short_long;
	unsigned char tab_switch;
	unsigned char tab_idx;
	unsigned char option;
	unsigned char protect_lev;
	unsigned short subchsz;
	int serv_cnt;
	unsigned char local_flag;
	unsigned char caid;
	unsigned char ncomps;
	unsigned char comp_cnt;
	unsigned char tmid;
	unsigned char ascty;
	unsigned char ps;
	unsigned char ca_flag;
	int lang_cnt;
	unsigned char ls;
	unsigned char msc_fic;
	unsigned char subchid_fidcid;
	unsigned char language;
	unsigned short scid;
	
	switch (type) {
		case FIB_FIG_TYPE_MCI:
			cn = data[0]>>7;
			if(cn) {
				std::cerr<<"Next configuration: unsupported yet. ignored."<<std::endl;
				break; // ignore this FIG
			}
			oe = (data[0]>>6)&0x01;
			if(oe) {
				std::cerr<<"Other ensemble: unsupported yet. ignored."<<std::endl;
				break; // ignore this FIG
			}
			pd = (data[0]>>5)&0x01;
			if(pd) {
				std::cerr<<"Data service: unsupported yet. ignored."<<std::endl;
				break; // ignore this FIG
			}
			ext = data[0]&0x1f;
			switch(ext) {
				case 0: // FIG 0/0, ensemble info
					ens_id = (data[1]<<8) | data[2];
					change_flag = (data[3]>>6);
					al_flag = (data[3]>>5) & 0x01;
					cif_count_hi = data[3]&0x1f;
					cif_count_lo = data[4];
					cif_count = cif_count_hi*250 + cif_count_lo;
					occ_change = data[5];
					break;
				case 1: // FIG 0/1, basic subchannel organization
					subch_cnt=0;
					idx = 1;
					while(idx<length) {
						subchid = (data[idx]>>2);
						start_addr = ((data[idx]&0x03)<<8) | data[idx+1];
						idx+=2;
						short_long = data[idx]>>7;
						if(short_long) { // long form
							option = (data[idx]>>4) & 0x07;
							protect_lev = (data[idx]>>2) & 0x03;
							subchsz = ((data[idx]&0x03)<<8) | data[idx+1];
							idx+=2;
							// store the subchannel organization
							update_subch_org(subchid, start_addr, subchsz, option, protect_lev);
						}else { // short form
							tab_switch = (data[idx]>>6) & 0x01;
							tab_idx = data[idx] & 0x3f;
							idx++;
						}
						subch_cnt++;
					}
					break;
				case 2: // FIG 0/2, basic service organization
					serv_cnt = 0;
					idx = 1;
					while(idx<length) {
						if(pd) { // 32 bit SId
							sid = (data[idx]<<24) | (data[idx+1]<<16) | (data[idx+2]<<8) | data[idx+3];
							idx += 4;
						}else { // 16 bit SId
							sid = (data[idx]<<8) | data[idx+1];
							idx += 2;
						}
						local_flag = data[idx]>>7;
						caid = (data[idx]>>4) & 0x07;
						ncomps = data[idx] & 0x0f;
						idx++;
						for(comp_cnt=0;comp_cnt<ncomps;comp_cnt++) {
							tmid = data[idx]>>6;
							if(!tmid) { // MSC stream audio
								ascty = data[idx] & 0x3f;
								subchid = data[idx+1]>>2;
								ps = (data[idx+1]>>1)&0x01;
								ca_flag = data[idx+1] & 0x01;
								// store service organization
								update_service_org(subchid, sid);
							}
							idx+=2;
						}
						serv_cnt++;
					}
					break;
				case 5: // FIG 0/5, service component language
					lang_cnt=0;
					idx=1;
					while(idx<length) {
						ls=data[idx]>>7;
						if(ls) { //
							scid=((data[idx]&0x0f)<<8) |data[idx+1];
							idx+=2;
							language = data[idx];
							idx++;
						}else { //
							msc_fic = (data[idx]>>6)&0x01;
							subchid_fidcid = data[idx]&0x3f;
							idx++;
							language = data[idx];
							idx++;
						}
						lang_cnt++;
					}
					break;
				case 13: // FIG 0/13, user application information
					//std::cerr<<"Unsupported FIG 0/"<<(int)ext<<" yet."<<std::endl;
					break;
				case 18: // FIG 0/18, announcement support
					//std::cerr<<"Unsupported FIG 0/"<<(int)ext<<" yet."<<std::endl;
					break;
				default:
					//std::cerr<<"Unsupported FIG 0/"<<(int)ext<<std::endl;
					break;
			}
			break;
		case FIB_FIG_TYPE_LABEL1: // label type 1
			charset = (data[0]>>4);
			oe = ((data[0]>>3) & 1);
			ext = (data[0]&0x07);
			assert(length>=19);
			memcpy(label,&data[length-18],16);
			label[16]=0;
			switch(ext) {
				case 0: // ensemble label
					ens_id = (data[1]<<8) | data[2];
					memcpy(d_ens_label,label,17);
					break;
				case 1: // program service label
					sid = (data[1]<<8) | data[2];
					// store label
					if(update_label(sid,label)<0) 
						return -1; // enough duplicates have been obtained
					break;
				case 5: // data service label
					sid = (data[1]<<24) | (data[2]<<16) | (data[3]<<8) | data[4];
					break;
				case 4: // service component label
					pd = (data[1]>>7);
					scids = (data[1] & 0x0f);
					if(pd==0)
						sid = (data[2]<<8) | data[3];
					else
						sid = (data[2]<<24) | (data[3]<<16) | (data[4]<<8) | data[5];
					break;
				case 6: // X-PAD user application label
					std::cerr<<"X-PAD user application label. Unsupported Yet."<<std::endl;
					break;
				default:
					std::cerr<<"Unsupported label extension."<<std::endl;
					break;
			}
			flag = (data[length-2]<<8) | data[length-1];
			break;
            
		case FIB_FIG_TYPE_LABEL2:
			std::cerr<<"Label Type 2 Unsupported Yet."<<std::endl;
			break;
		case FIB_FIG_TYPE_FIDC:
			std::cerr<<"FIDC: unsupported."<<std::endl;
			break;
		case FIB_FIG_TYPE_CA:
			std::cerr<<"CA (conditional access) not supported yet."<<std::endl;
			break;
		default:
			std::cerr<<"Unsupported FIG type."<<std::endl;
			break;
	}
	return 0;
}

int dabp_fib_sink::work (int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items)
{
    const unsigned char *in = (const unsigned char *) input_items[0];
    assert(noutput_items%FIB_LENGTH==0);
    int nblocks=noutput_items/FIB_LENGTH;
    for (int i=0; i<nblocks; i++) {
        if(process_fib(in)<0) {
            std::cerr<<"Enough information has been retrieved from FIC. Exiting."<<std::endl;
            return -1; // enough information has been obtained from FIC, exiting.
        }
        in+=FIB_LENGTH;
    }
    return noutput_items;
}

int dabp_fib_sink::update_label(unsigned int sid, unsigned char *label)
{
	unsigned char i;
	for(i=0;i<MAX_NUM_SUBCH;i++) {
		if(d_subch[i].subchsz && d_subch[i].subchid==i && d_subch[i].sid==sid) {
			memcpy(d_subch[i].label, label, 17);
			break;
		}
	}
	if(i>=MAX_NUM_SUBCH) // sid not found
		return 0;
	for(i=0;i<MAX_NUM_SUBCH;i++) {
		if(d_subch[i].subchsz && d_subch[i].subchid==i && d_subch[i].label[0] && d_duplicate[i]<MAX_DUPLICATE) // not enough duplicates for this non-empty subchannel
			return 0;
	}
	return -1; // enough duplicates for all non-empty subchannels. exiting
}

void dabp_fib_sink::update_subch_org(unsigned char subchid, unsigned short start_addr, unsigned short subchsz, unsigned char option, unsigned char protect_lev)
{	
	d_subch[subchid].start_addr=start_addr;
	d_subch[subchid].subchsz=subchsz;
	d_subch[subchid].option=option;
	d_subch[subchid].protect_lev=protect_lev;
}

void dabp_fib_sink::update_service_org(unsigned char subchid, unsigned int sid)
{
	if(d_subch[subchid].subchid==subchid) { // already recorded. this is a duplicate
		d_duplicate[subchid]++;
	}
	d_subch[subchid].sid=sid;
	d_subch[subchid].subchid=subchid;
}

void dabp_fib_sink::print_subch()
{
    for(int k=0;k<73;k++)
        std::cerr<<"=";
    std::cerr<<std::endl;
	std::cerr<<"Ensemble Label: "<<d_ens_label<<std::endl;
	for(int k=0;k<73;k++)
        std::cerr<<"=";
    std::cerr<<std::endl;
	
    std::cerr<<std::setw(8)<<"subchid"<<std::setw(10)<<"sid"<<std::setw(17)<<"label"<<std::setw(11)<<"start_addr"<<std::setw(8)<<"subchsz"<<std::setw(7)<<"option"<<std::setw(12)<<"protect_lev"<<std::endl;
	for(unsigned char i=0;i<MAX_NUM_SUBCH;i++) {
		if(d_subch[i].subchsz && d_subch[i].subchid==i && d_subch[i].label[0]) {
			std::cerr<<std::setw(8)<<(int)i<<std::setw(10)<<d_subch[i].sid<<std::setw(17)<<d_subch[i].label<<std::setw(11)<<d_subch[i].start_addr<<std::setw(8)<<d_subch[i].subchsz<<std::setw(7)<<(int)d_subch[i].option<<std::setw(12)<<(int)d_subch[i].protect_lev<<std::endl;
		}
	}
}

void dabp_fib_sink::save_subch(const char *filename)
{
    std::ofstream f(filename);
    f<<"#";
    for(int k=0;k<73;k++)
        f<<"=";
    f<<std::endl;
	f<<"#Ensemble Label: "<<d_ens_label<<std::endl;
	f<<"#";
    for(int k=0;k<73;k++)
        f<<"=";
    f<<std::endl;
	
    f<<std::setw(8)<<"#subchid"<<std::setw(11)<<"sid "<<std::setw(17)<<"label "<<std::setw(11)<<"start_addr "<<std::setw(8)<<"subchsz "<<std::setw(7)<<"option "<<std::setw(12)<<"protect_lev"<<std::endl;
	for(unsigned char i=0;i<MAX_NUM_SUBCH;i++) {
		if(d_subch[i].subchsz && d_subch[i].subchid==i && d_subch[i].label[0]) {
			f<<std::setw(7)<<(int)i<<','<<std::setw(10)<<d_subch[i].sid<<','<<std::setw(16)<<d_subch[i].label<<','<<std::setw(10)<<d_subch[i].start_addr<<','<<std::setw(7)<<d_subch[i].subchsz<<','<<std::setw(6)<<(int)d_subch[i].option<<','<<std::setw(12)<<(int)d_subch[i].protect_lev<<std::endl;
		}
	}
}

