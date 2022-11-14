/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 * 
 * This file is part of FOI-MIMO
 * 
 * FOI-MIMO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * FOI-MIMO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FOI-MIMO; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include<foimimo_crc32_inserter.h>
#include<gr_io_signature.h>
#include<gr_crc32.h>
#include<string.h>
#include <cstdio>

#define VERBOSE 0


foimimo_crc32_inserter_sptr
foimimo_make_crc32_inserter(unsigned int bytes_per_pkt, unsigned int msgq_limit)
{
  return foimimo_crc32_inserter_sptr(new foimimo_crc32_inserter(bytes_per_pkt,msgq_limit));
}

foimimo_crc32_inserter::foimimo_crc32_inserter(unsigned int bytes_per_packet, unsigned int msgq_limit)
  : gr_sync_block("foimimo_crc32_inserter",
                  gr_make_io_signature (0, 0, 0),
                  gr_make_io_signature2 (2, 2, sizeof(char),sizeof(char))),
                  d_msgq(gr_make_msg_queue(msgq_limit)),
                  d_nresidbyte(0),
                  d_bytes_per_packet(bytes_per_packet-HEADER_SIZE-CRC32_SIZE),
                  d_eof(false)
{
  set_output_multiple(d_bytes_per_packet+HEADER_SIZE+CRC32_SIZE);
  set_relative_rate(d_bytes_per_packet+HEADER_SIZE+CRC32_SIZE);
  if (VERBOSE)
    printf("foimimo_crc32_inserter bytes_per_packet:%i\n",d_bytes_per_packet);

}
foimimo_crc32_inserter::~foimimo_crc32_inserter(void)
{
}

unsigned int
foimimo_crc32_inserter::make_header(unsigned int bytes_per_packet){
  // header format upper nibble is unused the three lower nibble contains nr of bytes in one packet.
  // the 16 bit header is copied so that the receiver can check if the header is correct by comparing
  // both parts.


  return ((bytes_per_packet) & 0x0fff) | (((bytes_per_packet) & 0x0fff) << 16);
}

int
foimimo_crc32_inserter::work(int noutput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items)
{
  unsigned char *out = (unsigned char *)output_items[0];
  unsigned char *out_new_pkt = (unsigned char *)output_items[1];

  int i_byte = 0;
  int i_header = 0;
  unsigned int crc32;

  if(d_eof) {
    return -1;
  }

  if (!d_msg){
    d_msg = d_msgq->delete_head();         // block, waiting for a message
    d_msg_offset = 0;
    if((d_msg->length() == 0) && (d_msg->type() == 1)){ // check if it is the last msg coming
      d_eof = true;
      d_msg.reset();

      return -1;                // We're done; no more messages coming.
    }
    if(d_msg->length() == 0){
      d_eof = true;
      d_msg.reset();

      return -1;                // We're done; no more messages coming.
    }

  }

  // copy msg bytes to output vector
  while (d_msg_offset+d_bytes_per_packet <= d_msg->length() && d_bytes_per_packet+CRC32_SIZE+HEADER_SIZE+i_byte <= noutput_items){
    memset(&out_new_pkt[i_byte],0,sizeof(char)*(d_bytes_per_packet+CRC32_SIZE+HEADER_SIZE));
    memset(&out[i_byte],0,sizeof(char)*(d_bytes_per_packet+CRC32_SIZE+HEADER_SIZE));
    //insert header
    unsigned int header_tmp = make_header(d_bytes_per_packet+CRC32_SIZE);

    if (VERBOSE)
      printf("CRC32: header:%x\n",header_tmp);
    out_new_pkt[i_byte] = 1;
    out[i_byte++]= (unsigned char) ((header_tmp >> 24) & 0xFF); //fixme insert code rate
    out[i_byte++]= (unsigned char) ((header_tmp >> 16) & 0xFF);
    out[i_byte++]= (unsigned char) ((header_tmp >> 8) & 0xFF); //fixme insert code rate
    out[i_byte++]= (unsigned char) ((header_tmp) & 0xFF);
    //insert payload
    for(int j = 0; j<d_bytes_per_packet; j++){
      out[i_byte+j] = d_msg->msg()[d_msg_offset++];
    }
    //insert crc32
    crc32 = gr_crc32(&out[i_byte],d_bytes_per_packet);
    if (VERBOSE)
        printf("foimimo_crc32_inserter> crc:0x%X\n",crc32);

    out[i_byte + d_bytes_per_packet]   = ((crc32>>24)& 0xff);
    out[i_byte + d_bytes_per_packet+1] = ((crc32>>16)& 0xff);
    out[i_byte + d_bytes_per_packet+2] = ((crc32>>8) & 0xff);
    out[i_byte + d_bytes_per_packet+3] =      (crc32 & 0xff);

    i_byte += d_bytes_per_packet + CRC32_SIZE;

    if (VERBOSE)
        printf("i_byte:%i d_msg_offset:%i msg_length:%i \n",i_byte,d_msg_offset,d_msg->length());
  }

  // Save the residual byte
  int j = 0;
  int nresidbyte =  d_msg->length()-d_msg_offset;
  if (nresidbyte == 0){
    d_msg.reset();                          // finished packet, free message
    assert(i_byte > 0);
    return i_byte;
  }
  //Check if we have residual bytes and if there are enough output space left.
  if (nresidbyte < d_bytes_per_packet && nresidbyte+CRC32_SIZE+HEADER_SIZE+i_byte <= noutput_items){
      memset(&out_new_pkt[i_byte],0,sizeof(char)*(nresidbyte+CRC32_SIZE+HEADER_SIZE));
      memset(&out[i_byte],0,sizeof(char)*(nresidbyte+CRC32_SIZE+HEADER_SIZE));
      unsigned int header_tmp = make_header(nresidbyte+CRC32_SIZE);
      out_new_pkt[i_byte] = 1;
      out[i_byte++]= (unsigned char) ((header_tmp >> 24) & 0xFF); //fixme insert code rate
      out[i_byte++]= (unsigned char) ((header_tmp >> 16) & 0xFF);
      out[i_byte++]= (unsigned char) ((header_tmp >> 8) & 0xFF); //fixme insert code rate
      out[i_byte++]= (unsigned char) ((header_tmp) & 0xFF);
     
      while (d_msg_offset < d_msg->length()){
        out[i_byte+j]= d_msg->msg()[d_msg_offset++];
        j++;
      }
      //insert CRC23
      crc32 = gr_crc32(&out[i_byte],j);

      i_byte += j;
      out[i_byte]   = ((crc32>>24)& 0xff);
      out[i_byte+1] = ((crc32>>16)& 0xff);
      out[i_byte+2] = ((crc32>>8) & 0xff);
      out[i_byte+3] =      (crc32 & 0xff);
      i_byte += CRC32_SIZE;

      d_msg.reset();                          // finished packet, free message
  }
  assert(i_byte <= noutput_items);
  assert(i_byte > 0);
  return i_byte;
}

