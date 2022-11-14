/* -*- c++ -*- */
/*
 * Copyright 2008 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
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

#include <gr_hdlc_router_sink_b.h>
#include <gr_io_signature.h>
#include <ppio.h>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdexcept>

// win32 (mingw/msvc) specific
#ifdef HAVE_IO_H
#include <io.h>
#endif
#ifdef O_BINARY
#define	OUR_O_BINARY O_BINARY
#else
#define	OUR_O_BINARY 0
#endif

// should be handled via configure
#ifdef O_LARGEFILE
#define	OUR_O_LARGEFILE	O_LARGEFILE
#else
#define	OUR_O_LARGEFILE 0
#endif


//--------------- Private Methods ------------------------------------


//                                      16   12   5
// this is the CCITT CRC 16 polynomial X  + X  + X  + 1.
// This is 0x1021 when x is 2, but the way the algorithm works
// we use 0x8408 (the reverse of the bit pattern).  The high
// bit is always assumed to be set, thus we only use 16 bits to
// represent the 17 bit value.
//
// We have to use the reverse pattern because serial comm using
// UARTs abd USRTs send out each byte LSBit first, but the
// CRC algorithm is specified using MSBit first.  So this
// whole algorithm is reversed to compensate.  The main loop
// shifts right instead of left as in the cannonical algorithm
//

unsigned short gr_hdlc_router_sink_b::crc16(unsigned char *data_p, unsigned short length)
{
      const int     POLY = 0x8408;   // 1021H bit reversed 
      unsigned char i;
      unsigned int  data;
      unsigned int  crc = 0xffff;

      if (length == 0)
            return (~crc);
      do
      {
            for (i=0, data=(unsigned int)0xff & *data_p++;
                 i < 8; 
                 i++, data >>= 1)
            {
                  if ((crc & 0x0001) ^ (data & 0x0001))
                        crc = (crc >> 1) ^ POLY;
                  else  crc >>= 1;
            }
      } while (--length);

      crc = ~crc;
      data = crc;
      crc = (crc << 8) | (data >> 8 & 0xff);

      return (crc);
}


int
gr_hdlc_router_sink_b::crc_valid(int frame_size, unsigned char * frame)
  {
    unsigned short frame_crc;
    unsigned short calc_crc;

    frame_crc = frame[frame_size-1] | (frame[frame_size-2] << 8);
    calc_crc = crc16(frame, frame_size-2);
    //printf("Frame_crc = %04X   Calc_crc = %04X\n", frame_crc, calc_crc);
    return(calc_crc == frame_crc);
  }        


void
gr_hdlc_router_sink_b::route_packet(int hdlc_frame_size, unsigned char * hdlc_frame)
  {
    static int                ip_socket = 0;
    static struct sockaddr_in dest_addr;
    int                       packet_size;
    unsigned char *           packet;
    int                       flags;
    int                       hincl;
    int                       stat;

    if(ip_socket == 0)
      {
        bzero((char *) &dest_addr, sizeof(dest_addr));
        dest_addr.sin_family      = AF_INET;
        dest_addr.sin_port        = htons(0);
        //dest_addr.sin_addr.s_addr = inet_addr("192.168.3.10");

        ip_socket = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
        if (ip_socket < 0)
          {
            fprintf(stderr, "Failure opening raw IP socket. Are you root?\n");
            perror("");
            exit(-1);
          }
        hincl = 1;           /* 1 = on, 0 = off */
        stat = setsockopt(ip_socket, IPPROTO_IP, IP_HDRINCL, &hincl, sizeof(hincl));
        if(stat < 0)
          {
            fprintf(stderr, "setsockopt failed to set IP_HDRINCL.\n");
            perror("");
            exit(-1);
          }
      }

    // Extract payload size from hdlc frame
    packet_size = (hdlc_frame[6] << 8) | hdlc_frame[7];

    // Set pointer to start of payload
    packet = hdlc_frame + 4;

    // Copy packet's 4-byte dest address into dest_addr structure
    // that is needed for sendto
    bcopy(packet+16, &dest_addr.sin_addr.s_addr, 4);

    // Send IP packet
    flags = 0;
    stat = sendto(ip_socket, 
                  packet, 
                  packet_size, 
                  flags, 
                  (struct sockaddr *)&dest_addr, 
                  sizeof(dest_addr));
    if(stat < 0)
      {
        fprintf(stderr, "sendto failed");
        perror("");
        exit(-1);
      }
  }




int
gr_hdlc_router_sink_b::unstuff(int             bit_buf_size, 
                               unsigned char * bit_buf, 
                               int *           frame_buf_size, 
                               unsigned char * frame_buf)
  {
    int           i;
    unsigned char data_bit;
    int           accumulated_bits;
    int           bytes;
    int           consecutive_one_bits;
    int           status;
 
    accumulated_bits = 0;
    bytes = 0;
    consecutive_one_bits = 0;

    for(i=0; i<bit_buf_size; i++)
      {
        data_bit = bit_buf[i];
        if( (consecutive_one_bits != 5) || (data_bit == 1) )
          { 
            // Not a stuffed 0,  Write it to frame_buf
            frame_buf[bytes] = (frame_buf[bytes] >> 1) | (data_bit << 7);
            accumulated_bits++;
            if(accumulated_bits == 8)
              {
                bytes++;
                accumulated_bits = 0;
              }
          }

        if(data_bit == 1)
          {
            consecutive_one_bits++;
          }
        else
          {
            consecutive_one_bits = 0;
          }
      }

    // Now check for an integer number of bytes
    if(accumulated_bits == 0)
      {
        status = SUCCESS;
        *frame_buf_size = bytes;
      }
    else
      {
        status = FAIL;
        *frame_buf_size = 0;
        //*frame_buf_size = bytes;
      }

    return status;
  }



void
gr_hdlc_router_sink_b::hdlc_state_machine(unsigned char next_bit)
{
  int           next_state;
  int           i;
  unsigned char frame_buf[FRAME_BUF_MAX];
  int           frame_size;
  int           status;
  int           dlci;


  switch(d_state)
    {
      case HUNT:
        //fprintf(stderr, "State = HUNT\n");
        // Preload the first 7 bits to get things started
        d_byte = (d_byte >> 1) | (next_bit << 7);
        d_accumulated_bits++;
        if(d_accumulated_bits < 7)
          {
            next_state = HUNT;
          }
        else
          {
            next_state = IDLE;
          }
        break;

      case IDLE:
        //fprintf(stderr, "State = IDLE\n");
        d_byte = (d_byte >> 1) | (next_bit << 7);
        if(d_byte == FLAG)
          {
            // Count it and keep hunting for more flags
            d_flag_cnt++;
            d_byte = 0x00;
            d_accumulated_bits = 0;
            next_state = HUNT;
          }
        else
          {
            // A non-FLAG byte starts a frame
            // Store the bits in the bit_buf, lsb first, and 
            // change states.
            for(i=0; i<8; i++)
              {
                d_bit_buf[i] = (d_byte >> i) & 0x01;
              }
            d_bit_buf_size = 8;
            next_state = FRAMING;
          }
        break;

      case FRAMING:
        //fprintf(stderr, "State = FRAMING   bit_buf_size = %d\n", bit_buf_size);
        // Collect frame bits in bit_buf for later unstuffing
        if(d_bit_buf_size < BIT_BUF_MAX)
          {
            d_bit_buf[d_bit_buf_size] = next_bit;
            d_bit_buf_size++;
          }

        // Count consecutive 1 bits
        if(next_bit == 1)
          {
            d_consecutive_one_bits++;
          }
        else
          {
            d_consecutive_one_bits = 0;
          }

        // Test for Aborts and FLAGs
        if(d_consecutive_one_bits > 7)
          {
            // Too many 1 bits in a row. Abort.
            d_abort_cnt++;
            d_seven_ones_cnt++;
            d_byte = 0x00;
            d_accumulated_bits = 0;
            next_state = HUNT;
          }
        else
          {
            // Pack bit into byte buffer and check for FLAG
            d_byte = (d_byte >> 1) | (next_bit << 7);
            if(d_byte != FLAG)
              {
                // Keep on collecting frame bits
                next_state = FRAMING;
              }
            else 
              {
                // It's a FLAG. Frame is terminated.
                d_flag_cnt++;

                // Remove flag from bit_buf
                d_bit_buf_size -= 8;

                // Process bit_buf and
                // see if we got a good frame.
                status = unstuff(d_bit_buf_size, d_bit_buf, &frame_size, frame_buf);
                //fprintf(stderr, "  Unstuffed Frame Size = %d\n", frame_size);
                if(status == FAIL)
                  {
                    // Not an integer number of bytes.  Abort.
                    d_abort_cnt++;
                    d_non_align_cnt++;
                    //print_frame(frame_size, frame_buf);
                    //fprintf(stderr, "    NON-ALIGNED FRAME\n\n");
                    //fflush(stderr);
                  }
                else
                  {
                    // Check frame size
                    if(frame_size < 6)
                      {
                        // Too small
                        d_runt_cnt++;
                        //fprintf(stderr, "    RUNT\n\n");
                        //fflush(stderr);
                    }
                    else if(frame_size > FRAME_MAX)
                      {
                        // Too big
                        d_giant_cnt++;
                        //fprintf(stderr, "    GIANT\n\n");
                        //fflush(stderr);
                      }
                    else
                      {
                        // Size OK. Check crc
                        status = crc_valid(frame_size, frame_buf);
                        if(status == FAIL)
                          {
                            // Log crc error
                            d_crc_err_cnt++;
                            //print_frame(frame_size, frame_buf);
                            //fprintf(stderr, "    BAD CRC\n\n");
                            //fflush(stderr);
                          }
                        else
                          {
                            // Good frame! Log statistics
                            d_good_frame_cnt++;
                            d_good_byte_cnt += frame_size-2; // don't count CRC
                            //fprintf(stdout, "    Good Frame\n\n");
                            //fflush(stdout);

                            // Check for proper virtual channel (DLCI)
                            dlci = ((frame_buf[0]<<2) & 0x03F0) | 
                                   ((frame_buf[1]>>4) & 0x000F);
                            if(dlci == d_dlci)
                              {
                                // Correct channel. Log it and check for IP payload.
                                d_good_dlci_cnt++;
                                //fprintf(stdout, "    Good DLCI\n\n");
                                //fflush(stdout);
                                if( (frame_buf[2] == 0x03) &&
                                    (frame_buf[3] == 0xCC))
                                  {
                                    // It's an IP packet. Route it.
                                    //fprintf(stdout, "    IP Packet Found\n\n");
                                    //fflush(stdout);
                                    route_packet(frame_size, frame_buf);
                                  }
                              }
                          }
                      }
                  }
                // Hunt for next flag or frame
                d_byte = 0x00;
                d_accumulated_bits = 0;
                next_state = HUNT;
              }
           }     
        break;

    } // end switch

    // Save new state and return
    d_state = next_state;
}




//--------------- Protected Methods ----------------------------------

// Constructor

gr_hdlc_router_sink_b::gr_hdlc_router_sink_b(int dlci)
  : gr_sync_block ("hdlc_router_sink_b",
		   gr_make_io_signature(1, 1, sizeof(unsigned char)),
		   gr_make_io_signature(0, 0, 0)),
    d_dlci(dlci), 
    d_state(HUNT),
    d_byte(0x00),
    d_accumulated_bits(0),
    d_bit_buf_size(0),
    d_consecutive_one_bits(0),
    d_flag_cnt(0),
    d_good_frame_cnt(0),
    d_good_byte_cnt(0),
    d_good_dlci_cnt(0),
    d_crc_err_cnt(0),
    d_abort_cnt(0),
    d_seven_ones_cnt(0),
    d_non_align_cnt(0),
    d_giant_cnt(0),
    d_runt_cnt(0)
{
  //******INITIALIZATION HERE ******
}



//--------------- Friend (public constructor) ------------------------

gr_hdlc_router_sink_b_sptr
gr_make_hdlc_router_sink_b (int dlci)
{
  return gr_hdlc_router_sink_b_sptr (new gr_hdlc_router_sink_b (dlci));
}


//--------------- Public Methods --------------------------------------

// Destructor 

gr_hdlc_router_sink_b::~gr_hdlc_router_sink_b ()
{
}


int 
gr_hdlc_router_sink_b::work (int                       noutput_items,
		                     gr_vector_const_void_star &input_items,
		                     gr_vector_void_star       &output_items)
{
  unsigned char * inbuf = (unsigned char *) input_items[0];
  int             i;
  int             nwritten = 0;
  unsigned char   next_byte;
  int             bit_count;
  unsigned char   next_bit;

  // Loop thru each byte of the input stream, one bit per byte
  for(i=0; i<noutput_items; i++)
  {
    hdlc_state_machine(inbuf[i] & 0x01);  // Low order bit
    nwritten++;
  }

  return nwritten;
}
