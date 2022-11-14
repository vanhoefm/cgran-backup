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

// Temp kludge until proper m4 configure scrpts
//  will generate correct define
#define HAVE_LINUX_TUN_TAP 1

#if !defined(HAVE_OSX_TUN_TAP) && !defined(HAVE_LINUX_TUN_TAP) 
#error Must define either HAVE_OSX_TUN_TAP or HAVE_LINUX_TUN_TAP
#endif

#include <gr_hdlc_router_source_b.h>
#include <stdio.h>
#include <gr_io_signature.h>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdexcept>
#include <cerrno>
#include <sys/ioctl.h>
#include <net/if.h>

#ifdef HAVE_LINUX_TUN_TAP
#include <linux/if_tun.h>
#endif

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


//--------------- Debug Macros ---------------------------------------

//#define DEBUG 1

#ifdef DEBUG
#define DEBUG0(s)       fprintf(stderr, s); fflush(stderr)
#define DEBUG1(s,a1)    fprintf(stderr, s, a1); fflush(stderr)
#define DEBUG2(s,a1,a2) fprintf(stderr, s, a1, a2); fflush(stderr)
#else
#define DEBUG0(s)       
#define DEBUG1(s,a1)    
#define DEBUG2(s,a1,a2) 
#endif


//--------------- Private Methods ------------------------------------


//                                      16   12   5
// this is the CCITT CRC 16 polynomial X  + X  + X  + 1.
// This is 0x1021 when x is 2,	 but the way the algorithm works
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

unsigned short gr_hdlc_router_source_b::crc16(unsigned char *data_p, unsigned short length)
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


// Push a FLAG into the bit fifo, lsb first.

void gr_hdlc_router_source_b::push_flag(void)
{
  int           i;
  unsigned char flag = FLAG;
  unsigned char bit;

  // Take care of any pending zero-stuffing
  if(d_consecutive_one_bits == 5)
    {
      d_fifo.push(0);  // Stuff a zero before the next bit
      d_consecutive_one_bits = 0;
    }

  if (d_fifo.space_left() >= 8)
    {
      for(i=0; i<8; i++)
        {
          bit = flag & 0x01;
          flag = flag >> 1;
          d_fifo.push(bit);
        }
      d_flag_cnt++;
      // Reset 1's counter
      d_consecutive_one_bits = 0;
    }
  else
    {
      fprintf(stderr, "Router Source: Fifo overflow while pushing flag.\n");
    }
}


// Take a byte and bitstuff it into the fifo, lsb first.

void gr_hdlc_router_source_b::bitstuff_byte(unsigned char byte)
{
  unsigned char bit;
  int           i;

  for(i=0; i<8; i++)  // For each bit
    {
      bit = byte & 0x01;
      byte = byte >> 1;

      if(d_consecutive_one_bits == 5)
        {
          d_fifo.push(0);  // Stuff a zero before the next bit
          d_consecutive_one_bits = 0;
        }

      d_fifo.push(bit);

      if(bit == 0)
        {
          d_consecutive_one_bits = 0;
        }
      else // Bit is '1'
        {
          d_consecutive_one_bits++;
        }
    }
  d_byte_cnt++;
}


// Take a packet and it's crc, frame them, bitstuff them, and put the bits
// into the fifo.

void gr_hdlc_router_source_b::bitstuff_and_frame_packet(unsigned char * frame_buf,
                                                        int             frame_size)
{
    int i;

    // Push Flag byte into fifo to mark frame start
    push_flag();

    // Bitstuff frame and push it into fifo
    for(i=0; i<frame_size; i++)  // For each byte of frame
      {
        bitstuff_byte(frame_buf[i]);
      }

    // Count frames
    d_frame_cnt++;
}


// Read packet from tun device. Return 0 if no packet ready.

int gr_hdlc_router_source_b::read_packet(unsigned char * packet_buf)
{
  int  byte_cnt;

  DEBUG0("read_packet: entry.\n");
  byte_cnt = read(d_tun_fd, packet_buf, FRAME_MAX);
  DEBUG1("read_packet: read completed. Result = %d\n", byte_cnt);
  if(byte_cnt < 0)
    {
      if((errno != EAGAIN) && (errno != EINTR))
        {
          perror("Router_source - Error reading tun device");
        }
      byte_cnt = 0;
    }
  return byte_cnt;
}


// Read a packet from the TUN pseudo network device, bitstuff it,
// frame it in MPoFR, and push frame's bits into the bitbuf fifo.

void gr_hdlc_router_source_b::encapsulate_incoming_packet(void)
{
  int            packet_size;
  unsigned short crc;
  unsigned char  frame_buf[FRAME_MAX];

  packet_size = read_packet(frame_buf+4);  // Allow space for MPoFR header
  if (packet_size > 0)
    {
      DEBUG1("encapsulate: Got packet, %d bytes.\n", packet_size);
      // Copy MPoFR frame header onto packet
      for(int i = 0; i<4; i++)
        {
          frame_buf[i] = d_header[i];
        }

      // Calculate frame's crc, including header
      crc = crc16(frame_buf, packet_size+4);

      // Add crc to end of frame
      frame_buf[packet_size+4] = crc >> 8;   // MS Byte first
      frame_buf[packet_size+5] = crc & 0xff; // LS Byte last
      
      // Add FLAG and do bitstuffing
      bitstuff_and_frame_packet(frame_buf, packet_size+6);
    }
}



//--------------- Protected Methods ----------------------------------

// Constructor

gr_hdlc_router_source_b::gr_hdlc_router_source_b(int dlci, char* local_addr, char* remote_addr)
  : gr_sync_block ("hdlc_router_source_b",
		   gr_make_io_signature(0, 0, 0),
		   gr_make_io_signature(1, 1, sizeof(unsigned char))),
    d_dlci(dlci),
    d_consecutive_one_bits(0),
    d_flag_cnt(0),
    d_frame_cnt(0),
    d_byte_cnt(0)
{
  DEBUG0("Constructor: entry.\n");

  // Copy string args, avoiding buffer overrun and insuring a null terminator.
  strncpy(d_local_addr, local_addr, STR_MAX);
  d_local_addr[STR_MAX-1] = '\0';
  strncpy(d_remote_addr, remote_addr, STR_MAX);
  d_remote_addr[STR_MAX-1] = '\0';

  // Pre-construct MPoFR header, using 10-bit DLCI
  d_header[0] = (d_dlci>>2) & 0xFC;  // hi-order 6 bits of DLCI into bits 7 thru 2
  d_header[1] = (d_dlci<<4) & 0xF0;  // lo-order 4 bits of DLCI into bits 7 thru 4
  d_header[2] = 0x03;
  d_header[3] = 0xCC;

  // Open the tun device.  This is a simple hack to identify
  // which way to do it until a single consistent tun/tap API gets incorporated
  // into GnuRadio.

#ifdef HAVE_LINUX_TUN_TAP
  // Open tun cloner device
  d_tun_fd = open("/dev/net/tun", O_RDWR | O_NONBLOCK, 0);
  if(d_tun_fd < 0)
    {
      fprintf(stderr, "Error: Can't open /dev/net/tun. Are you root?\n");
      exit(-1);
    }

  // set network pseudo-device name & type
  struct ifreq  ifr;
  memset(&ifr, 0, sizeof(ifr));
  ifr.ifr_flags = IFF_TUN | IFF_NO_PI;   // tun type; no extra protocol info
  strncpy(ifr.ifr_name, "tun0", IFNAMSIZ);
  if( ioctl(d_tun_fd, TUNSETIFF, (void *) &ifr) < 0 )
    {
      close(d_tun_fd);
      perror("Error: ioctl failed");
      fprintf(stderr, "Are you running as root?\n");
      exit(-1);;
  }

  // Set IP addresses
  char command[STR_MAX];
  sprintf(command, "/sbin/ifconfig tun0 %s pointopoint %s", d_local_addr, d_remote_addr);
  system(command);

  // Enable IP forwarding.
  system("/sbin/sysctl -w net.ipv4.ip_forward=1");
#endif

#ifdef HAVE_OSX_TUN_TAP
  // Open tun0 character device and tun0 pseudo network device
  d_tun_fd = open("/dev/tun0", O_RDWR | O_NONBLOCK, 0);
  if(d_tun_fd < 0)
    {
      fprintf(stderr, "Error: Can't open /dev/tun0. Are you root?\n");
      exit(-1);
    }

  // Set addresses
  char command[STR_MAX];
  sprintf(command, "/sbin/ifconfig tun0 %s %s", d_local_addr, d_remote_addr);
  system(command);

  // Enable ip forwarding.
  system("/usr/sbin/sysctl -w net.inet.ip.forwarding=1");
#endif
}


//--------------- Friend (public constructor) ------------------------

gr_hdlc_router_source_b_sptr
gr_make_hdlc_router_source_b (int dlci, char* local_adr, char* remote_addr)
{
  return gr_hdlc_router_source_b_sptr (new gr_hdlc_router_source_b (dlci, local_adr, remote_addr));
}


//--------------- Public Methods --------------------------------------

// Destructor 

gr_hdlc_router_source_b::~gr_hdlc_router_source_b ()
{
  close(d_tun_fd);

  #ifdef HAVE_OSX_TUN_TAP
    system("/usr/sbin/sysctl -w net.inet.ip.forwarding=0");
  #endif

  #ifdef HAVELINUX_TUN_TAP
    system("/sbin/sysctl -w net.ipv4.ip_forward=0");
  #endif

  fprintf(stdout, "gr_hdlc_router_source_b:\n");
  fprintf(stdout, "   Frames = %d\n", d_frame_cnt);
  fprintf(stdout, "   Bytes  = %d\n", d_byte_cnt);
  fprintf(stdout, "   Flags  = %d\n", d_flag_cnt);
  fflush(stdout);
}


int 
gr_hdlc_router_source_b::work (int                       noutput_items,
		               gr_vector_const_void_star &input_items,
		               gr_vector_void_star       &output_items)
{ 
  unsigned char * outbuf = (unsigned char *) output_items[0];
  int             i;

  DEBUG0("Work: entry.\n");

  // Loop until the requested number of output stream bytes have been generated
  for(i=0; (i<noutput_items) && (i<1000); i++)
  {
    DEBUG1("work: i = %d\n", i);
    if (d_fifo.empty())
      {
        DEBUG0("work: fifo empty, checking for packet.\n");
        // Process a single incoming packet on the tun/tap pseudo network device
        // by bitstuffing it, encapsulating it in an MPoFR frame, and pushing
        // it into the bit_buf fifo.
        encapsulate_incoming_packet();
      }

    // If bit_buf fifo is still empty, push a FLAG into it
    if(d_fifo.empty())
      {
        DEBUG0("work: fifo empty, pushing flag.\n");
        push_flag();
      }

    DEBUG1("work: %d bytes left in fifo.\n", d_fifo.space_left() );
    DEBUG1("work: Frames = %d\n", d_frame_cnt);
    DEBUG1("work: Flags  = %d\n", d_flag_cnt);
    DEBUG1("work: Bytes  = %d\n", d_byte_cnt);

    // Output a bit from the bitbuf in the low-order bit of the output byte
    outbuf[i] = d_fifo.pop();
  }

  return i;
}  
