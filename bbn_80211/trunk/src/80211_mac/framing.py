#!/usr/bin/env python
#
# Copyright 2006 Free Software Foundation, Inc.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
# 

from gnuradio import gr, gru
import struct

def seq_num_gen():
   num = -1
   while True:
      num = (num + 1) % 4096
      yield num

seq_num = seq_num_gen().next

class framed_packet(object):
    """
    Top Level class for an 802.11 framed packet

    *** Note, the packing and unpacking methods currently only deal
        with the data frame format
    """
    def __init__(self, my_mac, verbose=False):
       # lots more that could be here
       self.VERSION = 0x0000
       self.TYPE_DATA = 0x2000
       self.SUBTYPE_DATA = 0x0000
       self.TO_DS = 0x0080
       self.FROM_DS = 0x0040
       self.MORE_FRAG = 0x0020
       self.RETRY = 0x0010
       self.PWR_MGMT = 0x0008
       self.MORE_DATA = 0x0004
       self.WEP = 0x0002
       self.ORDER = 0x0001

       # some constants
       self.broadcast_addr = "\xff\xff\xff\xff\xff\xff"

       self.addr1 = "\0\0\0\0\0\0"
       self.addr2 = "\0\0\0\0\0\0"
       self.addr3 = "\0\0\0\0\0\0"
       self.addr4 = "\0\0\0\0\0\0"
       self.duration = 0
       self.version = self.VERSION
       self.type = self.TYPE_DATA
       self.subtype = self.SUBTYPE_DATA
       self.to_ds = False
       self.from_ds = False
       self.more_frag = False
       self.retry = False
       self.pwr_mgmt = False
       self.more_data = False
       self.wep = False
       self.order = False
       self.frag_num = 0
       self.seq_num = 0

       self.my_mac = my_mac
       self.payload = ''

       self.verbose = verbose

    def pack_header(self):
       """
       Packs the header

       Header fields are packed using data in the class variables.
       Returns the packed header
       """
       frame_control = self.version | self.type | self.subtype
       if self.to_ds:
          frame_control |= self.TO_DS
       if self.from_ds:
          frame_control |= self.FROM_DS
       if self.more_frag:
          frame_control |= self.MORE_FRAG
       if self.retry:
          frame_control |= self.RETRY
       if self.pwr_mgmt:
          frame_control |= self.PWR_MGMT
       if self.more_data:
          frame_control |= self.MORE_DATA
       if self.wep:
          frame_control |= self.WEP
       if self.order:
          frame_control |= self.ORDER
       sequence_ctl = (self.frag_num << 12) | self.seq_num

       header = struct.pack('HH', frame_control, self.duration) + \
                self.addr1 + self.addr2 + self.addr3 + \
                struct.pack('H', sequence_ctl)
       if self.to_ds and self.from_ds:
          header = header + self.addr4
       return header

    def pack(self):
       """
       Packs the header and payload and adds the CRC.

       Header and payload fields are packed using data in the class variables.
       Returns the packed packet
       """
       packet = self.pack_header() + self.payload
       return gru.gen_and_append_crc32(packet)

    def unpack_header(self, header):
       """
       Unpacks the header into the class variables.
       Returns the length of the header since it is variable length
       """
       # header length to return as it is variable length
       hdr_len = 30
       # first break out the various components
       (frame_control, self.duration) = struct.unpack('HH', header[0:4])
       self.addr1 = header[4:10]
       self.addr2 = header[10:16]
       self.addr3 = header[16:22]
       (sequence_ctl,) = struct.unpack('H', header[22:24])

       self.to_ds = frame_control & self.TO_DS
       self.from_ds = frame_control & self.FROM_DS

       if self.to_ds and self.from_ds:
          self.addr4 = header[22:28]
       else:
          hdr_len -= 6

       # now pull out the information from each piece
       self.version = (frame_control >> 14) & 0x03
       self.type = (frame_control >> 12) & 0x03
       self.subtype = (frame_control >> 8) & 0x0F

       self.more_frag = frame_control & self.MORE_FRAG
       self.retry = frame_control & self.RETRY
       self.pwr_mgmt = frame_control & self.PWR_MGMT
       self.more_data = frame_control & self.MORE_DATA
       self.wep = frame_control & self.WEP
       self.order = frame_control & self.ORDER

       self.frag_num = (sequence_ctl >> 12) & 0x0F
       self.seq_num = sequence_ctl & 0x0FFF
     
       return hdr_len

    def unpack(self, packet):
       """
       Checks the CRC and unpacks the header and payload.

       Header and payload fields are unpacked into the class variables.
       Returns True if unpacking is successful, False if there are any
          errors or the CRC check fails
       """
       (crc_ok, checked_pkt) = gru.check_crc32(packet)
       if not crc_ok:
          if self.verbose:
             print "CRC check failed."
          return False
       hdr_len = self.unpack_header(checked_pkt)
       self.payload = checked_pkt[hdr_len:]
       return True


class packet_sender(object):
    """
    This class sends packets with an 802.11 header
 
    This class turns a given payload into packets and sends them.
    It deals with fragmentation. 
    """

    def __init__(self, pkttype, my_mac, bssid, send_pkt, cache, verbose=False):
        """
	@param pkttype "ip" for IP packets, "eth" for ethernet. Others may
            be implemented as needed.
        @param my_mac The MAC address for this radio
        @param send_pkt  The function to call to send the assembled packet.
            This assumes all packets are sent using the same function.
        """
	self.pkttype = pkttype
        self.my_mac = my_mac
        self.bssid = bssid
        self.send_pkt = send_pkt
        self.cache = cache
	self.verbose = verbose

    def send(self, payload):
        """
        Send the payload
        """
        sequence_num = seq_num()

	if self.pkttype == "ip":
	   # If it's "tun" we have an IP packet
	   # Get IP addr
           (vhl,) = struct.unpack("B", payload[:1])
           vers = (vhl >> 4) & 0xf
	   dstipaddr = "\0\0\0\0"
           if vers == 0x4:
	      dstipaddr = payload[16:20]
           else:
              if vers == 0x6:
                 dstipaddr = payload[24:40]
	   dest_mac = self.cache.get_mac_addr(dstipaddr)
           if dest_mac == None:
              if self.verbose:
                 print "Unknown destination address: ", dstipaddr 
              return
        else:
           # otherwise it's "eth" and an ethernet packet
           dest_mac = payload[0:6];
           self.my_mac = payload[6:12];
    
        MAX_PAYLOAD = 2312
        pl_len = len(payload)

        # If payload is not longer than the maximum, then there's no
        # need to fragment.  This is just a subset of the fragmentation
        # case, but saves some work in the common case of no fragmentation 
	if pl_len <= MAX_PAYLOAD:
           packet = framed_packet(self.my_mac, self.verbose)
           packet.payload = payload
           packet.addr1 = dest_mac
           packet.addr2 = self.my_mac
           packet.addr3 = self.bssid
           packet.duration = 0x8000
           packet.seq_num = sequence_num
          
           pkt = packet.pack()
           self.send_pkt(pkt)
        else:
           pkt_start = 0
           pkt_end = MAX_PAYLOAD
           frag_num = 0

           # build each fragment.
           # Each fragment has the same sequence number, and an
           # incrementing fragment number.  Need to set more_frag
           # flag on all but the last fragment.
           while pkt_start < pl_len:
              fragment = payload[pkt_start:pkt_end]
              packet = framed_packet(self.my_mac, verbose)
              packet.payload = fragment
              packet.addr1 = dest_mac
              packet.addr2 = self.my_mac
              packet.addr3 = self.bssid
              packet.duration = 0x8000
              packet.seq_num = sequence_num
              packet.frag_num = frag_num
              if pl_len > pkt_end:
                 packet.more_frag = True
          
              pkt = packet.pack()
              self.send_pkt(pkt)

              # advance pointers and frag count
              frag_num += 1
              pkt_start = pkt_end
              pkt_end += MAX_PAYLOAD
              
              # end fragment while 


class packet_assembler(object):
    """
    This class helps with packet reassembly
 
    """
    def __init__(self, verbose=False):
        self.packets = list()
        self.MAX_PACKETS = 16
        # sixteen slots for messages, one for the number of fragments received
        # and one for the number of fragments that is filled in when the 
        # fragment without a more_frags flay is received
        self.blank_list = [None, None, None, None, None, None, None, None, None, \
                           None, None, None, None, None, None, None, 0, None]
	self.count_index = 16
	self.numfrags_index = 17
        self.packets = list()
	self.verbose = verbose

    def assemble_fragment(self, seq_num, frag_num, more_frag, fragment):
        """
        Assembles the fragment

        Returns the assembled packet if it is complete, False otherwise
        """
        pkt_list = get_packet_list(seq_num)
        # make sure this fragment isn't already here
        if pkt_list[frag_num] != None:
           return False

        # insert fragment into fragment list
        pkt_list[frag_num] = fragment
        
        # increment counter of number of fragments received
        pkt_list[self.count_index] += 1

        # if more_frag is not set, this is the last fragment, so we
        # know the total number of fragments to expect
        if not more_frag:
           pkt_list[self.numfrags_index] = frag_num + 1

        # If we've received all the fragments -- we've received the 
        # last fragment (numfrags_index is filled) and all the fragments
        # have been received (count == num frags expected) -- then we
        # assemble the full message and return it.  Otherwise, return False.
        if (pkt_list[self.numfrags_index] == None) or \
           (pkt_list[self.count_index] != pkt_list[self.numfrags_index]):
           return False

        try: 
           # Assemble the message
           msg = ''
           for frag in pkt_list[:pkt_list[self.numfrags_index]]:
              msg += frag

           # We now have a packet.  Remove the packet list and return
           # the completed packet.
           remove_packet_list(seq_num)
           return msg

        except TypeError:
           # if we get this, there is something wrong with the list of 
           # fragments, so just purge it
           remove_packet_list(seq_num)
           return False

    def get_packet_list(self, seq_num):
	# find and return the fragment list associated with seq_num
	for plist in self.packets:
           if plist[0] == seq_num:
              return plist[1]

        # if we are here, it doesn't exist, yet, so create it.  If we
        # already have the maximum number of packets around, purge one, first.
        if len(self.packets) == self.MAX_PACKETS:
           purge_packet_from_list()

        new = [seq_num, list(self.blank_list)]
	self.packets.append(new)
	return new[1]

    def purge_packet_from_list(self):
	# for the time being, just pull the first from the list as
        # it is the oldest and most likely to be stale.
        del self.packets[0]

    def remove_packet_list(self, seq_num):
	for plist in self.packets:
           if plist[0] == seq_num:
              self.packets.remove(plist)


class packet_receiver(object):
    """
    This class receives packets with an 802.11 header
 
    This class extracts a payload from one or more packets and 
    hands them off to a receiving funciton. It deals with reassembly. 
    """
    def __init__(self, my_mac, bssid, rcv_pkt, verbose=False):
        """
        @param my_mac The MAC address for this radio
        @param rcv_pkt  The function to call to send the received payload.
            This assumes all payloads are handled the same way.
        """
        self.my_mac = my_mac
        self.bssid = bssid
        self.rcv_pkt = rcv_pkt
        self.assembler = packet_assembler()
	self.verbose = verbose

    def receive(self, packet):
        """
        Receive the packet

	Takes the packet, reassembles if necessary and when a complete
        payload exists, then send it to the rcv_pkt function
        """
        message = framed_packet(self.my_mac)
        if not message.unpack(packet):
           if self.verbose:
              print "Bad packet received, dropping."
           return

        # check that the packet is meant for us. If it isn't, drop it
        # check if it's to my_mac, broadcast or multicast
	if (message.addr3 != self.bssid):
           if self.verbose:
              print "Packet not for this bssid."
           return

	if (message.addr1 != self.my_mac) and \
           (message.addr1 != message.broadcast_addr) and \
           (message.addr1[0] != "\x01"):
           if self.verbose:
              print "Packet not for this host."
	   return 

	# if message.addr1[0] == "\x01":
	# *** still need to handle the multicast case, though higher
        #     levels should be able to deal

        # If this is the entire packet, just receive it and go on.
        if not message.more_frag and (message.frag_num == 0):
           if self.verbose:
              print "Full message received."
           self.rcv_pkt(message.payload)
           return

        # If we are here, the packet is a fragment and we need to 
        # reassemble it.
        pkt = self.assembler.assemble_fragment(message.seq_num, message.frag_num,\
                                               message.more_frag, message.payload)

        # If a packet was returned, then we send it to the receive function
        if pkt:
           if self.verbose:
              print "All fragments received."
           self.rcv_pkt(message.payload)
