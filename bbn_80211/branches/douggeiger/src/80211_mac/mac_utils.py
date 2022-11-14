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

#import re 
import struct
import socket
import os
import platform

def get_mac_for_interface(iface):
    """
    Finds the MAC address for the current interface, if it is 
    there to be found.

    This function is very platform dependend and only the NetBSD
    version is tested.  

    A zero'd MAC address is returned if an address could not be 
    found or an unsupported platform is used.
    """
    mac = "00:00:00:00:00:00"
    if platform.system() == 'NetBSD':
        for line in os.popen("/sbin/ifconfig %s" % (iface)):
            if line.find('address:') > -1:
                mac = line.split(": ")[1]
                return mac[:-1]
    if platform.system() == 'Linux':
        for line in os.popen("/sbin/ifconfig %s" % (iface)):
            if line.find('Ether') > -1:
                mac = line.split("Ether ")[1]
                return mac[:-1]
    if platform.system() == 'FreeBSD':
        for line in os.popen("/sbin/ifconfig %s" % (iface)):
            if line.find('ether') > -1:
                mac = line.split("ether ")[1]
                return mac[:-1]
    return mac


""" if line.find('Ether') > -1: """

def validate_mac_addr(mac_string):
    """
    Checks validity of the the string representation of a MAC address
    and converts it to an integer

    Returns The integer representation of the MAC if it is valid, 0 if
	it is invalid.
    """
#    try:
#	mac_string = re.sub('[:.]', '', mac_string)
#	if len(mac_string) != 12:
#            return 0
#        return(long(mac_string, 16))
#    except ValueError:
#        return 0
    try:
	mac_bytes = mac_string.split(':')
	if len(mac_bytes) != 6:
            return 0
        return(struct.pack('BBBBBB', int(mac_bytes[0], 16),
	                      int(mac_bytes[1], 16),
	                      int(mac_bytes[2], 16),
	                      int(mac_bytes[3], 16),
	                      int(mac_bytes[4], 16),
	                      int(mac_bytes[5], 16)))
    except ValueError:
        return 0

def get_mac_string(mac_addr):
    """
    Converts the integer representation of a MAC address to its string 
    representation.

    Returns The string representation of mac_addr
    """
#    return "%02x:%02x:%02x:%02x:%02x:%02x" % (0xff & (mac_addr >> 40),
#	  			              0xff & (mac_addr >> 32),
#				              0xff & (mac_addr >> 24),
#				              0xff & (mac_addr >> 16),
#				              0xff & (mac_addr >> 8),
#				              0xff & mac_addr)
    return "%02x:%02x:%02x:%02x:%02x:%02x" % struct.unpack('BBBBBB', mac_addr)


class mac_cache(object):
    """
    This class holds IP to MAC address mappings
 
    """
    def __init__(self):
        # list of [IP, MAC] tuples
        self.mapping = list()

        # include broadcast mapping
	self.mapping.append(["\xff\xff\xff\xff", "\xff\xff\xff\xff\xff\xff"])

    def add(self, ipaddr, macaddr):
        """
        Adds and ipaddr->macaddr mapping to the cache
        """
        self.mapping.append([ipaddr, macaddr])

    def get_mac_addr(self, ipaddr):
        """
        Return MAC address that maps to the IP address provided.  Returns
        None, if the mapping does not exist
        """
	for map in self.mapping:
	   if map[0] == ipaddr:
              return map[1]
        return None

    def print_map(self):
        for map in self.mapping:
	   print socket.inet_ntoa(map[0]), "\t=> ", get_mac_string(map[1])
