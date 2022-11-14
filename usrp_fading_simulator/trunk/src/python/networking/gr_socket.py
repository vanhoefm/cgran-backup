#!/usr/bin/env python
#
# Copyright 2005 Free Software Foundation, Inc.
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

from gnuradio import gr
import socket

#
# The client sources samples from the network.
#

def make_socket_source(addr, port, data_size):
	
	# Create socket
	fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	
	# make socket
	fd.connect((addr,port))
	print "Connection To: ", addr, " ", port
	
	# setup file src
	filesrc = gr.file_descriptor_source(data_size, fd.fileno())
	
	return (filesrc, fd)

#
# The server sinks samples into the network
#

def make_socket_sink(port, data_size):
	
	# Create socket
	fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	
	# Ensure you can restart server quickly when it terminates
	fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	
	print "Waiting for connection on port: ", port
	
	# set the client sock's TCP num
	fd.bind(('0.0.0.0',port))
	fd.listen(1)
	conn, connaddr = fd.accept()
	print "Connection from Address: ", connaddr
	
	# connect to a file descriptor
	filesink = gr.file_descriptor_sink(data_size, conn.fileno())
	# return 
	return (filesink, fd, conn)
    

    
