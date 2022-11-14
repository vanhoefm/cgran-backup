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

# Sources complex samples from the network, applies wide band fm demodulation and 
# then sinks the samples to an audio sink.
# Use the complementary wfm_sock_server script to transmit theses samples.

from gnuradio import gr
from gnuradio import audio
from gnuradio import blks
from gnuradio.eng_option import eng_option
from optparse import OptionParser

from gr_socket import *

#
# return a gr.flow_graph
#
def build_graph (address, port):
	quad_rate = 250e3
	audio_decimation = 8
	audio_rate = quad_rate / audio_decimation 

	fg = gr.flow_graph ()
	
	# create socket client
	(src, fd) = make_socket_source(address, 		
								port,
								gr.sizeof_gr_complex)

	# common body of wfm receiver
	guts = blks.wfm_rcv (fg, quad_rate, audio_decimation)

	# sound card as final sink
	audio_sink = audio.sink (int (audio_rate))

	# now wire it all together
	fg.connect (src, guts, audio_sink)

	return (fg,fd)

def main ():
	usage = "usage: %prog [options]"
	parser = OptionParser(option_class=eng_option, usage=usage)
	parser.add_option("-i", "--address", type="string",
			help="specify the ip address to connect to",default="localhost")
	parser.add_option("-p", "--port", type="int",
			help="specify the port to connect to",default=8881)
	(options, args) = parser.parse_args()
	
	(fg,fd) = build_graph (options.address, options.port)

	fg.start ()        # fork thread(s) and return
	raw_input ('Press Enter to quit: ')
	fg.stop ()

if __name__ == '__main__':
	main ()
