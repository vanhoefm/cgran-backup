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

from gnuradio import gr, eng_notation, tait
from gnuradio import audio
from gnuradio import usrp
from gnuradio.eng_option import eng_option
from optparse import OptionParser

from gr_socket import *

class usrp_source_c(gr.hier_block):
    """
    Create a USRP source object supplying complex floats.
    
    Selects user supplied subdevice or chooses first available one.

    Calibration value is the offset from the tuned frequency to 
    the actual frequency.       
    """
    def __init__(self, fg, subdev_spec, decim, gain=None, calibration=0.0):
	self._decim = decim
        self._src = usrp.source_c()
        if subdev_spec is None:
            subdev_spec = usrp.pick_rx_subdevice(self._src)
        self._subdev = usrp.selected_subdev(self._src, subdev_spec)
        self._src.set_mux(usrp.determine_rx_mux_value(self._src, subdev_spec))
        self._src.set_decim_rate(self._decim)

	# If no gain specified, set to midrange
	if gain is None:
	    g = self._subdev.gain_range()
	    gain = (g[0]+g[1])/2.0

        self._subdev.set_gain(gain)
        self._cal = calibration
	gr.hier_block.__init__(self, fg, self._src, self._src)

    def tune(self, freq):
    	result = usrp.tune(self._src, 0, self._subdev, freq+self._cal)
    	# TODO: deal with residual

    def rate(self):
	return self._src.adc_rate()/self._decim

#
# return a gr.flow_graph
#
def build_graph (port, frequency):
	
	fg = gr.flow_graph ()
	
	usrp_decim = 256

        USRP = usrp_source_c(fg,		# Flow graph
			     None,		# Daugherboard spec
	                     usrp_decim)	# IF decimation ratio

	USRP.tune(frequency)
	print "USRP.rate() = ", USRP.rate()
	
	FILE_SINK = gr.file_sink (gr.sizeof_float, "check.dat")
	CX_2_F = gr.complex_to_float()
	INTERLEAVE = gr.interleave(gr.sizeof_float)
	SOCK_CONV = tait.socket_encode_fchar()
	CH_2_F = gr.char_to_float()
	
	# create socket server
	(sink, fd, conn) = make_socket_sink(port, gr.sizeof_char)
	
	# Converts a complex stream from the USRP into two float streams.
	# These real and imaginary float streams are interleaved to create a single
	# stream which is then converted to to a char stream for transmision over the
	# network.
	fg.connect (USRP, (CX_2_F, 0), (INTERLEAVE, 1))
	fg.connect ((CX_2_F, 1), (INTERLEAVE, 0))
	fg.connect (INTERLEAVE, SOCK_CONV, sink)

	
	# very important to keep fd and conn in scope of run thread
	return (fg, fd, conn)

def main ():
	usage = "usage: %prog [options]"
	parser = OptionParser(option_class=eng_option, usage=usage)
	parser.add_option("-p", "--port", type="int",
			help="spectify the port to connect to",default=8881)
	parser.add_option("-f", "--freq", type="eng_float",
			help="Set usrp freq",default=452.75e6)
	(options, args) = parser.parse_args()


    
	(fg,fd, conn) = build_graph (options.port, options.freq)
	
	fg.start ()        # fork thread(s) and return
	raw_input ('Press Enter to quit: ')
	fg.stop ()

if __name__ == '__main__':
    main ()
