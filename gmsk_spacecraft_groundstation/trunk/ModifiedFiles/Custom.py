"""
Copyright 2007 Free Software Foundation, Inc.
This file is part of GNU Radio

GNU Radio Companion is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

GNU Radio Companion is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
"""
##@package SignalBlockDefs.Custom
#Create your own custom blocks.
#@author You!

from DataTypes import *
from SignalBlockConstants import *
from gnuradio import gr

###########################################################################
#Read "creating_a_signal_block_def.txt" in the notes directory, 
#and add your own signal block definitions below:
###########################################################################

#def CustomBlockDef1(sb):
#...
#...

def HdlcRouterSink(sb):
        fcn = gr.hdlc_router_sink_b
        sb.add_input_socket('in', Byte())
        sb.add_param('DLCI', Int(17, min=0, max=1023))
        sb.set_docs('''\
HDLC router sink. \
Consumes an unpacked bitstream. Looks for bitstuffed frames delimited by \
an hdlc FLAG byte (0x7e). If the unstuffed frame passes the checksum, \
matches the correct dlci, and contains an IP packet, the \
IP packet is extracted and injected into the network stack. \
Must be run as root.''')
        return sb, lambda fg, dlci: fcn(dlci.parse())

def HdlcRouterSource(sb):
        fcn = gr.hdlc_router_source_b
        sb.add_output_socket('out', Byte())
        sb.add_param('DLCI', Int(17, min=0, max=1023))
        sb.add_param('Local Address', String('10.0.0.1'))
        sb.add_param('Remote Address', String('10.0.0.2'))
        sb.set_docs('''\
HDLC router source. \
Creates a Tun/Tap pseudo point-to-point network device with \
the specified addresses. Any IP packets that are addressed \
to this "device" are read in, encapsulated in a MultiProtocol \
over Frame Relay (MPoFR) packet and output as an unpacked \
bitstream. When no packets are available, a steady stream \
of hdlc FLAG bytes (0x7f) is output.\
Must be run as root.''')
        return sb, lambda fg, dlci, laddr, raddr: fcn(dlci.parse(), laddr.parse(), raddr.parse())

def NrziToNrz(sb):
	fcn = gr.nrzi_to_nrz_bb
	sb.add_input_socket('in', Byte())
	sb.add_output_socket('out', Byte())
	sb.add_param('Preload', Int(0, min=0, max=1))
	return sb, lambda fg, preload: fcn(preload.parse())

def NrzToNrzi(sb):
	fcn = gr.nrz_to_nrzi_bb
	sb.add_input_socket('in', Byte())
	sb.add_output_socket('out', Byte())
	sb.add_param('Preload', Int(0, min=0, max=1))
	return sb, lambda fg, preload: fcn(preload.parse())

def Randomize(sb):
	fcn = gr.randomize_bb
	sb.add_input_socket('in', Byte())
	sb.add_output_socket('out', Byte())
	sb.add_param('Tap Mask', Int((2^17)+(2^12)))
	sb.add_param('Preload', Int(0, min=0, max=1))
	return sb, lambda fg, tmask, preload: fcn(tmask.parse(), preload.parse())

def Derandomize(sb):
	fcn = gr.derandomize_bb
	sb.add_input_socket('in', Byte())
	sb.add_output_socket('out', Byte())
	sb.add_param('Tap Mask', Int((2^17) + (2^12)))
	sb.add_param('Preload', Int(0, min=0, max=1))
	return sb, lambda fg, tmask, preload: fcn(tmask.parse(), preload.parse())

def DigitalUpsampler(sb):
	fcn = gr.digital_upsampler_bb
	sb.add_input_socket('in', Byte())
	sb.add_output_socket('out', Byte())
	sb.add_param('Input Rate', Int())
	sb.add_param('Output Rate', Int())
	return sb, lambda fg, irate, orate: fcn(irate.parse(), orate.parse())



###########################################################################
#Add custom blocks to the list below, 
#and the blocks will appear under the "Custom" category:
###########################################################################

##custom block list
CUSTOM_BLOCKS = [
        #('Custom Block 1', CustomBlockDef1),
        #('Custom Block 2', CustomBlockDef2),
        ('HDLC Router Sink', HdlcRouterSink),
        ('HDLC Router Source', HdlcRouterSource),
        ('NRZI to NRZ', NrziToNrz),
        ('NRZ to NRZI', NrzToNrzi),
        ('Randomize', Randomize),
        ('Derandomize', Derandomize),
        ('Digital Upsampler', DigitalUpsampler)
]

