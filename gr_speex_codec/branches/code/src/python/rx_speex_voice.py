#!/usr/bin/env python
#
# Copyright 2005,2006 Free Software Foundation, Inc.
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
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from gnuradio import gr, gru, modulation_utils
from gnuradio import usrp
from gnuradio import audio
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser

from gnuradio import speexcodec
from gnuradio import blks2
from gnuradio import wavfile
import sys

#from gnuradio import packetdrop

from gnuradio.vocoder import gsm_full_rate

import random
import struct

# from current dir
from receive_path import receive_path
import fusb_options



class audio_tx(gr.hier_block2):
    def __init__(self,audio_output_dev,sample_rate,quality,outfilename,spk):
        speex_enc_frame = [6,10,15,20,20,28,28,38,38,46,62]
 
        gr.hier_block2.__init__(self, "audio_tx",
				gr.io_signature(0, 0, 0), # Input signature
				gr.io_signature(0, 0, 0)) # Output signature


        qual = int(quality)
        speex_frame = speex_enc_frame[qual]
        print speex_frame
        self.packet_src = gr.message_source(speex_frame)
        sampling_rate = int(sample_rate)
        dec = speexcodec.speex_decoder(qual,0)
        packet_loss = speexcodec.packet_drop(0,qual,0)

        sink_scale = gr.multiply_const_ff(1.0/32767.)
        if spk:
            audio_sink = audio.sink(sampling_rate,"plughw:0,0")
        else:
            audio_sink = wavfile.wavfile_sink(outfilename,
				1,
				sampling_rate,
				16)

        self.connect(self.packet_src, packet_loss,sink_scale, audio_sink)

        
    def msgq(self):
        return self.packet_src.msgq()


class my_top_block(gr.top_block):

    def __init__(self, demod_class, rx_callback, options):
        gr.top_block.__init__(self)
        self.rxpath = receive_path(demod_class, rx_callback, options)
        self.audio_tx = audio_tx(options.audio_output,options.sample_rate,options.quality,options.outfilename,options.spk)
        self.connect(self.rxpath)
	self.connect(self.audio_tx)        



# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

global n_rcvd, n_right

def main():
    global n_rcvd, n_right

    n_rcvd = 0
    n_right = 0
    
    def rx_callback(ok, payload):
        print 'kshama: callback'

        global n_rcvd, n_right
        n_rcvd += 1
        if ok:
            n_right += 1
        tb.audio_tx.msgq().insert_tail(gr.message_from_string(payload))
        
        print "ok = %r  n_rcvd = %4d  n_right = %4d" % (
            ok, n_rcvd, n_right)
        print "received packet length = %2d"%(len(payload))

    demods = modulation_utils.type_1_demods()

    # Create Options Parser:
    parser = OptionParser (option_class=eng_option, conflict_handler="resolve")
    expert_grp = parser.add_option_group("Expert")

    parser.add_option("-m", "--modulation", type="choice", choices=demods.keys(), 
                      default='gmsk',
                      help="Select modulation from: %s [default=%%default]"
                            % (', '.join(demods.keys()),))
    parser.add_option("-O", "--audio-output", type="string", default="",
                      help="pcm output device name.  E.g., hw:0,0 or /dev/dsp")
    
    parser.add_option("-p", "--sample-rate", type="eng_float", default=8000,
                          help="set sample rate to RATE (8000)")
    parser.add_option("-Q", "--quality", type="eng_float", default=8)
    parser.add_option("-n", "--outfilename", type="string", default="speex.wav",
                          help="read input from FILE")
    parser.add_option("","--spk", action="store_true", default=False,
                      help="Choose between speaker/wav file output")





    receive_path.add_options(parser, expert_grp)

    for mod in demods.values():
        mod.add_options(expert_grp)

    fusb_options.add_options(expert_grp)

    #parser.set_defaults(bitrate=50e3)  # override default bitrate default
    (options, args) = parser.parse_args ()

    import sys

    if len(args) != 0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if options.rx_freq is None:
        sys.stderr.write("You must specify -f FREQ or --freq FREQ\n")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # build the graph
    tb = my_top_block(demods[options.modulation], rx_callback, options)

    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
        print "Warning: Failed to enable realtime scheduling."

    
    #tb.run()
   
    tb.start()        # start flow graph
    tb.wait()         # wait for it to finish

if __name__ == '__main__':
    #try:
        main()
    #except KeyboardInterrupt:
        #pass

