#!/usr/bin/env python
#
# Copyright 2004,2005,2007 Free Software Foundation, Inc.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
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


from gnuradio import gr, gru
from gnuradio import speexcodec
from gnuradio import audio,blks2
from gnuradio import wavfile
import sys
#from gnuradio import packetdrop
from gnuradio.eng_option import eng_option
from optparse import OptionParser

MIN_SNCD_RATE = 44100


class my_top_block(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self)
       
        usage="%prog: [options] outfilename"
        parser = OptionParser(option_class=eng_option,usage=usage)
   
        parser.add_option("-I", "--infilename", type="string", default="female.wav",
                          help="read input from FILE")
        parser.add_option("-r", "--sample-rate", type="eng_float", default=8000,
                          help="set sample rate to RATE (8000)")
        parser.add_option("-Q", "--quality", type="eng_float", default=8)
        #parser.add_option("-O", "--outfilename", type="string", default="femalespeex.wav",
                          #help="Write to output FILE")
        parser.add_option("-D", "--drop-rate", type="eng_float", default=0,
                          help="Specify packet drop rate")
        #parser.add_option("-A", "--audio-output", type="string", default="",
                          #help="pcm output device name.  E.g., hw:0,0 or /dev/dsp")
        
        (options, args) = parser.parse_args()
        if len(args) != 1 :
            parser.print_help()
            raise SystemExit, 1
        outfilename = args[0]
        sampling_rate = int(options.sample_rate)
        quality = int(options.quality)
        wf_in  = wavfile.wavfile_source(options.infilename)
        enc = speexcodec.speex_encoder(sampling_rate,quality,0,2,0)
        dec = speexcodec.speex_decoder(quality,0)
        wf_out = wavfile.wavfile_sink(outfilename,
				1,
				sampling_rate,
				16)

        src_scale = gr.multiply_const_ff(32767)
        sink_scale = gr.multiply_const_ff(1.0/32767.)
        packet_loss = speexcodec.packet_drop(options.drop_rate,quality,0)
        audio_out = audio.sink(sampling_rate, "hw:0,0",0)
        print "FILENAME = %s   SAMPLING RATE =  %d   QUALITY = %d  DROP RATE = %d%%" %(options.infilename,sampling_rate,quality,options.drop_rate * 100) 
        if sampling_rate < MIN_SNCD_RATE :
            out_lcm = gru.lcm(sampling_rate, MIN_SNCD_RATE)
	    out_interp = int(out_lcm // sampling_rate)
	    out_decim  = int(out_lcm // MIN_SNCD_RATE)
	    resampler = blks2.rational_resampler_fff(out_interp, out_decim)
            self.connect(wf_in,src_scale,enc,packet_loss,sink_scale,resampler,audio_out)
        else:
            self.connect(wf_in,src_scale,enc,packet_loss,sink_scale,audio_out)

               
        #self.fg.connect(wf_in,src_scale,enc,raw_file)
	#self.fg.run()
	wf_out.close()

        #src = gr.file_source (gr.sizeof_float, options.filename, options.repeat)
        #dst = audio.sink (sample_rate, options.audio_output)
        #self.connect(src, dst)


if __name__ == '__main__':
    try:
        my_top_block().run()
    except KeyboardInterrupt:
        pass
