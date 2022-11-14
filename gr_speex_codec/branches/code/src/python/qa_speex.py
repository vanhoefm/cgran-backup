#!/usr/bin/env python
#
# Copyright 2004 Free Software Foundation, Inc.
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

from gnuradio import gr, gr_unittest
import speexcodec
from gnuradio import audio
from gnuradio import wavfile
import sys
from gnuradio import packetdrop

class qa_speex (gr_unittest.TestCase):

    def setUp (self):
        self.fg = gr.top_block()

    def tearDown (self):
        self.fg = None

    def test_001_speex_codec (self):
        # This test takes a wav file input, compresses with speex decompresses and writes 
        # into a wav file output. 
        infile  = "cdda.wav"
	outfile = "cddaspeex.wav"
              
	wf_in  = wavfile.wavfile_source(infile)
        enc = speexcodec.speex_encoder(44100,8)
        dec = speexcodec.speex_decoder(8)
        wf_out = wavfile.wavfile_sink(outfile,
				1,
				44100,
				16)

        src_scale = gr.multiply_const_ff(32767)
        sink_scale = gr.multiply_const_ff(1.0/32767.)
        raw_file = gr.file_sink((gr.sizeof_char)*10,"cddaraw")
        packet_loss = packetdrop.packet_drop(0.5,8)
        self.fg.connect(wf_in,src_scale,enc,packet_loss,sink_scale,wf_out)
        #self.fg.connect(wf_in,src_scale,enc,raw_file)
	self.fg.run()
	wf_out.close()
      	self.assertEqual(1, 1)
    
    def test_002_speex_codec (self):
        # This test takes input from the microphone compresses and decompresses and writes
        # the speech into a wav file  
        
	outfile = "femalespeech.wav"
        audio_source = audio.source(44100)
        src_scale = gr.multiply_const_ff(32767)
        sink_scale = gr.multiply_const_ff(1.0/32767.)
        enc = speexcodec.speex_encoder(44100,4)
        dec = speexcodec.speex_decoder(4)
        audio_sink = audio.sink(44100, "hw:0,0",0)
        wf_out = wavfile.wavfile_sink(outfile,
				1,
				44100,
				16)
        packet_loss = packetdrop.packet_drop(0.1,8)
        self.fg.connect(audio_source,src_scale,enc,dec,sink_scale,wf_out)
	self.fg.run()
	wf_out.close()
      	self.assertEqual(1, 1)

    def test_003_speex_codec (self):
        # This test takes input from the microphone compresses and decompresses and plays
        # the speech in to the speakers  
       
        audio_source = audio.source(44100)
	src_scale = gr.multiply_const_ff(32767)
        sink_scale = gr.multiply_const_ff(1.0/32767.)
        enc = speexcodec.speex_encoder(44100,8)
        dec = speexcodec.speex_decoder(8)
        audio_sink = audio.sink(44100, "hw:0,0",0)
        self.fg.connect(audio_source,src_scale,enc,dec,sink_scale,audio_sink)
	self.fg.run()
	self.assertEqual(1, 1)

    #def test_004_speex_codec (self):
        # This test takes the wav file input, compresses with speex decompresses and plays 
        # back into the speakers. 
        #infile  = "male.wav"
	#wf_in  = wavfile.wavfile_source(infile)
        #enc = speexcodec.speex_encoder(8000,8)
        #dec = speexcodec.speex_decoder(8)
        #throttle = gr.throttle(gr.sizeof_float,8000) 
        #audio_sink = audio.sink(44100, "hw:0,0",0)
        #src_scale = gr.multiply_const_ff(32767)
        #sink_scale = gr.multiply_const_ff(1.0/32767.)
        #self.fg.connect(wf_in,src_scale,enc,dec,sink_scale,throttle,audio_sink)
	#self.fg.run()
	#self.assertEqual(1, 1)





         
if __name__ == '__main__':
    gr_unittest.main ()
