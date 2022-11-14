#!/usr/bin/env python
#
# Copyright 2004,2005,2007,2008 Free Software Foundation, Inc.
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

from gnuradio import gr
from gnuradio import audio
from gnuradio.eng_option import eng_option
from optparse import OptionParser

class my_top_block(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self)

        parser = OptionParser(option_class=eng_option)
        parser.add_option("-O", "--audio-output", type="string", default="",
                          help="pcm output device name.  E.g., hw:0,0 or /dev/dsp")
        parser.add_option("-r", "--sample-rate", type="eng_float", default=48000,
                          help="set sample rate to RATE (48000)")
        parser.add_option("-D", "--dont-block", action="store_false", default=True,
                          dest="ok_to_block")
        (options, args) = parser.parse_args()
        if len(args) != 0:
            parser.print_help()
            raise SystemExit, 1
        sample_rate = int(options.sample_rate)
        self.src = gr.sig_source_f(sample_rate,
                                   gr.GR_SIN_WAVE,
                                   1000.0,
                                   0.1)
        self.dst = audio.sink(sample_rate,
                              options.audio_output,
                              options.ok_to_block)
        self.connect(self.src, (self.dst, 0))

    def get_ampl(self):
        return self.src.amplitude()

    def set_ampl(self, ampl = 0.1):
        self.src.set_amplitude(ampl)

    def get_freq(self):
        return self.src.frequency()

    def set_freq(self, freq = 1000.0):
        self.src.set_frequency(freq)

if __name__ == '__main__':
    try:
        my_top_block().run()
    except KeyboardInterrupt:
        pass
