#!/usr/bin/env python

from gnuradio import gr, gru, usrp, optfir, audio, eng_notation, blks, tait
from gnuradio.eng_option import eng_option
from optparse import OptionParser

class app_flow_graph(gr.flow_graph):
	def __init__(self, options, args):
		gr.flow_graph.__init__(self)
		self.options = options
		self.args = args
		
		sin_freq = int(options.frequency)
		#sin_freq = 2970
		ampl = 8000
		# Tests the biquad4 signal proccesing block.
		SRC = gr.sig_source_f (48000, gr.GR_SIN_WAVE, sin_freq, ampl)
		DOWN = blks.rational_resampler_fff(self, 1, 6)
		F_2_S = gr.float_to_short()
		BIQUAD4 = tait.biquad4_ss()
		S_2_F = gr.short_to_float()
		UP = blks.rational_resampler_fff(self, 6, 1)
		AUDIO = audio.sink(48000, "")
		CHECK = gr.file_sink (gr.sizeof_float, "probe.dat")
		
		self.connect(SRC, DOWN, F_2_S, BIQUAD4, S_2_F, UP, CHECK)

	
def main():
    parser = OptionParser(option_class=eng_option)
    parser.add_option("-f", "--frequency", type="eng_float", default=200,
                      help="set generated sinewave frequancy to Hz")
    (options, args) = parser.parse_args()

    fg = app_flow_graph(options, args)
    try:
        fg.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
