#!/usr/bin/env python

from gnuradio import gr, tait, eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser


# Sources samples from a ".dat" file assumed to contain binary encoded complex 
# base-band samples. These samples are then passed through the Rayleigh 
# fading channel, before being sunk to a results ".dat" file. This has been 
# created to benchmark the statistical performance of the USRPF simulator (which
# incorporates the same fading algorithm). Refer to the accompanying Matlab files. 
#
# The source file should have a sample rate of 192 kS/s. With fd set to 0.0002,  
# RF frequency equal to 440.1 MHz the receiver is traveling at 94 km/h.
class benchmark_USRPF_run_fading(gr.flow_graph):

	def __init__(self):
		gr.flow_graph.__init__(self)
		
		parser=OptionParser(option_class=eng_option)
		parser.add_option("-I", "--in-file-name", type="string", default="c4fm1011test.dat",
						help="Name of the source file.  E.g. data.dat")
		parser.add_option("-O", "--out-file-name", type="string", default="faded_c4fm1011test.dat",
						help="Name of file to capture USRP output.  E.g. data.dat")

		(options, args) = parser.parse_args()
		if len(args) != 0:
			parser.print_help()
			sys.exit(1)

		# The file source. 
		src = gr.file_source(gr.sizeof_gr_complex, options.in_file_name)
		# The file sink. 
		snk = gr.file_sink(gr.sizeof_gr_complex, options.out_file_name)
		
		# Set up flat Rayleigh Fading channel.
		#------------------------------------------------------------------
		chan_seed = -115 	# Seeds the channel
		fdT = 0.0002 		    # fd = Doppler Fade rate, T = 1/sample rate of channel
		flag_indep = False	# treats individual blocks as one continuous one.
		chan_pwr = 1.0 		# the square root power of the resulting channel wave.
		# This creates the fading channel.
		fade_channel = tait.flat_rayleigh_channel_cc(chan_seed, fdT, chan_pwr, flag_indep)
		#------------------------------------------------------------------
		

		self.connect(src, fade_channel, snk)
		
if __name__ == '__main__':
	fg = benchmark_USRPF_run_fading()
	try:
		fg.run()
	except KeyboardInterrupt:
		pass
