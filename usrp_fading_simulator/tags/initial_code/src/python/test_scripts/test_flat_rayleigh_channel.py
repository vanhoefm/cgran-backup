#!/usr/bin/env python

from gnuradio import gr, tait
from Numeric import *

# Test to ensure Christos Komninakis channel fading code, which is now 
# integrated into GNU Radio, produces the same results as his original 
# "runRayleigh.cpp" test bench. In other words,
# this script replicates runRayleigh.cpp.
#
# Note that the two tests produce identical results indicating that his 
# code has successfully been integrated into GNU Radio.
class my_graph(gr.flow_graph):

    def __init__(self):
        gr.flow_graph.__init__(self)

	# Creates a vector of 400,000 ones (ints). 
	vect = ones(10000000, Int)
	src = gr.vector_source_f(vect)
	# with only src connected this will produce a stream of "1 + Oi"
	f2c = gr.float_to_complex() 
	
	# Set up the fading channel:
	
	chan_seed = -115 	# Seeds the channel
	fdT = 0.002 		# fd = Doppler Fade rate, 1/T = sample rate of channel
	flag_indep = False	# treats indivdual blocks as one continous one.
	chan_pwr = 1.0 		# the square root power of the resulting channel wave.
	# This creates the fading channel.
	channel = tait.flat_rayleigh_channel_cc(chan_seed, fdT, chan_pwr, flag_indep)
	
	# Sink for the faded samples.
	dst = gr.file_sink (gr.sizeof_gr_complex, "channel.dat")
	# For checking the data source.
	data_check = gr.file_sink (gr.sizeof_gr_complex, "data.dat")

	self.connect(src, f2c, channel, dst)
	self.connect(f2c, data_check)

if __name__ == '__main__':
    try:
        my_graph().run()
    except KeyboardInterrupt:
        pass
