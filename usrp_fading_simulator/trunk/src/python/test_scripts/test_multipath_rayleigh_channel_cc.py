#!/usr/bin/env python

from gnuradio import gr
from multipath_rayleigh_channel_cc import *

# Tests the multipath Rayleigh fadding implementation.

class my_graph(gr.flow_graph):

    def __init__(self):
        gr.flow_graph.__init__(self)

	# A sample signal
	#sig_vec = [complex(0,0), complex(1,0), complex(-1,0)]
	
	
	# If true then the src becomes a file source.
	if 0:
		filename = "c4fm_452.75Mhz_256ksps_30secs_shorts_-30dBm.dat"
		src = gr.file_source(gr.sizeof_gr_complex, filename)
	# src becomes a vector source
	else:
		sig_vec = [];	
		# Zero pad the signal
		zero_padding = 400000
		for i in range (zero_padding):
			sig_vec.append(complex(1,0))
	
		# Creates the signal source.
		src = gr.vector_source_c(sig_vec)
	
	# Set up the parameters for the multipath fading channel:
	
	vehicle_speed = 100 # the mobile speed (km/h)
	carrier_freq = 552.96e6 # transmiter carrier frequance (Hz)
	chan_rate = 1e6 # channel sample rate (Samples/sec)
	chan_seed = -115 	# Seeds the channel
	chan_pwrs = (1, 1, 1, 1, 1 )	# The square root power of each resulting channel wave.
	path_delays = (0, 1, 2, 3, 4)	# The delays for each path (us).
	flag_indep = False	# Treats individual blocks as one continuous one.
	flag_norm = True # Normalizes the channel output power. Thus chan_pwrs are relative only.
	
	# This creates the fading channel based on the above parameters.
	chan = 	multipath_rayleigh_channel_cc(self, src, vehicle_speed, carrier_freq,
	chan_rate, chan_seed, chan_pwrs, path_delays, flag_indep, flag_norm)
	
	# File sinks are added to check the results.
	#dst1 = gr.file_sink(gr.sizeof_gr_complex, "normalised_faded_signal.dat") 
	dst1 = gr.file_sink(gr.sizeof_gr_complex, "chan.dat") 
	dst2 = gr.file_sink(gr.sizeof_gr_complex, "signal.dat") 
	
	# Note that src has already been connected by the multipath_rayleigh_channel_cc() call.
	# This is a temporary work around. Ideally one should be 
	# able to write "self.connect(src, chan, dst1)" and not need to pass src 
	# to multipath_rayleigh_channel_cc().
	self.connect(chan, dst1)
	self.connect(src, dst2)


if __name__ == '__main__':
    try:
        my_graph().run()
    except KeyboardInterrupt:
        pass
