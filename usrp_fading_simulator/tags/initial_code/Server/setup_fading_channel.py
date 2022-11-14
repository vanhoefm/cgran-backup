#!/usr/bin/env python
from gnuradio import tait
from random import randint

def setup_fading_channel(speed, rx_freq, sample_rate):
	"""Simply sets-up a flat Rayleigh fading channel.
	
	Inputs:
	
	-speed: Speed of receiver (km/h)
	-tx_freq: The frequency that the receiver is receiving on (Hz).
	-sample_rate: The sample rate of the channel (Samples/second).
	
	Returns a GNU Radio signal processing block (that executes fading)
	with complex sample input and complex sample output.
	
	"""			
	chan_seed = randint(-10e3, 10e3) 		# For channel seed
	fd = ((speed / 3.6) * rx_freq) / 3e8 	# Doppler frequency in Hz
	T = 1.0/sample_rate
	print "fd = %f" % fd
	fdT = fd*T		
	flag_indep = False	# treats individual blocks as one continuous one.
	chan_pwr = 1.0 		# the square root power of the resulting channel wave.
	# This creates the fading channel.
	return tait.flat_rayleigh_channel_cc(chan_seed, fdT, chan_pwr, flag_indep)

