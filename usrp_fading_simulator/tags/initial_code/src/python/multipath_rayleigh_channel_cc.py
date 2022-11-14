#!/usr/bin/env python

from gnuradio import gr, tait, blks
from math import sqrt
from Numeric import add
import sys 

class multipath_rayleigh_channel_cc(gr.hier_block):
	"""
	Applies multipath Rayleigh fadding to a complex stream input.
	
	@param fg: 			flowgraph
	@param src: 		Block which forms the head of the multipath channel.
						This is a temporary work around. Ideally this head block would not need
						to be passed to this class.
	@param vehicle_speed: 	This is the speed at which the vehicle is travling in km/h.
	@type vehicle_speed:	float
	@param carrier_freq: 	Transmission carrier frequancy in Hz.
	@type carrier_freq: 	float
	@param chan_rate: 	The sample rate at which the channel is operating (Samples/second).
	@type chan_rate: 	float
	@param chan_seed:  	Used to seed the channels. Each new channel is seeded with chan_seed++.
	@type chan_seed: 	interger
	@param chan_pwrs:	Array containing the square root power of the fading waveform for each path
						(e.g. pwr = 5 would produce a fading waveform with an 
						output power of 25).
	@type chan_pwrs: 	array of floats
	@param path_delays:	Array of path delays in micro-seconds. Note the delay time resolution is
						dependent on the sample rate of the channel. For example, a channel operating at 
						256k samples/second, delays will be nearest interger multiples of 0.256.
	@type path_delays: 	array of ints
	@param flag_indep: 	Determines whether individual blocks processed by the channel should 
						be treated as independent blocks or as one continuous block.
						1 if blocks are independent, 0 if continuous across blocks.
						By default blocks are not treated independently.
	@type flag_indep: 	bool
	@param flag_norm: 	Determines whether the total output power should be normalized.
						If this is true then the values of chan_pwrs are relative only.
						By default the channel is normalized.
	@type flag_norm: 	bool
	"""
	def __init__(self, fg, src, vehicle_speed, carrier_freq, chan_rate, chan_seed, \
		chan_pwrs, path_delays, flag_indep = False, flag_norm = True):
		
		# Checks that there is the same number of delays as there are powers.
		if len(chan_pwrs) != len(path_delays):
			raise ValueError, "The vector length of chan_pwrs does not match the vector length of path_delays."
			# Could this be improved?
			sys.exit(1)
		
		# Speed of light in km/h
		speed_of_light = 3e8 * 3.6
		# vehicle_speed (km/h), carrier_freq (Hz) and speed_of_light (km/h)
		# units: (km/h * Hz) / (km/h) = Hz. 
		fd = (vehicle_speed * carrier_freq) / speed_of_light
		# 'T' is the inverse channel rate.
		self.fdT = fd * (1.0 / chan_rate)
		# Testing only
		print "self.fdT = ", self.fdT
		
		self.c2f_blks = [] 		# for list of gr.complex_to_float().
		self.delay_blks = [] 	# for list of gr.filter_delay_fc ().
		self.chan_blks = [] 	# for list of tait.flat_rayleigh_channel_cc().
		
		self.chan_pwrs = []		# local copy of chan_pwrs (in case normalization is required)
		# stores the path delays translated from micro-seconds to 
		# equivalent delays in samples (based on the channel rate)
		self.path_delay_samples = [] 
		
		# channel rate in samples/micro-seconds
		chan_rate_us = chan_rate * 1e-6
		
		# Warn the user of the limited channel delay resolution
		if chan_rate_us < 1:
			print "Warning: at a channel rate of ", chan_rate, \
			" samples/s the delay resolution is ", 1.0/chan_rate, "seconds"
		
		# Translate the micro-second delays into the appropriate sample delays.
		for i in range(len(path_delays)):
			# path_delays[] are delays in micro-seconds and chan_rate_us is the 
			# channel delay resolution in micro-seconds.
			# units: us*(samples/us) = samples
			self.path_delay_samples.append(int (round(path_delays[i] * chan_rate_us)))
			
		# For testing only	
		print "path delays in samples ", self.path_delay_samples
		
		# Normalizes the channel powers.
		if flag_norm is True:
			# add.reduce(chan_pwrs) is equivalent to the sum of the chan_pwrs array.
			# Need the square root since the channel squares the power.
			denom = sqrt(add.reduce(chan_pwrs))
			for i in range(len(chan_pwrs)):
				self.chan_pwrs.append((sqrt(chan_pwrs[i]))/denom)
		# Normalization is not required.
		else:
			self.chan_pwrs = chan_pwrs
		
		# Populate the lists above with the correct number of blocks.
		for i in range (len(chan_pwrs)):
			
			# Delay block is required.
			if self.path_delay_samples[i] > 0:
				
				# Since the delay introduced by gr.firdes_hilbert(ntaps)
				# is equal to (ntaps - 1) / 2. 
				ntaps = (self.path_delay_samples[i] * 2) + 1
				taps = gr.firdes_hilbert (ntaps)
				self.delay_blks.append(gr.filter_delay_fc (taps))
				
				# Each delay requires a conversion.
				self.c2f_blks.append(gr.complex_to_float())
				
			else:
				# No delay or conversion required.
				self.delay_blks.append(None)
				self.c2f_blks.append(None)
				
			self.chan_blks.append(tait.flat_rayleigh_channel_cc (chan_seed + i, self.fdT, self.chan_pwrs[i], flag_indep))
		
		
		self.sum = gr.add_cc ()
		
		# Create multiple instances of the "src -> delay -> channel" connection.
		for i in range (len(chan_pwrs)):
			# No conversion block and delay block required.
			if self.delay_blks[i] is None:
				fg.connect(src, self.chan_blks[i], (self.sum, i))
			else:
				fg.connect(src, self.c2f_blks[i], self.delay_blks[i], self.chan_blks[i], (self.sum, i))

			
		# Creates the hierarchical block.
		gr.hier_block.__init__(self, fg, src, self.sum)
	
	
	
