#!/usr/bin/env python

from gnuradio import gr, gru, eng_notation
from gnuradio import blks
from gnuradio import usrp
from setup_fading_channel import *

class filesrc_fade_transmit(gr.flow_graph):

	def __init__(self, rf_tx_freq, recv_speed, file_sample_rate, rf_tx_power, filename):
		"""" Transmits the contents of a ".dat" file assumed to contain 
			 binary encoded complex base-band samples. Before transmission the 
			 samples are passed through a flat Rayleigh channel. No modulation is applied.
			 
	 	Inputs:
	 	
		- rf_tx_freq: The RF transmission frequency (Hz).
		- recv_speed: The receiver speed (km/h). If set to zero then 
					  Rayleigh fading is disabled.
		- file_sample_rate: The sample rate of the source file (Samples/sec) 
		- rf_tx_power: The RF transmit power (dB) (no affect on fading parameters).
		- filename: The name of the source file (string).
					  
		Example:
		
			fsr_fade_tx = filesrc_fade_transmit(440.1e6, 100, 192e3, -35, 'c4fm_test_tone.dat')

		"""		
		gr.flow_graph.__init__(self)
		
		# Set up Transmitter	
		#-----------------------------------------------------------------
		self.u = usrp.sink_c ()       # the USRP sink (consumes samples)
		self.dac_rate = self.u.dac_rate()                    # 128 MS/s
		self.usrp_interp = 500
		self.u.set_interp_rate(self.usrp_interp)
		self.usrp_rate = self.dac_rate / self.usrp_interp    # 256 kS/s
		# determine the daughterboard subdevice we're using
		tx_subdev_spec = usrp.pick_tx_subdevice(self.u)
		m = usrp.determine_tx_mux_value(self.u, tx_subdev_spec)
		self.u.set_mux(m)
		self.subdev = usrp.selected_subdev(self.u, tx_subdev_spec)
		print "Using TX d'board %s" % (self.subdev.side_and_name(),)
		self.subdev.set_gain(self.subdev.gain_range()[1])    # set max Tx gain
		self.set_freq(rf_tx_freq)
		self.subdev.set_enable(True)                         # enable transmitter
		#----------------------------------------------------------------- 


		# Re-sampler (required to match the file sample rate with the usrp
		# rate).
		#-----------------------------------------------------------------
		input_rate = file_sample_rate
		output_rate = self.usrp_rate
		print "input_rate = ", input_rate
		print "output_rate = ", output_rate
		interp = gru.lcm(input_rate, output_rate) / input_rate
		decim = gru.lcm(input_rate, output_rate) / output_rate
		print "interp  = ", int(interp)
		print "decim = ", int(decim)
		rr = blks.rational_resampler_ccc(self, int(interp), int(decim))
		#------------------------------------------------------------------
		
		
		# Setup file source and gain stage. 
		#------------------------------------------------------------------
		src = gr.file_source(gr.sizeof_gr_complex, filename)
		gain = gr.multiply_const_cc (rf_tx_power)
		print "Filename = ", filename
		
		#------------------------------------------------------------------
		
		# Adds Rayleigh fading.
		if recv_speed != 0:
			fade_channel = setup_fading_channel(recv_speed, rf_tx_freq, file_sample_rate)
			self.connect(src, fade_channel, rr, gain, self.u)
		else:
			print "Fading is disabled"
			#self.connect(src, rr, gain, self.u)
			self.connect(src, self.u)
	
    
	def set_freq(self, target_freq):
		"""
		Set the center frequency we're interested in.
		
		@param target_freq: frequency in Hz
		@rypte: bool
		
		Tuning is a two step process.  First we ask the front-end to
		tune as close to the desired frequency as it can.  Then we use
		the result of that operation and our target_frequency to
		determine the value for the digital up converter.  Finally, we feed
		any residual_freq to the s/w freq translator.
		"""
		
		r = self.u.tune(self.subdev._which, self.subdev, target_freq)
		if r:
			print "r.baseband_freq =", eng_notation.num_to_str(r.baseband_freq)
			print "r.dxc_freq      =", eng_notation.num_to_str(r.dxc_freq)
			print "r.residual_freq =", eng_notation.num_to_str(r.residual_freq)
			print "r.inverted      =", r.inverted
			
			# Could use residual_freq in s/w freq translator
			return True
	
		return False

