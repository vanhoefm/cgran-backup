#!/usr/bin/env python

from gnuradio import gr, eng_notation
from gnuradio import usrp
from setup_fading_channel import *

class rxsrc_fade_transmit(gr.flow_graph):

	def __init__(self, rf_tx_freq, recv_speed, rf_rx_freq, usrp_side):
		"""" Applies fading to what is received on one frequency and retransmits the 
		 	 faded signal on another frequency.
		
		Due to RF leakage ensure that the RF rx and tx frequencies are well spaced apart.
			 
	 	Inputs:
	 	
	 	- rf_tx_freq: The RF transmission frequency (Hz). 
		- recv_speed: The receiver speed (km/h). If set to zero then 
					  Rayleigh fading is disabled.	 	 
	 	- rf_rx_freq: The RF receiver frequency (Hz) (no affect on fading parameters).		
		
		Example:
		
			fsr_fade_tx = rxsrc_fade_transmit(440.1e6, 100, 410e6)

		"""		
		gr.flow_graph.__init__(self)
		
		# Set up Receiver	
		#-----------------------------------------------------------------
		self.src = usrp.source_c()				# usrp is data source
		adc_rate = self.src.adc_rate()			# 64 MS/s
		usrp_decim = 256
		self.src.set_decim_rate(usrp_decim)
		usrp_rate = adc_rate / usrp_decim		# 256 kS/s
		# determine the daughterboard subdevice we're using
		rx_subdev_spec = usrp.pick_rx_subdevice(self.src)
		self.src.set_mux(usrp.determine_rx_mux_value(self.src, rx_subdev_spec))
		self.subdev = usrp.selected_subdev(self.src, rx_subdev_spec)
		# Selects the RX2 for reception
		self.subdev.select_rx_antenna('RX2')
		g = self.subdev.gain_range()
		self.subdev.set_gain((g[0] + g[1])/ 2)    # set mid point Rx gain
		#self.subdev.set_gain(g[0])    # set minimum gain
		#self.subdev.set_gain(g[1])    # set maximum gain	
		print "Using RX d'board %s" % (self.subdev.side_and_name(),)
		self.set_rx_freq(rf_rx_freq)
		# ----------------------------------------------------------------			
			
			
		# Set up Transmitter	
		# ----------------------------------------------------------------
		self.dst = usrp.sink_c ()       # the USRP sink (consumes samples)
		self.dac_rate = self.dst.dac_rate()                  # 128 MS/s
		self.usrp_interp = 512
		self.dst.set_interp_rate(self.usrp_interp)
		self.usrp_rate = self.dac_rate / self.usrp_interp    # 256 kS/s
		# determine the daughterboard subdevice we're using. Modify to make this configurable by
		# the Client.
		
		#usrp_side is either 0 (A) or 1 (B)
		if usrp_side == 1:
			tx_subdev_spec = (1,0)	# Selects side B, subdevice 1
		else:
			#Picks first subdev found which will be the Flex400 board			
			tx_subdev_spec = usrp.pick_tx_subdevice(self.dst)

		m = usrp.determine_tx_mux_value(self.dst, tx_subdev_spec)
		self.dst.set_mux(m)
		self.subdev = usrp.selected_subdev(self.dst, tx_subdev_spec)
		print "Using TX d'board %s" % (self.subdev.side_and_name(),)
		g = self.subdev.gain_range()	#returns a 2 element array with minimum and maximum gains
		#self.subdev.set_gain((g[0] + g[1])/ 2)    # set mid point Tx gain
		self.subdev.set_gain(g[1])    # set max Tx gain

		#Define how much software gain to apply to the complex samples heading out to the LFTX
		MULTI = gr.multiply_const_cc(1.0)
		
		self.set_tx_freq(rf_tx_freq)
		self.subdev.set_enable(True)                         # enable transmitter
		# ----------------------------------------------------------------

		d = gr.enable_realtime_scheduling()
    		if d != gr.RT_OK:
        		print "Warning: Failed to enable realtime scheduling."

		# Adds Rayleigh fading.
		if recv_speed != 0:
			fade_channel = setup_fading_channel(recv_speed, rf_tx_freq, self.usrp_rate)
			self.connect(self.src, fade_channel, self.dst)
		else:
			print "Fading is disabled"
			# Connects the usrp receive path to the usrp transmit path.
			#self.connect(self.src, MULTI, self.dst)
			self.connect(self.src, self.dst)

    
	def set_tx_freq(self, target_freq):
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
		
		r = self.dst.tune(self.subdev._which, self.subdev, target_freq)
		if r:
			print "r.baseband_freq =", eng_notation.num_to_str(r.baseband_freq)
			print "r.dxc_freq      =", eng_notation.num_to_str(r.dxc_freq)
			print "r.residual_freq =", eng_notation.num_to_str(r.residual_freq)
			print "r.inverted      =", r.inverted
			
			# Could use residual_freq in s/w freq translator
			return True
	
		return False
	
	def set_rx_freq(self, target_freq):
		"""
		Set the center frequency we're interested in.
		
		@param target_freq: frequency in Hz
		@rypte: bool
		
		Tuning is a two step process.  First we ask the front-end to
		tune as close to the desired frequency as it can.  Then we use
		the result of that operation and our target_frequency to
		determine the value for the digital down converter.
		"""
		r = self.src.tune(0, self.subdev, target_freq)
		
		if r:
			print "r.baseband_freq =", eng_notation.num_to_str(r.baseband_freq)
			print "r.dxc_freq      =", eng_notation.num_to_str(r.dxc_freq)
			print "r.residual_freq =", eng_notation.num_to_str(r.residual_freq)
			print "r.inverted      =", r.inverted
			
			# Could use residual_freq in s/w freq translator
			
			return True
		
		return False
	
        
if __name__ == '__main__':
    try:
    	t = rxsrc_fade_transmit(440.1e6, 440.1e6)
        t.run()
    except KeyboardInterrupt:
        pass
