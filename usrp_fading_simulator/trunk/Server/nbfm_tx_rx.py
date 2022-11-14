#!/usr/bin/env python

from gnuradio import gr, eng_notation, tait
from gnuradio import audio, blks
from gnuradio import usrp
from gnuradio.eng_option import eng_option
from optparse import OptionParser
from multipath_rayleigh_channel_cc import *
import sys

# Simply retransmits what is received on one frequency, on another frequency (with 
# no intermidate processing).
# However, can optionally apply multipath Rayleigh fading (see below).

class my_graph(gr.flow_graph):

	def __init__(self):
		gr.flow_graph.__init__(self)
		
		usage="%prog: [options] input_filename (file must contain samples of binary encoded shorts)"
		parser = OptionParser (option_class=eng_option)
		parser.add_option("-T", "--tx-subdev-spec", type="subdev", default=None,
					help="select USRP Tx side A or B")
		parser.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
					help="select USRP Rx side A or B")
		parser.add_option("-t", "--tx-freq", type="eng_float", default=None,
					help="set Tx frequency to FREQ [required]", metavar="FREQ")
		parser.add_option("-r", "--rx-freq", type="eng_float", default=None,
					help="set Rx frequency to FREQ [required]", metavar="FREQ")
		(options, args) = parser.parse_args ()
	
		if len(args) != 0 or options.rx_freq == None or options.tx_freq == None:
			parser.print_help()
			sys.exit(1)
	
		if options.tx_freq < 1e6 or options.rx_freq < 1e6:
			parser.print_help()
			raise ValueError, "Frequency must be in MHz" 
			
		# ----------------------------------------------------------------
		# Set up Receiver
		self.src = usrp.source_c()                    # usrp is data source
		
		adc_rate = self.src.adc_rate()                # 64 MS/s
		usrp_decim = 128						
		self.src.set_decim_rate(usrp_decim)
		usrp_rate = adc_rate / usrp_decim           # 500kS/s
		
		# determine the daughterboard subdevice we're using
		if options.rx_subdev_spec is None:
			options.rx_subdev_spec = usrp.pick_rx_subdevice(self.src)
		
		self.src.set_mux(usrp.determine_rx_mux_value(self.src, options.rx_subdev_spec))
		self.subdev = usrp.selected_subdev(self.src, options.rx_subdev_spec)

		# Selects the RX2 for reception
		self.subdev.select_rx_antenna('RX2')
		
		g = self.subdev.gain_range()
		self.subdev.set_gain((g[0] + g[1])/ 2)    # set min Rx gain
		
		print "Using RX d'board %s" % (self.subdev.side_and_name(),)
		self.set_rx_freq(options.rx_freq)
		# ----------------------------------------------------------------			
			
			
		# ----------------------------------------------------------------
		# Set up Transmitter
	
		self.dst = usrp.sink_c ()       # the USRP sink (consumes samples)

		self.dac_rate = self.dst.dac_rate()                    # 128 MS/s
		self.usrp_interp = 256
		self.dst.set_interp_rate(self.usrp_interp)
		self.usrp_rate = self.dac_rate / self.usrp_interp    # 500kS/s
		
		# determine the daughterboard subdevice we're using
		if options.tx_subdev_spec is None:
			options.tx_subdev_spec = usrp.pick_tx_subdevice(self.dst)
		
		m = usrp.determine_tx_mux_value(self.dst, options.tx_subdev_spec)
		#print "mux = %#04x" % (m,)
		self.dst.set_mux(m)
		self.subdev = usrp.selected_subdev(self.dst, options.tx_subdev_spec)
		print "Using TX d'board %s" % (self.subdev.side_and_name(),)
		
		self.subdev.set_gain(self.subdev.gain_range()[1])    # set max Tx gain
		self.set_tx_freq(options.tx_freq)
		self.subdev.set_enable(True)                         # enable transmitter
		# ----------------------------------------------------------------

		# Adds multipath Rayleigh fading.
		if 0:
			
			# Set up the parameters for the multipath fading channel:
			chan_seed = -115 	# Seeds the channel
			
	
			vehicle_speed = 100 		# the mobile speed (km/h)
			carrier_freq = options.tx_freq 	# transmiter carrier frequance (Hz)
			chan_rate = usrp_rate 		# channel sample rate (Samples/sec)
			chan_seed = -115 		# Seeds the channel
			chan_pwrs = (1.0,1.0)		# The square root power of each resulting channel wave.
			path_delays = (0,4 )		# The delays, in microseconds, for each channel.
			flag_indep = False		# Treats individual blocks as one continuous one.
			flag_norm = True 		# Normalizes the channel output power. Thus chan_pwrs are relative 								#only.
				
			
			# This creates the fading channel based on the above parameters.
			#channel = multipath_rayleigh_channel_cc (self, self.src, chan_seed, fdT, chan_pwrs, path_delays, False, True)
			# This creates the fading channel based on the above parameters.
			channel = multipath_rayleigh_channel_cc(self, self.src, vehicle_speed, carrier_freq,
					chan_rate, chan_seed, chan_pwrs, path_delays, flag_indep, flag_norm)
			print "Fading enabled"
			self.connect(channel, self.dst)
		else:
			# Connects the usrp receive path to the usrp transmit path.
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
			return True
		
		return False
	
        
if __name__ == '__main__':
    try:
        my_graph().run()
    except KeyboardInterrupt:
        pass
