#!/usr/bin/env python

from gnuradio import gr, eng_notation, tait
from gnuradio import audio, blks
from gnuradio import usrp
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import sys

# Transmits the contents of a ".dat" file assumed to contain 
# binary encoded shorts. Before transmission the samples
# are passed through a flat Rayleigh channel.
# 
class my_graph(gr.flow_graph):

	def __init__(self):
		gr.flow_graph.__init__(self)
		
		usage="%prog: [options] input_filename (file must contain samples of binary encoded shorts)"
		parser = OptionParser (option_class=eng_option, usage = usage)
		parser.add_option("-T", "--tx-subdev-spec", type="subdev", default=None,
					help="select USRP Tx side A or B")
		parser.add_option("-f", "--freq", type="eng_float", default=None,
					help="set Tx frequency to FREQ [required]", metavar="FREQ")
		(options, args) = parser.parse_args ()
		
		#check for input file name
		(options, args) = parser.parse_args ()
		if len(args) != 1:
			parser.print_help()
			raise SystemExit, 1
		
		filename = args[0]
		
		if options.freq is None:
			sys.stderr.write("Please specify frequency with -f FREQ\n")
			parser.print_help()
			sys.exit(1)
			
		# ----------------------------------------------------------------
		# Set up constants and parameters
	
		self.u = usrp.sink_c ()       # the USRP sink (consumes samples)

		self.dac_rate = self.u.dac_rate()                    # 128 MS/s
		self.usrp_interp = 500
		self.u.set_interp_rate(self.usrp_interp)
		self.usrp_rate = self.dac_rate / self.usrp_interp    # 256 kS/s
		
		# determine the daughterboard subdevice we're using
		if options.tx_subdev_spec is None:
			options.tx_subdev_spec = usrp.pick_tx_subdevice(self.u)
		
		m = usrp.determine_tx_mux_value(self.u, options.tx_subdev_spec)
		#print "mux = %#04x" % (m,)
		self.u.set_mux(m)
		self.subdev = usrp.selected_subdev(self.u, options.tx_subdev_spec)
		print "Using TX d'board %s" % (self.subdev.side_and_name(),)
		
		self.subdev.set_gain(self.subdev.gain_range()[1])    # set max Tx gain
		self.set_freq(options.freq)
		self.subdev.set_enable(True)                         # enable transmitter

		src = gr.file_source(gr.sizeof_short, filename)
		s2c = gr.interleaved_short_to_complex()

		# Set up flat Rayleigh Fading channel.

		chan_seed = -115 	# Seeds the channel
		fdT = 0.002 		# fd = Doppler Fade rate, 1/T = sample rate of channel
		flag_indep = False	# treats individual blocks as one continuous one.
		chan_pwr = 1.0 		# the square root power of the resulting channel wave.
		# This creates the fading channel.
		channel = tait.flat_rayleigh_channel_cc(chan_seed, fdT, chan_pwr, flag_indep)


		# Connects the file src to the fading channel and then onto
		# the usrp sink for transmission.
		self.connect(src, s2c, channel, self.u)
		
		## As a check the faded samples are sunk to file.
		#c2s = gr.complex_to_interleaved_short()
		#dst = gr.file_sink(gr.sizeof_short, "faded.dat")
		#self.connect(src, s2c, channel, c2s, dst)		
		
    
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
        
if __name__ == '__main__':
    try:
        my_graph().run()
    except KeyboardInterrupt:
        pass
