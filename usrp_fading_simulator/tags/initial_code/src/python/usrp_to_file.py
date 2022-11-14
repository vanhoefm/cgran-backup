#!/usr/bin/env python

from gnuradio import gr, gru, eng_notation
from gnuradio import usrp
#import usrp_dbid
import sys
from gnuradio.eng_option import eng_option
from optparse import OptionParser
"""
Receives samples from the usrp to a file.
"""
"""
def pick_subdevice(u):
	
	The user didn't specify a subdevice on the command line.
	Try for one of these, in order: TV_RX, BASIC_RX, whatever is on side A.

	@return a subdev_spec
	
	return usrp.pick_subdev(u, (usrp_dbid.TV_RX,
								usrp_dbid.TV_RX_REV_2,
								usrp_dbid.BASIC_RX))
"""

class wfm_rx_graph (gr.flow_graph):

	def __init__(self):
		gr.flow_graph.__init__(self)

		parser=OptionParser(option_class=eng_option)
		parser.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
						help="select USRP Rx side A or B (default=A)")
		parser.add_option("-f", "--freq", type="eng_float", default=452.75e6,
						help="set frequency to FREQ", metavar="FREQ")
		parser.add_option("-g", "--gain", type="eng_float", default=None,
						help="set gain in dB (default is midpoint)")
		parser.add_option("-O", "--file-name", type="string", default="data.dat",
						help="Name of file to capture USRP output.  E.g. data.dat")

		(options, args) = parser.parse_args()
		if len(args) != 0:
			parser.print_help()
			sys.exit(1)
		
		self.state = "FREQ"
		self.freq = 0

		# build graph
		
		self.u = usrp.source_c()                    # usrp is data source

		adc_rate = self.u.adc_rate()                # 64 MS/s
		usrp_decim = 250
		self.u.set_decim_rate(usrp_decim)
		usrp_rate = adc_rate / usrp_decim           # 256 kS/s

		# determine the daughterboard subdevice we're using
		rx_subdev_spec = usrp.pick_rx_subdevice(self.u)
		self.u.set_mux(usrp.determine_rx_mux_value(self.u, rx_subdev_spec))
		self.subdev = usrp.selected_subdev(self.u, rx_subdev_spec)
		# Selects the RX2 for reception
		self.subdev.select_rx_antenna('RX2')
		
		#g = self.subdev.gain_range()
		#self.subdev.set_gain((g[0] + g[1])/ 2)    # set mid point Rx gain
		#self.subdev.set_gain(g[0])    # set minimum gain
		#self.subdev.set_gain(g[1])    # set maximum gain	
		print "Using RX d'board %s" % (self.subdev.side_and_name(),)
		#self.set_rx_freq(rf_rx_freq)
#---------------------------------------------------------------------------------------------
		# Create the file sink to which the USRP samples are captured.
		self.file_sink = gr.file_sink(gr.sizeof_gr_complex, options.file_name)
	
		# now wire it all together
		self.connect (self.u, self.file_sink)
#---------------------------------------------------------------------------------------------

		if options.gain is None:
			# if no gain was specified, use the mid-point in dB
			g = self.subdev.gain_range()
			options.gain = float(g[0]+g[1])/2

		if abs(options.freq) < 1e6:
			options.freq *= 1e6

		# set initial values

		self.set_gain(options.gain)

		if not(self.set_freq(options.freq)):
			self._set_status_msg("Failed to set initial frequency")
			sys.exit(1)
		else:
			print "Collecting samples from USRP to file. Hit ctrl-c to stop"

	def set_freq(self, target_freq):
		"""
		Set the center frequency we're interested in.

		@param target_freq: frequency in Hz
		@rypte: bool

		Tuning is a two step process.  First we ask the front-end to
		tune as close to the desired frequency as it can.  Then we use
		the result of that operation and our target_frequency to
		determine the value for the digital down converter.
		"""
		r = self.u.tune(0, self.subdev, target_freq)
		
		if r:
			self.freq = target_freq
			self.update_status_bar()
			self._set_status_msg("OK", 0)
			return True

		self._set_status_msg("Failed", 0)
		return False

	def set_gain(self, gain):
		self.subdev.set_gain(gain)

	def update_status_bar (self):
		msg = "Freq: %s  Setting:%s" % (
			eng_notation.num_to_str(self.freq), self.state)
		self._set_status_msg(msg, 1)
		
	def _set_status_msg(self, msg, which=0):
		print msg

	
if __name__ == '__main__':
	fg = wfm_rx_graph()
	try:
		fg.run()
	except KeyboardInterrupt:
		pass

