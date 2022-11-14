#!/usr/bin/env python

"""
Transmits a 800 Hz tone, as a narrow band FM signal.
"""
from gnuradio import gr
from gnuradio import usrp
from gnuradio import blks
from Numeric import convolve, array

class test_tone(gr.flow_graph):

	def __init__(self):
		gr.flow_graph.__init__(self)

		tone_freq = 800
		tx_freq = 440.1e6

		tone = transmit_path(self, None, tone_freq)
		tone.set_freq(tx_freq)
		tone.set_enable(True)


    
class transmit_path(gr.hier_block):
	def __init__(self, fg, subdev_spec, tone_freq):
		self.u = usrp.sink_c ()

		dac_rate = self.u.dac_rate();
		self.if_rate = 320e3                               # 320 kS/s
		self.usrp_interp = int(dac_rate // self.if_rate)
		self.u.set_interp_rate(self.usrp_interp)
		self.sw_interp = 10
		self.audio_rate = self.if_rate // self.sw_interp   #  32 kS/s

		self.audio_gain = 1
		self.normal_gain = 500

		self.audio = gr.sig_source_f (self.audio_rate,    	# sample rate
			                            gr.GR_SIN_WAVE, 	# waveform type
			                            tone_freq,     		# frequency
			                            1.0,            	# amplitude
			                            0)              	# DC Offset


		self.audio_amp = gr.multiply_const_ff(self.audio_gain)

		lpf = gr.firdes.low_pass (1,                # gain
		                            self.audio_rate,            # sampling rate
		                            3800,               # low pass cutoff freq
		                            300,                # width of trans. band
		                            gr.firdes.WIN_HANN) # filter type 

		hpf = gr.firdes.high_pass (1,                # gain
		                            self.audio_rate,            # sampling rate
		                            325,               # low pass cutoff freq
		                            50,                # width of trans. band
		                            gr.firdes.WIN_HANN) # filter type 

		audio_taps = convolve(array(lpf),array(hpf))
		self.audio_filt = gr.fir_filter_fff(1,audio_taps)

		self.pl = blks.ctcss_gen_f(fg, self.audio_rate,123.0)
		self.add_pl = gr.add_ff()
		fg.connect(self.pl,(self.add_pl,1))

		self.fmtx = blks.nbfm_tx(fg, self.audio_rate, self.if_rate)
		self.amp = gr.multiply_const_cc (self.normal_gain)

		# determine the daughterboard subdevice we're using
		if subdev_spec is None:
			subdev_spec = usrp.pick_tx_subdevice(self.u)
		self.u.set_mux(usrp.determine_tx_mux_value(self.u, subdev_spec))
		self.subdev = usrp.selected_subdev(self.u, subdev_spec)
		print "TX using", self.subdev.side_and_name()

		fg.connect(self.audio, self.audio_amp, self.audio_filt,
		            (self.add_pl,0), self.fmtx, self.amp, self.u)

		gr.hier_block.__init__(self, fg, None, None)

		self.set_gain(self.subdev.gain_range()[1])  # set max Tx gain


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
			# Use residual_freq in s/w freq translator
			return True

		return False

	def set_gain(self, gain):
		self.gain = gain
		self.subdev.set_gain(gain)

	def set_enable(self, enable):
		self.subdev.set_enable(enable)            # set H/W Tx enable
		if enable:
			self.amp.set_k (self.normal_gain)
		else:
			self.amp.set_k (0)

