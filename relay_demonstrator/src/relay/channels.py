#!/usr/bin/env python

from gnuradio import gr
from openrd import pr, pr, conf, dmod, link, pyblk, channel
import sys

class channels(gr.hier_block2):
	def __init__(self, top, c):
		gr.hier_block2.__init__(self, "channels", 
				gr.io_signature(0, 0, 0),
				gr.io_signature(0, 0, 0))

		# Load configuration
		self.conf = c

		# Instantiate channels
		if self.conf.link() == 'socket' or self.conf.link() == 'wired':
			if self.conf.channel_mode() == pr.CHMODE_IDEAL:
				self.srchannel = channel.channel_ideal()
				self.rdchannel = channel.channel_ideal()
			elif self.conf.channel_mode() == pr.CHMODE_AWGN:
				self.srchannel = channel.channel_awgn(self.conf.modulation(),
						self.conf.symbol_length())
				self.srchannel.set_noise(self.conf.sd_awgn())
				self.rdchannel = channel.channel_awgn(self.conf.modulation(),
						self.conf.symbol_length())
				self.rdchannel.set_noise(self.conf.sd_awgn())
			elif self.conf.channel_mode() == pr.CHMODE_RAYLEIGH:
				self.srchannel = channel.channel_rayleigh(self.conf.modulation(),
						self.conf.carrier(), self.conf.link_rate(),
						self.conf.symbol_length())
				self.srchannel.set_speed(self.conf.sd_speed())
				self.srchannel.set_noise(self.conf.sd_awgn())
				self.rdchannel = channel.channel_rayleigh(self.conf.modulation(),
						self.conf.carrier(), self.conf.link_rate(),
						self.conf.symbol_length())
				self.rdchannel.set_speed(self.conf.sd_speed())
				self.rdchannel.set_noise(self.conf.sd_awgn())
			else:
				sys.exit(1)
		else:
			sys.exit(0)

		# Instantiate link
		if self.conf.link() == 'socket':
			self.urx = link.rxsocket_c(self.conf.socket_conn(), 3, self.conf.link_rate())
			self.utx = link.txsocket_c(self.conf.socket_conn(), 3, self.conf.link_rate())
			raw_input('Start all units, then press enter')
			self.urx.connect_blocks()
			self.utx.connect_blocks()
		elif self.conf.link() == 'wired':
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.utx = link.txusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.urx.tune(1, self.conf.usrp_carrier())
			self.utx.tune(1, self.conf.usrp_carrier())

		# Instantiate control interface
		if self.conf.control() == pr.CNTRLMODE_NONE:
			self.control = conf.control_none()
		elif self.conf.control() == pr.CNTRLMODE_CONSOLE:
			self.control = conf.control_console()
		else:
			sys.exit(1)

		self.control.set_disp_cmd(True)
		self.control.set_srchannel(self.srchannel)
		self.control.set_rdchannel(self.rdchannel)
		self.control.set_top(top)
		self.control.start()

		# Connect source-relay channel path
		self.connect((self.urx, 0), self.srchannel, 
				pr.insert_head(gr.sizeof_gr_complex, 16384), (self.utx, 0))
		self.connect((self.urx, 1), self.rdchannel, 
				pr.insert_head(gr.sizeof_gr_complex, 16384), (self.utx, 1))

class channels_top(gr.top_block):
	def __init__(self, c):
		gr.top_block.__init__(self)

		self.conf = c

		self.u4 = channels(self, self.conf)

		self.connect(self.u4)

	def quit(self):
		self.stop()

if __name__ == '__main__':
	gr.enable_realtime_scheduling()

	# Load configuration
	c = conf.conf('relay.cfg', 4)

	tb = channels_top(c)
	tb.run()

