#!/usr/bin/env python

import sys, sip, time, threading
from gnuradio import gr
from PyQt4 import QtGui, QtCore
from openrd import pr, conf, dmod, link, conf, pyblk, pyui, channel, aaf, daf

class relaynode(gr.hier_block2):
	def __init__(self, top, c):
		gr.hier_block2.__init__(self, "relaynode",
				gr.io_signature(0, 0, 0),
				gr.io_signature(4, 4, gr.sizeof_gr_complex))

		self.conf = c

		# Instantiate relay processing blocks
		if self.conf.relaying_mode() == pr.RELAYING_AAF:
			self.rx = dmod.receiver_cc(self.conf.modulation(), self.conf.symbol_length())
			self.relay = aaf.relay_aaf(mode='fixed', gain=1.0)
			self.tx = dmod.transmitter_cc(self.conf.modulation(), self.conf.symbol_length())
		elif self.conf.relaying_mode() == pr.RELAYING_DAF:
			self.rx = dmod.receiver_cc(self.conf.modulation(), self.conf.symbol_length())
			self.relay = daf.relay_daf(self.conf.block_size(), self.conf.coding(), self.conf.framing(), self.conf.frame_size(), self.conf.modulation(), self.conf.symbol_length())
			self.tx = dmod.transmitter_cc(self.conf.modulation(), self.conf.symbol_length())
		else:
			print "Invalid relaying mode %d" % self.conf.relaying_mode()
			sys.exit(1)

		# Instantiate link
		if self.conf.link() == 'socket':
			self.urx = link.rxsocket_c(self.conf.socket_conn(), 3, self.conf.link_rate())
			self.utx = link.txsocket_c(self.conf.socket_conn(), 3, self.conf.link_rate())
			raw_input('Start all units, then press enter')
			self.utx.connect_blocks()
			self.urx.connect_blocks()
		elif self.conf.link() == 'wired':
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.utx = link.txusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.urx.tune(1, self.conf.usrp_carrier())
			self.urx.tune(2, self.conf.usrp_carrier())
			self.utx.tune(1, self.conf.usrp_carrier())
			self.utx.tune(2, self.cunf.usrp_carrier())
		elif self.conf.link() == 'wireless':
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 1, self.conf.link_rate())
			self.utx = link.txusrp_c(self.conf.usrp_serial(), 2, self.conf.link_rate())
			self.tunesrrx(0)
			self.utx.tune(2, self.conf.usrp_carrier()+self.conf.usrp_relayoffset())
		else:
			sys.exit(1)

		# Instantiate SD channel if it is to be simulated
		if self.conf.link() == 'socket' or self.conf.link() == 'wired':
			if self.conf.channel_mode() == pr.CHMODE_IDEAL:
				self.sdchannel = channel.channel_ideal()
			elif self.conf.channel_mode() == pr.CHMODE_AWGN:
				self.sdchannel = channel.channel_awgn(self.conf.modulation(), self.conf.symbol_length())
				self.sdchannel.set_noise(self.conf.sd_awgn())
			elif self.conf.channel_mode() == pr.CHMODE_RAYLEIGH:
				self.sdchannel = channel.channel_rayleigh(self.conf.modulation(), self.conf.xcarrier(), self.conf.link_rate(), self.conf.symbol_length())
				self.sdchannel.set_speed(self.conf.sd_speed())
				self.sdchannel.set_noise(self.conf.sd_awgn())
			else:
				sys.exit(1)

		# Instantiate estimation blocks
		self.snrest = pr.snr_estimate_c(self.conf.modulation())
		self.symrate = pr.rate_estimate(gr.sizeof_gr_complex)

		# Instantiate control interface
		if self.conf.control() == pr.CNTRLMODE_NONE:
			self.control = conf.control_none()
		elif self.conf.control() == pr.CNTRLMODE_CONSOLE:
			self.control = conf.control_console()
		else:
			sys.exit(1)

		self.control.set_disp_cmd(True)
		self.control.set_top(top)
		self.control.set_tunesrrx(self.tunesrrx)
		self.control.set_relaycntrl(self.relaycntrl)
		if self.conf.link() == 'socket' or self.conf.link() == 'wired':
			self.control.set_sdchannel(self.sdchannel)
		self.control.start()

		# Sinusoidal reference signal
		self.ref = gr.sig_source_c(100, gr.GR_SIN_WAVE, 0, 1.0, 0)
		self.refamp = gr.multiply_const_cc(1)
		self.relayamp = gr.multiply_const_cc(0)
		self.add = gr.add_cc()

		# Scaler for transmitted symbols (to scale when sinus reference is transmitted)
		self.symamp = gr.multiply_const_cc(0)
		
		self.relaycntrl(1)

		# Connect relay blocks
		self.connect((self.urx, 0), self.rx, self.relay, self.tx, self.relayamp, (self.add, 0))
		self.connect(self.ref, self.refamp, (self.add, 1))
		self.connect(self.add, pr.insert_head(gr.sizeof_gr_complex, 16384), (self.utx, 0))

		# Connect SD channel blocks
		if self.conf.link() == 'socket' or self.conf.link() == 'wired':
			self.connect((self.urx, 1), self.sdchannel, pr.insert_head(gr.sizeof_gr_complex, 16384), (self.utx, 1))

		# Connect estimation blocks
		self.connect(self.rx, self.snrest)
		self.connect(self.rx, self.symrate)

		# Connect visualized signals
		self.connect((self.urx, 0), (self, 0)) # Received spectrum
		self.connect(self.rx, (self, 1)) # Received symbols
		self.connect(self.add, (self, 2)) # Transmitted spectrum
		self.connect(self.relay, self.symamp, (self, 3)) # Transmitted symbols

	def tunesrrx(self, offset = 0):
		self.urx.tune(1, self.conf.usrp_carrier() + offset)

	def relaycntrl(self, mode):
		if mode == 0:
			self.refamp.set_k(1)
			self.relayamp.set_k(0)
			self.symamp.set_k(0)
		elif mode == 1:
			self.refamp.set_k(0)
			self.relayamp.set_k(1)
			self.symamp.set_k(1)

	def snr(self):
		return self.snrest.snr()

	def symbol_rate(self):
		return self.symrate.rate()

class relaynode_top(gr.top_block):
	def __init__(self, c, app = None):
		gr.top_block.__init__(self)

		self.conf = c
		self.app = app

		self.finished = False

		fft_size = 1024

		# Create the UI
		if self.conf.relayui() == 'none':
			self.sigview = pyui.signal_viewer_dummy()
			self.snrview = pyui.number_viewer_dummy()
			self.symrateview = pyui.number_viewer_dummy()
		elif self.conf.relayui() == 'text':
			self.sigview = pyui.signal_viewer_dummy()
			self.snrview = pyui.number_viewer_text("Measured SNR", average=10, db=True)
			self.symrateview = pyui.number_viewer_text("Symbol rate", average=1)
		else:
			self.main = QtGui.QWidget()
			self.layout = QtGui.QVBoxLayout(self.main)
			self.hlayout = QtGui.QHBoxLayout()

			self.sigview = pyui.signal_viewer_qt(self.conf.link_rate(), self.conf.symbol_length(), fft_size)
			self.snrview = pyui.number_viewer_qt("Measured SNR", average=10, db=True)
			self.symrateview = pyui.number_viewer_qt("Symbol rate", average=1)

			self.layout.addLayout(self.hlayout)
			self.layout.addWidget(self.sigview)
			self.hlayout.addWidget(self.snrview)
			self.hlayout.addWidget(self.symrateview)

			self.main.show()

		# Instantiate the relay node flowgraph
		self.u2 = relaynode(self, self.conf)

		# Connect the flowgraph
		self.connect((self.u2, 0), (self.sigview, 0))
		self.connect((self.u2, 1), (self.sigview, 1))
		self.connect((self.u2, 2), gr.null_sink(gr.sizeof_gr_complex))
		self.connect((self.u2, 3), gr.null_sink(gr.sizeof_gr_complex))

		self.updater = threading.Thread(target = self.update)
		self.updater.start()

	def update(self):
		while not self.finished:
			self.snrview.add_values(self.u2.snr())
			self.symrateview.set_value(self.u2.symbol_rate())
			time.sleep(1.0)

	def quit(self):
		self.finished = True
		self.stop()
		if self.app is not None:
			self.app.quit()

if __name__ == '__main__':
	gr.enable_realtime_scheduling()
	c = conf.conf('relay.cfg', 2)
	if c.relayui() == 'none' or c.relayui() == 'text':
		tb = relaynode_top(c)
		tb.run()
	elif c.relayui() == 'gui':
		app = QtGui.QApplication(sys.argv)
		tb = relaynode_top(c, app)
		tb.start()
		app.exec_()
	else:
		print "Invalid user interface %s" % c.relayui()

