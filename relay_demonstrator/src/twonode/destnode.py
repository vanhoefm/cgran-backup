#!/usr/bin/env python

from gnuradio import gr
from PyQt4 import QtGui, QtCore
from openrd import pr, conf, link, dmod, pyblk, pyui
import sys, sip, time, threading, math

class destnode(gr.hier_block2):
	def __init__(self, top, c):
		gr.hier_block2.__init__(self, "destnode",
				gr.io_signature(0, 0, 0),
				gr.io_signaturev(2, 2, [
					gr.sizeof_gr_complex,   # Source signal samples
					gr.sizeof_gr_complex])) # Source signal symbols

		self.conf = c

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			from openrd import vtapp

		# Instantiate reference data source
		if self.conf.source_data_mode() == pr.SRCMODE_ZERO:
			self.data = pr.data_source_zero(self.conf.block_size())
		elif self.conf.source_data_mode() == pr.SRCMODE_COUNTER:
			self.data = pr.data_source_counter(self.conf.block_size())
		elif self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			pass
		else:
			print "Invalid source data mode %d" % self.conf.source_data_mode()
			sys.exit(1)

		# Instantiate receiver
		self.rec = dmod.receiver_cc(self.conf.modulation(), self.conf.symbol_length())

		# Instantiate reference processing block
		self.dest = pyblk.dest_ref(self.conf.block_size(), self.conf.coding(), self.conf.framing(), self.conf.frame_size(), self.conf.modulation())

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			# Instantiate packet receiver
			self.datadst = pr.data_sink_packet(vtapp.num_proto(self.conf.application()),
					self.conf.block_size(), pr.PACKET_SINK_BLOCK)
		else:
			# Instantiate data analyzer
			if self.conf.analysis_mode() == pr.AMODE_NONE:
				self.analyzer = pr.analyzer_none_vb(self.conf.block_size())
			elif self.conf.analysis_mode() == pr.AMODE_BER:
				self.analyzer = pr.analyzer_ber_vb(self.conf.block_size(), 10)
			else:
				print "Invalid analysis mode %d" % self.conf.analysis_mode()
				sys.exit(1)

		# Instantiate link
		if self.conf.link() == 'socket':
			self.urx = link.rxsocket_c(self.conf.socket_conn(), 1, self.conf.link_rate())
			raw_input('Start all units, then press enter')
			self.urx.connect_blocks()
		elif self.conf.link() == 'wired':
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 1, self.conf.link_rate())
			self.tunesdrx(0)
		elif self.conf.link() == 'wireless':	
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 1, self.conf.link_rate())
			self.tunesdrx(0)

		# Instantiate estimation blocks
		self.snrest = pr.snr_estimate_c(self.conf.modulation())
		self.symrate = pr.rate_estimate(gr.sizeof_gr_complex)
		self.pktrate = pr.rate_estimate(pr.pvec_alloc_size(pr.sizeof_rxmeta+self.conf.block_size()))

		# Instantiate control interface
		if self.conf.control() == pr.CNTRLMODE_NONE:
			self.control = conf.control_none()
		elif self.conf.control() == pr.CNTRLMODE_CONSOLE:
			self.control = conf.control_console()
		else:
			sys.exit(1)

		self.control.set_disp_cmd(True)
		self.control.set_top(top)
		self.control.set_tunesdrx(self.tunesdrx)
		self.control.start()

		# Connect received signal
		self.connect(self.urx, self.rec, self.dest)

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			# Connect received signal to packet sink
			self.connect(self.dest, self.datadst)
		else:
			# Connect data source and received signal to analyzer
			self.connect(self.data, (self.analyzer, 0))
			self.connect(self.dest, (self.analyzer, 1))

		# Connect estimation blocks
		self.connect(self.rec, self.snrest)
		self.connect(self.rec, self.symrate)
		self.connect(self.dest, self.pktrate)

		# Connect visualized signals
		self.connect(self.urx, (self, 0))
		self.connect(self.rec, (self, 1))

		# Create receiver application
		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.app = vtapp.dest(self.datadst, self.conf.application())
			self.app.set_size((80,60))
			self.app.start()

	def tunesdrx(self, offset = 0):
		self.urx.tune(1, self.conf.usrp_carrier()+offset)

	def ber(self):
		if self.conf.analysis_mode() == pr.AMODE_BER:
			return self.analyzer.ber()
		else:
			print "ber() called with analysis mode %s" % self.conf.analysis_mode()

	def snr(self):
		return self.snrest.snr()

	def symbol_rate(self):
		return self.symrate.rate()

	def packet_rate(self):
		return self.pktrate.rate();

	def quit(self):
		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.app.quit()

class destnode_top(gr.top_block):
	def __init__(self, c, app = None):
		gr.top_block.__init__(self)

		self.conf = c
		self.app = app

		self.finished = False

		fft_size = 1024

		# Create the UI
		if self.conf.destui() == 'none':
			self.sigview = pyui.signal_viewer_dummy()
			self.berview = pyui.number_viewer_dummy()
			self.snrview = pyui.number_viewer_dummy()
			self.symrateview = pyui.number_viewer_dummy()
			self.pktrateview = pyui.number_viewer_dummy()
		elif self.conf.destui() == 'text':
			self.sigview = pyui.signal_viewer_dummy()
			self.snrview = pyui.number_viewer_text("Measured SNR", average=10, db=True)
			self.symrateview = pyui.number_viewer_text("Symbol rate", average=1)
			self.pktrateview = pyui.number_viewer_text("Packet rate", average=1)
			self.berview = pyui.number_viewer_text("Bit error rate", average=10)
		elif self.conf.destui() == 'gui':
			self.main = QtGui.QWidget()
			self.layout = QtGui.QVBoxLayout(self.main)
			self.hlayout = QtGui.QHBoxLayout()

			self.sigview = pyui.signal_viewer_qt(self.conf.link_rate(), self.conf.symbol_length(), fft_size)
			self.snrview = pyui.number_viewer_qt("Measured SNR", average=10, db=True)
			self.symrateview = pyui.number_viewer_qt("Symbol rate", average=1)
			self.pktrateview = pyui.number_viewer_qt("Packet rate", average=1)
			self.berview = pyui.number_viewer_qt("Bit error rate", average=10)

			self.layout.addLayout(self.hlayout)
			self.layout.addWidget(self.sigview)
			self.hlayout.addWidget(self.snrview)
			self.hlayout.addWidget(self.symrateview)
			self.hlayout.addWidget(self.pktrateview)
			self.hlayout.addWidget(self.berview)

			self.main.show()

		# Instantiate the destination node flowgraph
		self.u3 = destnode(self, self.conf)

		self.connect((self.u3, 0), (self.sigview, 0))
		self.connect((self.u3, 1), (self.sigview, 1))

		self.updater = threading.Thread(target = self.update)
		self.updater.start()

	def update(self):
		while not self.finished:
			self.snrview.add_values(self.u3.snr())
			self.symrateview.set_value(self.u3.symbol_rate())
			self.pktrateview.set_value(self.u3.packet_rate())
			if self.conf.analysis_mode() == pr.AMODE_BER:
				self.berview.add_values(self.u3.ber())
			time.sleep(1.0)
	
	def quit(self):
		self.finished = True
		self.u3.quit()
		self.stop()
		if self.app is not None:
			self.app.quit()

if __name__ == '__main__':
	gr.enable_realtime_scheduling()
	c = conf.conf('twonode.cfg', 3)
	if c.destui() == 'none' or c.destui() == 'text':
		tb = destnode_top(c)
		tb.run()
	elif c.destui() == 'gui':
		app = QtGui.QApplication(sys.argv)
		tb = destnode_top(c, app);
		tb.start()
		app.exec_()
	else:
		print "Invalid user interface %s" % c.destui()

