#!/usr/bin/env python

from gnuradio import gr
from PyQt4 import QtGui, QtCore
from openrd import pr, conf, link, dmod, pyblk, pyui, aaf, daf
import sys, sip, time, threading

class destnode(gr.hier_block2):
	def __init__(self, top, c):
		gr.hier_block2.__init__(self, "destnode",
				gr.io_signature(0, 0, 0),
				gr.io_signaturev(4, 4,
					[gr.sizeof_gr_complex,   # Source signal samples
					gr.sizeof_gr_complex,   # Source signal symbols
					gr.sizeof_gr_complex,   # Relay signal samples
					gr.sizeof_gr_complex]))  # Relay signal symbols

		self.conf = c

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			from openrd import vtapp

		# Instantiate reference data sources
		if self.conf.source_data_mode() == pr.SRCMODE_ZERO:
			self.dataref = pr.data_source_zero(self.conf.block_size())
			self.datacoop = pr.data_source_zero(self.conf.block_size())
		elif self.conf.source_data_mode() == pr.SRCMODE_COUNTER:
			self.dataref = pr.data_source_counter(self.conf.block_size())
			self.datacoop = pr.data_source_counter(self.conf.block_size())
		elif self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			pass
		else:
			print "Invalid source data mode %d" % self.conf.source_data_mode()
			sys.exit(1)

		# Instantiate receivers
		self.recref = dmod.receiver_cc(self.conf.modulation(), self.conf.symbol_length())
		self.recrelay = dmod.receiver_cc(self.conf.modulation(), self.conf.symbol_length())

		# Instantiate reference processing block
		self.destref = pyblk.dest_ref(self.conf.block_size(), self.conf.coding(), self.conf.framing(), self.conf.frame_size(), self.conf.modulation())

		# Instantiate relay processing block
		if self.conf.relaying_mode() == pr.RELAYING_AAF:
			self.destcoop = aaf.dest_aaf(self.conf.block_size(), self.conf.coding(), self.conf.framing(), self.conf.frame_size(), self.conf.modulation())
		elif self.conf.relaying_mode() == pr.RELAYING_DAF:
			self.destcoop = daf.dest_daf(self.conf.block_size(), self.conf.coding(), self.conf.framing(), self.conf.frame_size(), self.conf.modulation())

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			# Instantiate packet receiver
			self.datadst = pr.data_sink_packet(vtapp.num_proto(self.conf.application()),
					self.conf.block_size(), pr.PACKET_SINK_BLOCK)
		else:
			# Instantiate data analyzers
			if self.conf.analysis_mode() == pr.AMODE_NONE:
				self.analyzerref = pr.analyzer_none_vb(self.conf.block_size())
				self.analyzercoop = pr.analyzer_none_vb(self.conf.block_size())
			elif self.conf.analysis_mode() == pr.AMODE_BER:
				self.analyzerref = pr.analyzer_ber_vb(self.conf.block_size(), 10)
				self.analyzercoop = pr.analyzer_ber_vb(self.conf.block_size(), 10)
			else:
				print "Invalid analysis mode %d" % self.conf.analysis_mode()
				sys.exit(1)

		# Instantiate link
		if self.conf.link() == 'socket':
			self.urx = link.rxsocket_c(self.conf.socket_conn(), 3, self.conf.link_rate())
			raw_input('Start all units, then press enter')
			self.urx.connect_blocks()
		elif self.conf.link() == 'wired':
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.urx.tune(1, self.conf.usrp_carrier())
			self.urx.tune(2, self.conf.usrp_carrier())
		elif self.conf.link() == 'wireless':	
			self.urx = link.rxusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.tunesdrx(0)
			self.tunerdrx(0)

		# Instantiate estimation blocks
		self.refsnrest = pr.snr_estimate_c(self.conf.modulation())
		self.relaysnrest = pr.snr_estimate_c(self.conf.modulation())
		self.refsymrate = pr.rate_estimate(gr.sizeof_gr_complex)
		self.relaysymrate = pr.rate_estimate(gr.sizeof_gr_complex)
		self.dirpktrate = pr.rate_estimate(pr.pvec_alloc_size(pr.sizeof_rxmeta+self.conf.block_size()))
		self.cooppktrate = pr.rate_estimate(pr.pvec_alloc_size(pr.sizeof_rxmeta+self.conf.block_size()))

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
		self.control.set_tunerdrx(self.tunerdrx)
		self.control.start()

		# Connect received signals
		self.connect((self.urx, 0), self.recref)
		self.connect((self.urx, 1), self.recrelay)
		self.connect(self.recref, self.destref)
		self.connect(self.recref, (self.destcoop, 0))
		self.connect(self.recrelay, (self.destcoop, 1))

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			# Connect received signal to packet sink
			self.connect(self.destcoop, self.datadst)
		else:
			# Connect data source and received signal to analyzer
			self.connect(self.dataref, (self.analyzerref, 0))
			self.connect(self.destref, (self.analyzerref, 1))
			self.connect(self.datacoop, (self.analyzercoop, 0))
			self.connect(self.destcoop, (self.analyzercoop, 1))

		# Connect estimation blocks
		self.connect(self.recref, self.refsnrest)
		self.connect(self.recrelay, self.relaysnrest)
		self.connect(self.recref, self.refsymrate)
		self.connect(self.recrelay, self.relaysymrate)
		self.connect(self.destref, self.dirpktrate)
		self.connect(self.destcoop, self.cooppktrate)

		# Connect visualized signals
		self.connect((self.urx, 0), (self, 0))
		self.connect(self.recref, (self, 1))
		self.connect((self.urx, 1), (self, 2))
		self.connect(self.recrelay, (self, 3))

		# Create receiver application
		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.app = vtapp.dest(self.datadst, self.conf.application())
			self.app.set_size((80,60))
			self.app.start()

	def tunesdrx(self, offset = 0):
		self.urx.tune(1, self.conf.usrp_carrier()+offset)

	def tunerdrx(self, offset = 0):
		self.urx.tune(2, self.conf.usrp_carrier() + self.conf.usrp_relayoffset() + offset)

	def ref_snr(self):
		return self.refsnrest.snr()

	def relay_snr(self):
		return self.relaysnrest.snr()

	def ref_symbol_rate(self):
		return self.refsymrate.rate()

	def relay_symbol_rate(self):
		return self.relaysymrate.rate()

	def direct_ber(self):
		if self.conf.analysis_mode() == pr.AMODE_BER:
			return self.analyzerref.ber()
		else:
			return [0]

	def coop_ber(self):
		if self.conf.analysis_mode() == pr.AMODE_BER:
			return self.analyzercoop.ber()
		else:
			return [0]

	def direct_packet_rate(self):
		return self.dirpktrate.rate()

	def coop_packet_rate(self):
		return self.cooppktrate.rate()

	def quit(self):
		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.app.quit()

class perf_viewer(QtGui.QWidget):
	def __init__(self):
		QtGui.QWidget.__init__(self, None)

		self.layout = QtGui.QHBoxLayout(self)

		self.berviewdir = pyui.number_viewer_qt("BER, direct", average=10)
		self.berviewdiravg = pyui.number_viewer_qt("BER, direct (avg)", average=100)
		self.berviewcoop = pyui.number_viewer_qt("BER, coop", average=10)
		self.berviewcoopavg = pyui.number_viewer_qt("BER, coop (avg)", average=100)
		self.pktrateviewdir = pyui.number_viewer_qt("Packet rate, direct")
		self.pktrateviewcoop = pyui.number_viewer_qt("Packet rate, coop")

		self.layout.addWidget(self.berviewdir)
		self.layout.addWidget(self.berviewdiravg)
		self.layout.addWidget(self.berviewcoop)
		self.layout.addWidget(self.berviewcoopavg)
		self.layout.addWidget(self.pktrateviewdir)
		self.layout.addWidget(self.pktrateviewcoop)
		self.layout.addStretch(1)

class link_viewer(QtGui.QWidget):
	def __init__(self, label, link_rate, symbol_length, fft_size):
		QtGui.QWidget.__init__(self, None)

		self.layout = QtGui.QVBoxLayout(self)
		self.labellayout = QtGui.QHBoxLayout()
		self.infolayout = QtGui.QHBoxLayout()

		self.label = QtGui.QLabel("<b><big>" + label + "</big></b>")
		self.sigview = pyui.signal_viewer_qt(link_rate, symbol_length, fft_size, "%s", 1)
		self.snrview = pyui.number_viewer_qt("SNR", average=10, db=True)
		self.rateview = pyui.number_viewer_qt("Symbol rate", average=1)

		self.layout.addLayout(self.labellayout)
		self.layout.addLayout(self.infolayout)
		self.layout.addWidget(self.sigview)
		self.labellayout.addStretch(1)
		self.labellayout.addWidget(self.label)
		self.labellayout.addStretch(1)
		self.infolayout.addWidget(self.snrview)
		self.infolayout.addWidget(self.rateview)
		self.infolayout.addStretch(1)

class destnode_top(gr.top_block):
	def __init__(self, c, app = None):
		gr.top_block.__init__(self)

		self.conf = c
		self.app = app

		self.finished = False

		fft_size = 1024

		# Create the UI
		if self.conf.destui() == 'none':
			self.sigviewref = pyui.signal_viewer_dummy()
			self.sigviewrelay = pyui.signal_viewer_dummy()
			self.berviewdir = pyui.number_viewer_dummy()
			self.berviewcoop = pyui.number_viewer_dummy()
			self.pktrateviewdir = pyui.number_viewer_dummy()
			self.pktrateviewcoop = pyui.number_viewer_dummy()
			self.snrviewref = pyui.number_viewer_dummy()
			self.snrviewrelay = pyui.number_viewer_dummy()
			self.symrateviewref = pyui.number_viewer_dummy()
			self.symrateviewrelay = pyui.number_viewer_dummy()
		elif self.conf.destui() == 'text':
			self.sigviewref = pyui.signal_viewer_dummy()
			self.sigviewrelay = pyui.signal_viewer_dummy()
			self.berviewdir = pyui.number_viewer_text("BER, direct", average=10)
			self.berviewcoop = pyui.number_viewer_text("BER, coop", average=10)
			self.pktrateviewdir = pyui.number_viewer_text("Packet rate, direct")
			self.pktrateviewcoop = pyui.number_viewer_text("Packet rate, coop")
			self.snrviewref = pyui.number_viewer_text("SNR, source", average=10, db=True)
			self.snrviewrelay = pyui.number_viewer_text("SNR, relay", average=10, db=True)
			self.symrateviewref = pyui.number_viewer_text("Symbol rate, source")
			self.symrateviewrelay = pyui.number_viewer_text("Symbol rate, relay")
		elif self.conf.destui() == 'gui':
			self.main = QtGui.QWidget()
			self.layout = QtGui.QVBoxLayout(self.main)
			self.linklayout = QtGui.QHBoxLayout()

			self.perfview = perf_viewer()
			self.srcview = link_viewer("From source", self.conf.link_rate(), self.conf.symbol_length(), fft_size)
			self.relayview = link_viewer("From relay", self.conf.link_rate(), self.conf.symbol_length(), fft_size)

			self.layout.addWidget(self.perfview)
			self.layout.addLayout(self.linklayout)
			self.linklayout.addWidget(self.srcview)
			self.linklayout.addWidget(self.relayview)

			self.sigviewref = self.srcview.sigview
			self.sigviewrelay = self.relayview.sigview
			self.berviewdir = self.perfview.berviewdir
			self.berviewcoop = self.perfview.berviewcoop
			self.berviewdiravg = self.perfview.berviewdiravg
			self.berviewcoopavg = self.perfview.berviewcoopavg
			self.pktrateviewdir = self.perfview.pktrateviewdir
			self.pktrateviewcoop = self.perfview.pktrateviewcoop
			self.snrviewref = self.srcview.snrview
			self.snrviewrelay = self.relayview.snrview
			self.symrateviewref = self.srcview.rateview
			self.symrateviewrelay = self.relayview.rateview

			self.main.show()

		self.u3 = destnode(self, self.conf)

		self.connect((self.u3, 0), (self.sigviewref, 0))
		self.connect((self.u3, 1), (self.sigviewref, 1))
		self.connect((self.u3, 2), (self.sigviewrelay, 0))
		self.connect((self.u3, 3), (self.sigviewrelay, 1))

		self.updater = threading.Thread(target = self.update)
		self.updater.start()

	def update(self):
		while not self.finished:
			direct_ber = self.u3.direct_ber()
			coop_ber = self.u3.coop_ber()
			self.berviewdir.add_values(direct_ber)
			self.berviewdiravg.add_values(direct_ber)
			self.berviewcoop.add_values(coop_ber)
			self.berviewcoopavg.add_values(coop_ber)
			self.pktrateviewdir.set_value(self.u3.direct_packet_rate())
			self.pktrateviewcoop.set_value(self.u3.coop_packet_rate())
			self.snrviewref.add_values(self.u3.ref_snr())
			self.snrviewrelay.add_values(self.u3.relay_snr())
			self.symrateviewref.set_value(self.u3.ref_symbol_rate())
			self.symrateviewrelay.set_value(self.u3.relay_symbol_rate())
			time.sleep(1.0)

	def quit(self):
		self.finished = True
		self.u3.quit()
		self.stop()
		if self.app is not None:
			self.app.quit()

if __name__ == '__main__':
	gr.enable_realtime_scheduling()
	c = conf.conf('relay.cfg', 3)
	if c.destui() == 'none' or c.destui() == 'text':
		tb = destnode_top(c)
		tb.run()
	elif c.destui() == 'gui':
		app = QtGui.QApplication(sys.argv)
		tb = destnode_top(c, app)
		tb.start()
		app.exec_()
	else:
		print "Invalid user interface %s" % c.destui()

