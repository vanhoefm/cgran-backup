#!/usr/bin/env python

from gnuradio import gr
from openrd import pr, conf, dmod, link, pyblk

class srcnode(gr.hier_block2):
	def __init__(self, top, c):
		gr.hier_block2.__init__(self, "srcnode",
				gr.io_signature(0, 0, 0),
				gr.io_signature(0, 0, 0))

		self.conf = c
		
		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			from openrd import vtapp

		# Instantiate data source
		if self.conf.source_data_mode() == pr.SRCMODE_ZERO:
			self.datasrc = pr.data_source_zero(self.conf.block_size())
		elif self.conf.source_data_mode() == pr.SRCMODE_COUNTER:
			self.datasrc = pr.data_source_counter(self.conf.block_size())
		elif self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.datasrc = pr.data_source_packet(vtapp.num_proto(self.conf.application()), 
					self.conf.block_size(), pr.PACKET_SOURCE_BLOCK)
		else:
			sys.exit(1)

		# Instantiate source processing blocks
		self.source = pyblk.source_ref(self.conf.block_size(), 
				self.conf.coding(), self.conf.framing(), 
				self.conf.frame_size(), self.conf.modulation())
		self.xmit = dmod.transmitter_cc(self.conf.modulation(), self.conf.symbol_length())

		# Instantiate link
		if self.conf.link() == 'socket':
			self.utx = link.txsocket_c(self.conf.socket_conn(), 3, self.conf.link_rate())
			raw_input('Start all units, then press enter')
			self.utx.connect_blocks()
		elif self.conf.link() == 'wired':
			# Not tested
			self.utx = link.txusrp_c(self.conf.usrp_serial(), 3, self.conf.link_rate())
			self.utx.tune(1, self.conf.usrp_carrier())
			self.utx.tune(2, self.conf.usrp_carrier())
		elif self.conf.link() == 'wireless':
			self.utx = link.txusrp_c(self.conf.usrp_serial(), 1, self.conf.link_rate())
			self.utx.tune(1, self.conf.usrp_carrier())
		else:
			sys.exit(1)

		# Instantiate control interface
		if self.conf.control() == pr.CNTRLMODE_NONE:
			self.control = conf.control_none()
		elif self.conf.control() == pr.CNTRLMODE_CONSOLE:
			self.control = conf.control_console()
		else:
			sys.exit(1)

		# Set controllable parameters
		self.control.set_disp_cmd(True)
		self.control.set_top(top)
		self.control.set_srctx(self.utx)
		self.control.start()

		# Connect the flow graph
		self.connect(self.datasrc, self.source, self.xmit)
		if self.conf.link() == 'socket' or self.conf.link() == 'wired':
			self.connect(self.xmit, (self.utx, 0))
			self.connect(self.xmit, (self.utx, 1))
		elif self.conf.link() == 'wireless':
			self.connect(self.xmit, self.utx)

		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.app = vtapp.src(self.datasrc, self.conf.application())
			self.app.set_size((80,60))
			self.app.start()

	def quit(self):
		if self.conf.source_data_mode() == pr.SRCMODE_PACKET:
			self.app.quit()

class srcnode_top(gr.top_block):
	def __init__(self, c):
		gr.top_block.__init__(self)

		self.conf = c

		self.u1 = srcnode(self, self.conf)
		self.connect(self.u1)
	
	def quit(self):
		self.u1.quit()
		self.stop()

if __name__ == '__main__':
	gr.enable_realtime_scheduling()

	# Load configuration
	c = conf.conf('relay.cfg', 1)

	tb = srcnode_top(c)
	tb.run()

