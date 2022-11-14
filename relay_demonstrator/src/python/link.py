#
# Copyright 2011 Anton Blad.
# 
# This file is part of OpenRD
# 
# OpenRD is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# OpenRD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from gnuradio import gr
from openrd import cusrp, pr
import socket
import struct

# Connection type to use for sockets: 'udp' or 'tcp'
socket_type = 'tcp'

# Number of header zeroes to inject when using sockets
socket_num_inject = 4096
socket_init_data = [0]*1024
socket_init_packet = struct.pack('f'*len(socket_init_data), *socket_init_data)

class txsocket_c(gr.hier_block2):
	def __init__(self, conn, mode, rate):
		self.conn = conn
		self.mode = mode
		self.rate = rate

		# Determine the number of inputs and outputs
		ntx = self.npaths(mode)
		self.ntx = ntx

		gr.hier_block2.__init__(self, "txsocket_c",
				gr.io_signature(ntx, ntx, gr.sizeof_gr_complex),
				gr.io_signature(0, 0, 0))

		# Create the client sockets
		csock = {}
		if ntx == 1:
			if socket_type == 'udp':
				s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			elif socket_type == 'tcp':
				s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			csock[0] = s
		else:
			if socket_type == 'udp':
				s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			elif socket_type == 'tcp':
				s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			csock[0] = s
			if socket_type == 'udp':
				s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			elif socket_type == 'tcp':
				s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			csock[1] = s
		
		self.csock = csock

	def connect_blocks(self):
		txfd = {}

		# Try to connect to receivers
		for k,s in self.csock.items():
			print "Connecting to %s:%d..." % (self.conn['txhost'][k], self.conn['txport'][k],),
			s.connect((self.conn['txhost'][k], self.conn['txport'][k]))
			print "Ok!"
			txfd[k] = s.fileno()

		self.txfd = txfd

		# Connect the sockets to the flow graph
		if self.ntx == 1:
			self.tx = pr.file_descriptor_sink(gr.sizeof_gr_complex, txfd[0])
			txthrottle = gr.throttle(gr.sizeof_gr_complex, self.rate)
			self.connect(self, txthrottle, self.tx)
		else:
			self.tx0 = pr.file_descriptor_sink(gr.sizeof_gr_complex, txfd[0])
			self.tx1 = pr.file_descriptor_sink(gr.sizeof_gr_complex, txfd[1])
			txthrottle0 = gr.throttle(gr.sizeof_gr_complex, self.rate)
			txthrottle1 = gr.throttle(gr.sizeof_gr_complex, self.rate)
			self.connect((self, 0), txthrottle0, self.tx0)
			self.connect((self, 1), txthrottle1, self.tx1)

	def npaths(self, mode):
		if mode == 0:
			n = 0
		elif mode == 1 or mode == 2:
			n = 1
		elif mode == 3:
			n = 2
		else:
			raise ValueError("Invalid mode")
		return n

class rxsocket_c(gr.hier_block2):
	def __init__(self, conn, mode, rate):
		self.conn = conn
		self.mode = mode
		self.rate = rate

		# Determine the number of inputs and outputs
		nrx = self.npaths(mode)
		self.nrx = nrx

		gr.hier_block2.__init__(self, "rxsocket_c",
				gr.io_signature(0, 0, 0),
				gr.io_signature(nrx, nrx, gr.sizeof_gr_complex))

		# Create the listening sockets
		lsock = {}
		if nrx == 1:
			if socket_type == 'udp':
				s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
				s.bind((socket.gethostname(), conn['rxport'][0]))
			elif socket_type == 'tcp':
				s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
				s.bind((socket.gethostname(), conn['rxport'][0]))
				s.listen(1)
			print "Listening on port %d" % conn['rxport'][0]
			lsock[0] = s
		else:
			if socket_type == 'udp':
				s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
				s.bind((socket.gethostname(), conn['rxport'][0]))
			elif socket_type == 'tcp':
				s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
				s.bind((socket.gethostname(), conn['rxport'][0]))
				s.listen(1)
			print "Listening on port %d" % conn['rxport'][0]
			lsock[0] = s
			if socket_type == 'udp':
				s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
				s.bind((socket.gethostname(), conn['rxport'][1]))
			elif socket_type == 'tcp':
				s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
				s.bind((socket.gethostname(), conn['rxport'][1]))
				s.listen(1)
			print "Listening on port %d" % conn['rxport'][1]
			lsock[1] = s

		self.lsock = lsock

	def connect_blocks(self):
		rxfd = {}
		# Accept the client connection
		if socket_type == 'udp':
			for k,s in self.lsock.items():
				rxfd[k] = self.lsock[k].fileno()
		elif socket_type == 'tcp':
			csock = {}
			for k,s in self.lsock.items():
				(client, addr) = self.lsock[k].accept()
				print "Connection received on port %d (from %s:%d)" % (self.lsock[k].getsockname()[1], addr[0], addr[1])
				csock[k] = client
				rxfd[k] = client.fileno()
			self.csock = csock
			del self.lsock

		self.rxfd = rxfd

		# Connect the sockets to the flow graph
		# The file_descriptor_source blocks when there is no data available.
		# Thus, we inject some zeroes to ensure that the rest of the flow
		# graph is started.
		if self.nrx == 1:
			self.rx = gr.file_descriptor_source(gr.sizeof_gr_complex, rxfd[0])
			rxinject = pr.insert_head(gr.sizeof_gr_complex, socket_num_inject)
			self.connect(self.rx, rxinject, self)
		else:
			self.rx0 = gr.file_descriptor_source(gr.sizeof_gr_complex, rxfd[0])
			self.rx1 = gr.file_descriptor_source(gr.sizeof_gr_complex, rxfd[1])
			rxinject0 = pr.insert_head(gr.sizeof_gr_complex, socket_num_inject)
			rxinject1 = pr.insert_head(gr.sizeof_gr_complex, socket_num_inject)
			self.connect(self.rx0, rxinject0, (self, 0))
			self.connect(self.rx1, rxinject1, (self, 1))

	def npaths(self, mode):
		if mode == 0:
			n = 0
		elif mode == 1 or mode == 2:
			n = 1
		elif mode == 3:
			n = 2
		else:
			raise ValueError("Invalid mode")
		return n

class txusrp_c(gr.hier_block2):
	def __init__(self, serial, mode, rate):
		self.serial = serial
		self.mode = mode
		self.rate = rate

		# Determine the number of inputs
		ntx = self.npaths(mode)
		self.ntx = ntx

		gr.hier_block2.__init__(self, "txusrp_c",
				gr.io_signature(ntx, ntx, gr.sizeof_gr_complex),
				gr.io_signature(0, 0, 0))

		# Determine interpolation rate
		interp_rate = self.interp_rate(rate)

		# Instantiate and set TX power
		if ntx == 1:
			self.txscale = gr.multiply_const_cc(0)
		elif ntx == 2:
			self.txscale0 = gr.multiply_const_cc(0)
			self.txscale1 = gr.multiply_const_cc(0)

		if ntx > 0:
			self.set_txpower(-20)

		# Instantiate the usrps
		if ntx > 0:
			self.utx = cusrp.sink_c(serial = serial, interp_rate = interp_rate, mode = mode)

		# Connect block inputs to usrp sink
		if ntx == 1:
			self.connect(self, self.txscale, self.utx)
		elif ntx == 2:
			self.connect((self, 0), self.txscale0, (self.utx, 0))
			self.connect((self, 1), self.txscale1, (self.utx, 1))

	def npaths(self, mode):
		if mode == 0:
			n = 0
		elif mode == 1 or mode == 2:
			n = 1
		elif mode == 3:
			n = 2
		else:
			raise ValueError("Invalid mode")
		return n

	def interp_rate(self, rate):
		if rate == 250000:
			int = 512
		elif rate == 500000:
			int = 256
		elif rate == 1000000:
			int = 128
		elif rate == 2000000:
			int = 64
		elif rate == 4000000:
			int = 32
		elif rate == 8000000:
			int = 16
		else:
			raise ValueError("Invalid rate")
		return int

	def set_txpower(self, txpower):
		scale = 1600000.0
		k = 10.0**((txpower*1.0-30.0)/20.0)*scale

		if k > 32000:
			raise ValueError("txpower too high")

		if self.ntx == 1:
			self.txscale.set_k(k)
		else:
			self.txscale0.set_k(k)
			self.txscale1.set_k(k)
			
	def tune(self, side, freq):
		self.utx.tune(side, freq)

class rxusrp_c(gr.hier_block2):
	def __init__(self, serial, mode, rate):
		self.serial = serial
		self.mode = mode
		self.rate = rate

		# Determine the number of inputs and outputs
		nrx = self.npaths(mode)
		self.nrx = nrx

		gr.hier_block2.__init__(self, "rxusrp_c",
				gr.io_signature(0, 0, 0),
				gr.io_signature(nrx, nrx, gr.sizeof_gr_complex))

		# Determine interpolation and decimation rate
		decim_rate = self.decim_rate(rate)

		# Instantiate and set RX gain
		if nrx == 1:
			self.rxscale = gr.multiply_const_cc(0)
		elif nrx == 2:
			self.rxscale0 = gr.multiply_const_cc(0)
			self.rxscale1 = gr.multiply_const_cc(0)

		if nrx > 0:
			self.set_rxgain(20)

		# Instantiate the usrps
		if nrx > 0:
			self.urx = cusrp.source_c(serial = serial, decim_rate = decim_rate, mode = mode)

		# Connect usrp source to block outputs
		if nrx == 1:
			self.connect(self.urx, self.rxscale, self)
		elif nrx == 2:
			self.connect((self.urx, 0), self.rxscale0, (self, 0))
			self.connect((self.urx, 1), self.rxscale1, (self, 1))

	def npaths(self, mode):
		if mode == 0:
			n = 0
		elif mode == 1 or mode == 2:
			n = 1
		elif mode == 3:
			n = 2
		else:
			raise ValueError("Invalid mode")
		return n

	def decim_rate(self, rate):
		if rate == 250000:
			dec = 256
		elif rate == 500000:
			dec = 128
		elif rate == 1000000:
			dec = 64
		elif rate == 2000000:
			dec = 32
		elif rate == 4000000:
			dec = 16
		elif rate == 8000000:
			dec = 8
		else:
			raise ValueError("Invalid rate")
		return dec

	def set_rxgain(self, rxgain):
		scale = 1.0/256.0
		k = 10.0**((rxgain*1.0-30.0)/20.0)*scale

		if self.nrx == 1:
			self.rxscale.set_k(k)
		else:
			self.rxscale0.set_k(k)
			self.rxscale1.set_k(k)

	def tune(self, side, freq):
		self.urx.tune(side, freq)

