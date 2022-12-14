#
# Copyright 2011 FOI
# 
# Copyright 2010 A.Kaszuba, R.Checinski, MUT 
#
# Copyright 2008, 2009 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

# This is a modification of packet.py from GNU Radio.

from gnuradio import gr, packet_utils
import gnuradio.gr.gr_threading as _threading

##payload length in bytes
DEFAULT_PAYLOAD_LEN = 512

##how many messages in a queue
DEFAULT_MSGQ_LIMIT = 2

##threshold for unmaking packets
DEFAULT_THRESHOLD = 12

##################################################
## Options Class for OFDM
##################################################
class options(object):
	def __init__(self, **kwargs):
		for key, value in kwargs.iteritems(): setattr(self, key, value)

##################################################
## Packet Encoder
##################################################
class _packet_encoder_thread(_threading.Thread):

	def __init__(self, msgq, payload_length, send):
		self._msgq = msgq
		self._payload_length = payload_length
		self._send = send
		_threading.Thread.__init__(self)
		self.setDaemon(1)
		self.keep_running = True
		self.start()

	def run(self):
		sample = '' #residual sample
		while self.keep_running:
			msg = self._msgq.delete_head() #blocking read of message queue
			sample = sample + msg.to_string() #get the body of the msg as a string
			while len(sample) >= self._payload_length:
				payload = sample[:self._payload_length]
				sample = sample[self._payload_length:]
				self._send(payload)

class packet_encoder(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet-based modulators.
	"""

	def __init__(self, samples_per_symbol, bits_per_symbol, access_code='', pad_for_usrp=True):
		"""
		packet_mod constructor.
		@param samples_per_symbol number of samples per symbol
		@param bits_per_symbol number of bits per symbol
		@param access_code AKA sync vector
		@param pad_for_usrp If true, packets are padded such that they end up a multiple of 128 samples
		@param payload_length number of bytes in a data-stream slice
		"""
		#setup parameters
		self._samples_per_symbol = samples_per_symbol
		self._bits_per_symbol = bits_per_symbol
		self._pad_for_usrp = pad_for_usrp
		if not access_code: #get access code
			access_code = packet_utils.default_access_code
		if not packet_utils.is_1_0_string(access_code):
			raise ValueError, "Invalid access_code %r. Must be string of 1's and 0's" % (access_code,)
		self._access_code = access_code
		self._pad_for_usrp = pad_for_usrp
		#create blocks
		msg_source = gr.message_source(gr.sizeof_char, DEFAULT_MSGQ_LIMIT)
		self._msgq_out = msg_source.msgq()
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"packet_encoder",
			gr.io_signature(0, 0, 0), # Input signature
			gr.io_signature(1, 1, gr.sizeof_char) # Output signature
		)
		#connect
		self.connect(msg_source, self)

	def send_pkt(self, payload):
		"""
		Wrap the payload in a packet and push onto the message queue.
		@param payload string, data to send
		"""
		packet = packet_utils.make_packet(
			payload,
			self._samples_per_symbol,
			self._bits_per_symbol,
			self._access_code,
			self._pad_for_usrp
		)
		msg = gr.message_from_string(packet)
		self._msgq_out.insert_tail(msg)

##################################################
## Packet Decoder
##################################################
class _packet_decoder_thread(_threading.Thread):

	def __init__(self, msgq, callback):
		_threading.Thread.__init__(self)
		self.setDaemon(1)
		self._msgq = msgq
		self.callback = callback
		self.keep_running = True
		self.start()

	def run(self):
		while self.keep_running:
			msg = self._msgq.delete_head()
			ok, payload = packet_utils.unmake_packet(msg.to_string(), int(msg.arg1()))
			if self.callback:
				self.callback(ok, payload)

class packet_decoder(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet-based demodulators.
	"""

	def __init__(self, access_code='', threshold=-1, callback=None):
		"""
		packet_demod constructor.
		@param access_code AKA sync vector
		@param threshold detect access_code with up to threshold bits wrong (0 -> use default)
		@param callback a function of args: ok, payload
		"""
		#access code
		if not access_code: #get access code
			access_code = packet_utils.default_access_code
		if not packet_utils.is_1_0_string(access_code):
			raise ValueError, "Invalid access_code %r. Must be string of 1's and 0's" % (access_code,)
		self._access_code = access_code
		#threshold
		if threshold < 0: threshold = DEFAULT_THRESHOLD
		self._threshold = threshold
		#blocks
		msgq = gr.msg_queue(DEFAULT_MSGQ_LIMIT) #holds packets from the PHY
		correlator = gr.correlate_access_code_bb(self._access_code, self._threshold)
		framer_sink = gr.framer_sink_1(msgq)
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"packet_decoder",
			gr.io_signature(1, 1, gr.sizeof_char), # Input signature
			gr.io_signature(0, 0, 0) # Output signature
		)
		#connect
		self.connect(self, correlator, framer_sink)
		#start thread
		_packet_decoder_thread(msgq, callback)

##################################################
## Packet Mod for OFDM Mod and Packet Encoder
##################################################
class packet_mimo_mod_base(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet source block.
	"""

	def __init__(self, packet_source=None, payload_length=0):
		if not payload_length: #get payload length
			payload_length = DEFAULT_PAYLOAD_LEN
		if payload_length%self._item_size_in != 0:	#verify that packet length is a multiple of the stream size
			raise ValueError, 'The payload length: "%d" is not a multiple of the stream size: "%d".'%(payload_length, self._item_size_in)
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"ofdm_mimo_mod",
			gr.io_signature(1, 1, self._item_size_in), # Input signature
			gr.io_signature(2, 2, packet_source._hb.output_signature().sizeof_stream_item(0)) # Output signature
		)
		#create blocks
		msgq = gr.msg_queue(DEFAULT_MSGQ_LIMIT)
		msg_sink = gr.message_sink(self._item_size_in, msgq, False) #False -> blocking
		#connect
		self.connect(self, msg_sink)
		self.connect((packet_source,0), (self,0))
		self.connect((packet_source,1), (self,1))		
		#start thread
		_packet_encoder_thread(msgq, payload_length, packet_source.send_pkt)

class packet_mimo_mod_b(packet_mimo_mod_base): _item_size_in = gr.sizeof_char
class packet_mimo_mod_s(packet_mimo_mod_base): _item_size_in = gr.sizeof_short
class packet_mimo_mod_i(packet_mimo_mod_base): _item_size_in = gr.sizeof_int
class packet_mimo_mod_f(packet_mimo_mod_base): _item_size_in = gr.sizeof_float
class packet_mimo_mod_c(packet_mimo_mod_base): _item_size_in = gr.sizeof_gr_complex

##################################################
## Packet Demod for OFDM Demod and Packet Decoder
##################################################
class packet_mimo_demod_base(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet sink block.
	"""

	def __init__(self, packet_sink=None):
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"ofdm_mimo_mod",
			gr.io_signature(2, 2, packet_sink._hb.input_signature().sizeof_stream_item(0)), # Input signature
			gr.io_signature(1, 1, self._item_size_out) # Output signature
		)
		#create blocks
		msg_source = gr.message_source(self._item_size_out, DEFAULT_MSGQ_LIMIT)
		self._msgq_out = msg_source.msgq()
		#connect
		self.connect((self,0), (packet_sink,0))
		self.connect((self,1), (packet_sink,1))
		#self.connect(self, packet_sink)
                self.connect(msg_source, self)
		if packet_sink._hb.output_signature().sizeof_stream_item(0):
			self.connect(packet_sink, gr.null_sink(packet_sink._hb.output_signature().sizeof_stream_item(0)))

	def recv_pkt(self, ok, payload):
		msg = gr.message_from_string(payload, 0, self._item_size_out, len(payload)/self._item_size_out)
		if ok: self._msgq_out.insert_tail(msg)

class packet_mimo_demod_b(packet_mimo_demod_base): _item_size_out = gr.sizeof_char
class packet_mimo_demod_s(packet_mimo_demod_base): _item_size_out = gr.sizeof_short
class packet_mimo_demod_i(packet_mimo_demod_base): _item_size_out = gr.sizeof_int
class packet_mimo_demod_f(packet_mimo_demod_base): _item_size_out = gr.sizeof_float
class packet_mimo_demod_c(packet_mimo_demod_base): _item_size_out = gr.sizeof_gr_complex
