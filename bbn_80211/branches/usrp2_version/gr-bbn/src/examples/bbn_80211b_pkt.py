#
# Copyright 2005 Free Software Foundation, Inc.
#
# Copyright (c) 2006 BBN Technologies Corp.  All rights reserved.
# Effort sponsored in part by the Defense Advanced Research Projects
# Agency (DARPA) and the Department of the Interior National Business
# Center under agreement number NBCHC050166.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
# 

from math import pi
import Numeric
import struct
from Queue import Queue

from gnuradio import gr, packet_utils
from gnuradio import bbn
from bbn_80211b import *

import gnuradio.gr.gr_threading as _threading
import bbn_80211b


# /////////////////////////////////////////////////////////////////////////////
#                   xPSK mod/demod with packets as i/o
# /////////////////////////////////////////////////////////////////////////////

class bbn_80211b_mod_pkts(gr.hier_block2):
  """
  PSK modulator that is a GNU Radio source.

  Send packets by calling send_pkt
  """
  def __init__(self, spb, alpha, gain, use_barker=0, msgq_limit=2):
      
    """
    Hierarchical block for PSK modulation.

    Packets to be sent are enqueued by calling send_pkt.
    The output is the complex modulated signal at baseband.

    @param fg: flow graph
    @type fg: flow graph
    @param msgq_limit: maximum number of messages in message queue
    @type msgq_limit: int

    See xpsk_mod for remaining parameters
    """
    gr.hier_block2.__init__(self, "bbn_80211b_mod_pkts", gr.io_signature(0, 0, 0), gr.io_signature(1, 1, gr.sizeof_gr_complex))
    self.spb = spb
    self.alpha = alpha
    self.gain = gain
    

    
    
    
    #thread to handle packet sending
    
    self.send_pkt_queue = Queue(msgq_limit)
    #sources for all the different bit rates and the scrambler
    self.pkt_input = gr.message_source(gr.sizeof_char, msgq_limit)
    self.pkt_input_h1 = gr.message_source(gr.sizeof_char, msgq_limit)
    self.pkt_input_h2 = gr.message_source(gr.sizeof_char, msgq_limit)
    self.pkt_input_1 = gr.message_source(gr.sizeof_char, msgq_limit)
    self.pkt_input_2 = gr.message_source(gr.sizeof_char, msgq_limit)
    self.final_tx_source = gr.message_source(gr.sizeof_gr_complex, 5)
    
    #sinks for different bit rates and scrambler
    self.header_queue_1 = gr.msg_queue(5)
    self.header_sink_1 = gr.message_sink(gr.sizeof_gr_complex, self.header_queue_1, False)
    self.header_queue_2 = gr.msg_queue(5)
    self.header_sink_2 = gr.message_sink(gr.sizeof_gr_complex, self.header_queue_2, False)
    
    self.payload_queue_1 = gr.msg_queue(5)
    self.payload_sink_1 = gr.message_sink(gr.sizeof_gr_complex, self.payload_queue_1, False)
    self.payload_queue_2 = gr.msg_queue(5)
    self.payload_sink_2 = gr.message_sink(gr.sizeof_gr_complex, self.payload_queue_2, False)
    
    self.scrambler_queue = gr.msg_queue(5)
    self.scrambler_sink = gr.message_sink(gr.sizeof_char, self.scrambler_queue, False)   
    #used to match the header and payload diff symbols
    self.last_payload_1 = 0
    self.last_payload_2 = 0
    #processing blocks
    self.scrambler = bbn.scrambler_bb(True)
    self.xpsk_mod_header_1 = bbn_80211b.bbn_80211b_mod(bpsk=True)
    self.xpsk_mod_header_2 = bbn_80211b.bbn_80211b_mod(bpsk=False)
    self.xpsk_mod_payload_1 = bbn_80211b.bbn_80211b_mod(bpsk=True)
    self.xpsk_mod_payload_2 = bbn_80211b.bbn_80211b_mod(bpsk=False)
    
    ntaps = 2 * self.spb - 1
    self.rrc_taps = gr.firdes.root_raised_cosine(
		4 * self.gain,     	# gain  FIXME may need to be spb
		self.spb,            # sampling freq
		1.0,		# symbol_rate
		alpha,
    ntaps)
    

    chip_rate = 11e6
    interpolate_rate = 25
    decimation_rate = 11
    sample_rate = chip_rate * interpolate_rate
    self.rate_change_filter = gr.firdes.low_pass_2(15, sample_rate, sample_rate / (2 * 25), .15 * sample_rate / 25, 7)
    #self.up = gr.interp_fir_filter_ccf(25, self.rate_change_filter)
    #self.down = gr.keep_one_in_n(gr.sizeof_gr_complex, 11)
    self.resample = gr.rational_resampler_base_ccf(25, 11, self.rate_change_filter)

    if use_barker:
      self.barker_taps = (1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0)
      self.tx_filter = gr.interp_fir_filter_ccf(11, self.barker_taps)
    else:
      self.tx_filter = gr.interp_fir_filter_ccf(self.spb, self.rrc_taps)
      
    #self.test_source = gr.vector_source_c([1.0 + 1.0j, 1.0 - 1.0j, 1.0, -1.0,1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0], False, 22)
    #self.stream_source = gr.vector_to_stream(gr.sizeof_gr_complex, 22)
    #self.out = gr.file_sink(gr.sizeof_gr_complex, "log.m")
    #self.new_slicer = bbn.slicer_cc(11, 16)
    #self.connect(self.test_source, self.stream_source, self.new_tx_filter, self.up, self.down, self.up2, self.down2, self.new_slicer, self.out)
    
    
    #connect everything together
    self.connect(self.pkt_input, self.scrambler, self.scrambler_sink)
    self.connect(self.pkt_input_h1, self.xpsk_mod_header_1, self.header_sink_1)
    self.connect(self.pkt_input_h2, self.xpsk_mod_header_2, self.header_sink_2)
    self.connect(self.pkt_input_1, self.xpsk_mod_payload_1, self.payload_sink_1)
    self.connect(self.pkt_input_2, self.xpsk_mod_payload_2, self.payload_sink_2)
    #self.connect(self.final_tx_source, self.tx_filter, self.up, self.down, self)
    self.connect(self.final_tx_source, self.tx_filter, self.resample, self)
    
    self.send_thread = _send_packet_thread(self)
    
  #send packet interface.  To send packets use this!!!!! send rate is 1 for 1Mbps and 2 for 2Mbps
  def send_pkt(self, payload='', send_rate=0, eof=False):
    payload_tuple = (payload, send_rate, eof)
    self.send_pkt_queue.put(payload_tuple, True)   
  #actually sends the packet to the phy
  def send_pkt_to_phy(self, payload='', send_rate=0, eof=False):
    """
    Send the payload.

    @param payload: data to send
    @type payload: string
    """
    if eof:
      msg = gr.message(1) # tell self.pkt_input we're not sending any more packets
    else:     
      
      short_preamble = 1  # allow short preamble

      # The long preamble uses 16 bytes of scrambled 1s with the proper SFD and
      # transmit the PLCP header at 1Mbps
      if short_preamble == 0:
        sync = chr(0xff) * 16
        start_frame_delim = chr(0xa0) + chr(0xf3)
      # If a short preamble, use 7 bytes of scrambled zeros and the proper SFD and
      # transmit the PCLP header at 2Mbps
      else:
        sync = chr(0x00) * 7
        start_frame_delim = chr(0xcf) + chr(0x05)

      preamble = sync + start_frame_delim
      if send_rate == 1:
        bits_per_chunk = 1
        xpsk_mod_payload = self.xpsk_mod_payload_1
        pkt_input_payload = self.pkt_input_1
        payload_queue  = self.payload_queue_1
        last_payload = self.last_payload_1
        signal = chr(0x0A) # 0x0A = 1 Mpbs, 0x14 = 2 Mbps
        length = chr(((len(payload) + 4)<< 3) & 0xff) + \
                 chr(((len(payload) + 4)>> 5) & 0xff)
      elif send_rate == 2:
        bits_per_chunk = 2
        xpsk_mod_payload = self.xpsk_mod_payload_2
        pkt_input_payload = self.pkt_input_2
        payload_queue = self.payload_queue_2
        last_payload = self.last_payload_2
        signal = chr(0x14) # 0x0A = 1 Mpbs, 0x14 = 2 Mbps
        length = chr(((len(payload) + 4)<< 2) & 0xff) + \
                 chr(((len(payload) + 4)>> 6) & 0xff)
      
      service = chr(0x00) # 802.11 Original Spec
      

      plcp_header = signal + service + length

      plcp_crc = bbn.crc16(plcp_header)
      plcp_header += chr(plcp_crc & 0xff) + chr(plcp_crc >> 8)

      payload_crc = bbn.crc32_le(payload)
      payload_crc_str = chr((payload_crc >> 0) & 0xff) + chr((payload_crc >> 8) & 0xff) + chr((payload_crc >> 16) & 0xff) + chr((payload_crc >> 24) & 0xff)
      msg = gr.message_from_string(preamble + plcp_header + payload + payload_crc_str + chr(0)*7);
      preamble_msg = gr.message_from_string(preamble)
      plcp_header_msg = gr.message_from_string(plcp_header)
      header_msg = gr.message_from_string(preamble + plcp_header)
      payload_msg = gr.message_from_string(payload + payload_crc_str + chr(0)*7)
      
    
    if eof:
      #close out queues
      self.xpsk_mod_header_1.bytes2chunks_src.msgq().insert_tail(msg)
      self.xpsk_mod_header_1.diff_encoded_src.msgq().insert_tail(msg)
      self.xpsk_mod_header_2.bytes2chunks_src.msgq().insert_tail(msg)
      self.xpsk_mod_header_2.diff_encoded_src.msgq().insert_tail(msg)
      self.xpsk_mod_payload_1.bytes2chunks_src.msgq().insert_tail(msg)
      self.xpsk_mod_payload_1.diff_encoded_src.msgq().insert_tail(msg)
      self.xpsk_mod_payload_2.bytes2chunks_src.msgq().insert_tail(msg)
      self.xpsk_mod_payload_2.diff_encoded_src.msgq().insert_tail(msg)
      self.final_tx_source.msgq().insert_tail(msg)
      self.pkt_input_h1.msgq().insert_tail(msg)
      self.pkt_input_h2.msgq().insert_tail(msg)
      self.pkt_input_1.msgq().insert_tail(msg)
      self.pkt_input_2.msgq().insert_tail(msg)
      self.pkt_input.msgq().insert_tail(msg)
      return  
    
    self.pkt_input.msgq().insert_tail(msg)
    msg_size = msg.length()
    
    scrambled_msg = gr.message_from_string(self.extract_from_queue(self.scrambler_queue, msg_size))
    if not self.scrambler_queue.empty_p():
      print 'problem, queue should be empty'
    
    self.pkt_input_h1.msgq().insert_tail(scrambled_msg)    
    pkt_input_payload.msgq().insert_tail(scrambled_msg)
        
    #crazy queues to merge multiple bit rates. . .
    
    
    

    original_length = msg.length()    
    
    # Extract the preamble and the PLCP header
    msg_size = msg.length() * 8 * gr.sizeof_char
    header_chunks = self.extract_from_queue(self.xpsk_mod_header_1.bytes2chunks_queue, msg_size)
    if not self.xpsk_mod_header_1.bytes2chunks_queue.empty_p():
      print 'problem, queue should be empty'

    # Only take care of the preamble first, which should be modulated at 1Mbps for both long and short
    # preamble.
    preamble_len = preamble_msg.length() * 8 * gr.sizeof_char
    plcp_len = plcp_header_msg.length() * 8 * gr.sizeof_char
    header_len = header_msg.length() * 8 * gr.sizeof_char
    preamble_chunks = header_chunks[0: preamble_len]
    plcp_chunks = header_chunks[preamble_len: preamble_len+plcp_len]
    self.xpsk_mod_header_1.bytes2chunks_src.msgq().insert_tail(gr.message_from_string(preamble_chunks))

    # Now modulate the PLCP header based on long (1Mbps) or short (2Mbps) preamble
    if short_preamble == 0:
      self.xpsk_mod_header_1.bytes2chunks_src.msgq().insert_tail(gr.message_from_string(plcp_chunks))
    else:
      self.xpsk_mod_header_2.bytes2chunks_src.msgq().insert_tail(gr.message_from_string(plcp_chunks))

    #diff encode the header only. Use last symbol of the header to continue diff encoding the payload
    # If it is a long preamble, pull out the rest from the 1Mbps queue, otherwise pull it from the
    # 2Mbps queue
    if short_preamble == 0:
      header_chunks = self.extract_from_queue(self.xpsk_mod_header_1.diff_encoded_queue, header_len)    
      if not self.xpsk_mod_header_1.diff_encoded_queue.empty_p():
        print 'problem, queue should be empty'
      self.xpsk_mod_header_1.diff_encoded_src.msgq().insert_tail(gr.message_from_string(header_chunks))
    else:
      preamble_chunks = self.extract_from_queue(self.xpsk_mod_header_1.diff_encoded_queue, preamble_len)    
      plcp_chunks = self.extract_from_queue(self.xpsk_mod_header_2.diff_encoded_queue, plcp_len)
      header_chunks = preamble_chunks + plcp_chunks
      self.xpsk_mod_header_1.diff_encoded_src.msgq().insert_tail(gr.message_from_string(preamble_chunks))
      self.xpsk_mod_header_2.diff_encoded_src.msgq().insert_tail(gr.message_from_string(plcp_chunks))
    
    #handles appending the correct symbol infront of the payload so that diff encoding works properly
    M = bits_per_chunk * 2
    if send_rate == 1:
      append_chunk = chr(ord(header_chunks[len(header_chunks) - 1]) + M - last_payload)
    elif send_rate == 2:
      if header_chunks[len(header_chunks) - 1] == chr(0x1):
        append_chunk = chr(2 + (M - last_payload))
      elif header_chunks[len(header_chunks) - 1] == chr(0x0):
        append_chunk = chr(M - last_payload)
      else:
        print 'unknown symbol at end of header'
      
    else:
      print "unsupported bit rate"

    msg_size = msg.length() * 8 * gr.sizeof_char / bits_per_chunk
    payload_chunks = self.extract_from_queue(xpsk_mod_payload.bytes2chunks_queue, msg_size)
    if not xpsk_mod_payload.bytes2chunks_queue.empty_p():
      print 'problem, queue should be empty'
    #add to payload for diff encoding
    payload_chunks = append_chunk + payload_chunks[(header_msg.length() * 8 * gr.sizeof_char / bits_per_chunk):]
   
    xpsk_mod_payload.bytes2chunks_src.msgq().insert_tail(gr.message_from_string(payload_chunks))

    #strip off the first symbol of the payload. This was only a pilot symbol to get the diff encode to work correctly
    msg_size = (payload_msg.length() * 8 / bits_per_chunk + 1) * gr.sizeof_char
    payload_chunks = self.extract_from_queue(xpsk_mod_payload.diff_encoded_queue, msg_size)
    payload_chunks = payload_chunks[1:]
    if not xpsk_mod_payload.diff_encoded_queue.empty_p():
      print 'problem, queue should be empty'
      
    xpsk_mod_payload.diff_encoded_src.msgq().insert_tail(gr.message_from_string(payload_chunks))
    #save the last element sent through the diff encoder
    if send_rate == 1:
      self.last_payload_1 =  ord(payload_chunks[len(payload_chunks) - 1])
    elif send_rate == 2:
      self.last_payload_2 =  ord(payload_chunks[len(payload_chunks) - 1])
    else:
      print 'error, unsupported bit rate'
    
    
    #extract the modulated header and payload and merge together
    msg_size = header_msg.length() * 8 * gr.sizeof_gr_complex

    # If long preamble, everything should be in header_queue_1
    if short_preamble == 0:
      header_chunks = self.extract_from_queue(self.header_queue_1, msg_size)
      if not self.header_queue_1.empty_p():
        print 'problem, queue should be empty'
    else:
      preamble_chunks = self.extract_from_queue(self.header_queue_1, preamble_len)
      plcp_chunks = self.extract_from_queue(self.header_queue_2, plcp_len)
      header_chunks = preamble_chunks + plcp_chunks
      if not self.header_queue_2.empty_p():
        print 'problem, queue should be empty'
   
    msg_size = payload_msg.length() * 8 * gr.sizeof_gr_complex / bits_per_chunk
    payload_chunks = self.extract_from_queue(payload_queue, msg_size)
    if not payload_queue.empty_p():
      print 'problem, queue should be empty'
    
    
    #gets sent to the RX filter/USRP
    
    self.final_tx_source.msgq().insert_tail(gr.message_from_string(header_chunks + payload_chunks))
    
  def extract_from_queue(self, queue, msg_size):
    total = 0
    msg_so_far = str()
    while total < msg_size:
      msg = queue.delete_head()
      total = total + msg.length()
      msg_so_far = msg_so_far + msg.to_string()
    
    return msg_so_far
    
class bbn_80211b_demod_pkts(gr.hier_block2):
   """
   PSK demodulator that is a GNU Radio sink.

   The input is complex baseband.  When packets are demodulated, they
   are passed to the app via the callback.  
   """
   def __init__(self, callback=None, spb=8, alpha=0.5, use_barker=0, check_crc=True):
    """
    Hierarchical block for PSK demodulation.

    The input is the complex modulated signal at baseband.
    Demodulated packets are sent to the handler.

    @param fg: flow graph
    @type fg: flow graph
    @param callback:  function of two args: ok, payload
    @type callback: ok: bool; payload: string

    See bbn_80211b_demod for remaining parameters.
    """

    self._rcvd_pktq = gr.msg_queue()          # holds packets from the PHY


    self._watcher = _queue_watcher_thread(self._rcvd_pktq, callback)
    ##
    ntaps = 2 * spb - 1

    self.rrc_taps = gr.firdes.root_raised_cosine(
      1,		# gain  FIXME may need to be spb
      spb,             # sampling freq
      1.0,		# symbol_rate
      alpha,
      ntaps)

    
    chip_rate = 25e6
    interpolate_rate = 11
    decimation_rate = 25
    sample_rate = chip_rate * interpolate_rate
    self.rate_change_filter2 = gr.firdes.low_pass_2(15, sample_rate, sample_rate / (2 * 11), .25 * sample_rate / 11, 9)
    #self.up2 = gr.interp_fir_filter_ccf(11, self.rate_change_filter2)
    #self.down2 = gr.keep_one_in_n(gr.sizeof_gr_complex, 25)
    self.resample = gr.rational_resampler_base_ccf(11, 25, self.rate_change_filter2)
        
    self.agc = gr.feedforward_agc_cc(25, 2.0)
    self.slicer = bbn.slicer_cc(11, 16)
    if use_barker == 1:
      self.barker_taps = (-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1)
      self.rx_filter = gr.fir_filter_ccf(1, self.barker_taps)    
    else:
      self.rx_filter = gr.fir_filter_ccf(1, self.rrc_taps)
      self.slicer = bbn.slicer_cc(spb, 16);

    
    self.demod = bbn.dpsk_demod_cb();
    self.descramble = bbn.scrambler_bb(False);
    self.plcp = bbn.plcp80211_bb(self._rcvd_pktq, check_crc);

    
    ##
    gr.hier_block2.__init__(self, "bbn_80211b_demod_pkts", gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(0, 0, 0))
    self.out = gr.file_sink(gr.sizeof_gr_complex, "log.m")
    #self.connect(self.down2, self.out)
    #self.connect(self, self.up2, self.down2, self.rx_filter, self.slicer)
    self.connect(self, self.resample, self.rx_filter, self.slicer)
    self.connect(self.slicer, self.demod)
    self.connect((self.demod, 0), (self.plcp, 0));
    self.connect((self.demod, 1), (self.plcp, 1));
    bbn.crc16_init()



class _queue_watcher_thread(_threading.Thread):
  def __init__(self, rcvd_pktq, callback):
    _threading.Thread.__init__(self)
    self.setDaemon(1)
    self.rcvd_pktq = rcvd_pktq
    self.callback = callback
    self.keep_running = True
    self.start()

  def stop(self):
    self.keep_running = False

  def run(self):
    while self.keep_running:
      payload = self.rcvd_pktq.delete_head().to_string()
      if self.callback:
        self.callback(True, payload)

#thread handles all the sending processing
class _send_packet_thread(_threading.Thread):
  def __init__(self, pkt_block):
    _threading.Thread.__init__(self)
    self.setDaemon(1)
    self.pkt_block = pkt_block
    self.keep_running = True
    self.start()

  def stop(self):
    self.keep_running = False

  def run(self):
    while self.keep_running:
      pkt_tuple = self.pkt_block.send_pkt_queue.get(True)
      
      self.pkt_block.send_pkt_to_phy(pkt_tuple[0], pkt_tuple[1], pkt_tuple[2])

