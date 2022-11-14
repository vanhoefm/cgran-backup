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

from gnuradio import gr, packet_utils
from gnuradio import bbn
from bbn_80211b import *

import gnuradio.gr.gr_threading as _threading
import bbn_80211b


# /////////////////////////////////////////////////////////////////////////////
#                   xPSK mod/demod with packets as i/o
# /////////////////////////////////////////////////////////////////////////////

class bbn_80211b_mod_pkts(gr.hier_block):
    """
    PSK modulator that is a GNU Radio source.

    Send packets by calling send_pkt
    """
    def __init__(self, fg, msgq_limit=2, pad_for_usrp=True, *args, **kwargs):
        
        """
	Hierarchical block for PSK modulation.

        Packets to be sent are enqueued by calling send_pkt.
        The output is the complex modulated signal at baseband.

	@param fg: flow graph
	@type fg: flow graph
        @param msgq_limit: maximum number of messages in message queue
        @type msgq_limit: int
        @param pad_for_usrp: If true, packets are padded such that they end up a multiple of 128 samples

        See xpsk_mod for remaining parameters
        """
        #self.pad_for_usrp = pad_for_usrp

        # accepts messages from the outside world
        self.pkt_input = gr.message_source(gr.sizeof_char, msgq_limit)

        self.xpsk_mod = bbn_80211b.bbn_80211b_mod(fg, *args, **kwargs)
        fg.connect(self.pkt_input, self.xpsk_mod)
        gr.hier_block.__init__(self, fg, None, self.xpsk_mod)

    def send_pkt(self, payload='', eof=False):
        """
        Send the payload.

        @param payload: data to send
        @type payload: string
        """
        if eof:
            msg = gr.message(1) # tell self.pkt_input we're not sending any more packets
        else:
            sync = chr(0xff) * 16
            start_frame_delim = chr(0xa0) + chr(0xf3)
            preamble = sync + start_frame_delim

            signal = chr(0x0A) # 0x0A = 1 Mpbs, 0x14 = 2 Mbps
            service = chr(0x00) # 802.11 Original Spec
            length = chr(((len(payload) + 4)<< 3) & 0xff) + \
                     chr(((len(payload) + 4)>> 5) & 0xff)

            plcp_header = signal + service + length
            
            plcp_crc = bbn.crc16(plcp_header)
            plcp_header += chr(plcp_crc & 0xff) + chr(plcp_crc >> 8)

            payload_crc = bbn.crc32_le(payload)
            payload_crc_str = chr((payload_crc >> 0) & 0xff) + \
                              chr((payload_crc >> 8) & 0xff) + \
                              chr((payload_crc >> 16) & 0xff) + \
                              chr((payload_crc >> 24) & 0xff)
            msg = gr.message_from_string(preamble + plcp_header +
                                         payload + payload_crc_str +\
                                         chr(0)*7);

        self.pkt_input.msgq().insert_tail(msg)

class bbn_80211b_demod_pkts(gr.hier_block):
    """
    PSK demodulator that is a GNU Radio sink.

    The input is complex baseband.  When packets are demodulated, they
    are passed to the app via the callback.  """

    def __init__(self, fg, callback=None, spb=8, alpha=0.5, *args, **kwargs):
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
        self.bit_receiver = bbn_80211b_demod(fg, spb=spb, alpha = alpha,
                                             pkt_queue=self._rcvd_pktq,
                                             *args, **kwargs)
        
        gr.hier_block.__init__(self, fg, self.bit_receiver, None)
        self._watcher = _queue_watcher_thread(self._rcvd_pktq, callback)

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
