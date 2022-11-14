#
# Copyright 2005, 2006, 2007 Free Software Foundation, Inc.
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

from math import pi
from gnuradio import gr
from packet_utils_dect import unmake_packet
import gnuradio.gr.gr_threading as _threading
from gnuradio import dectv2
import time




# /////////////////////////////////////////////////////////////////////////////
#                   mod/demod with packets as i/o
# /////////////////////////////////////////////////////////////////////////////



class mod_pkts(gr.hier_block2):
    """
    Wrap an arbitrary digital modulator in our packet handling framework.

    Send packets by calling send_pkt
    """
    def __init__(self, modulator, access_code=None, msgq_limit=2, pad_for_usrp=True, use_whitener_offset=False):
        """
	Hierarchical block for sending packets

        Packets to be sent are enqueued by calling send_pkt.
        The output is the complex modulated signal at baseband.

        @param modulator: instance of modulator class (gr_block or hier_block2)
        @type modulator: complex baseband out
        @param access_code: AKA sync vector
        @type access_code: string of 1's and 0's between 1 and 64 long
        @param msgq_limit: maximum number of messages in message queue
        @type msgq_limit: int
        @param pad_for_usrp: If true, packets are padded such that they end up a multiple of 128 samples
        @param use_whitener_offset: If true, start of whitener XOR string is incremented each packet
        
        See gmsk_mod for remaining parameters
        """

	gr.hier_block2.__init__(self, "mod_pkts",
				gr.io_signature(0, 0, 0),                    # Input signature
				gr.io_signature(1, 1, gr.sizeof_gr_complex)) # Output signature

        self._modulator = modulator
        self._pad_for_usrp = pad_for_usrp
        self._use_whitener_offset = use_whitener_offset
        self._whitener_offset = 0
        
        if access_code is None:
            raise ValueError, "No access_code specified"
        self._access_code = access_code

        # accepts messages from the outside world
        self._pkt_input = gr.message_source(gr.sizeof_char, msgq_limit)
        self.connect(self._pkt_input, self._modulator, self)

class demod_dect_pkts(gr.hier_block2):
    """
    Wrap an arbitrary digital demodulator in our packet handling framework.

    The input is complex baseband.  When packets are demodulated, they are passed to the
    app via the callback.
    """

    def __init__(self, demodulator, access_code=None, callback=None,tune_callback=None,stop_callback=None, threshold=-1):
        """
	Hierarchical block for demodulating and deframing packets.

	The input is the complex modulated signal at baseband.
        Demodulated packets are sent to the handler.

        @param demodulator: instance of demodulator class (gr_block or hier_block2)
        @type demodulator: complex baseband in
        @param access_code: AKA sync vector
        @type access_code: string of 1's and 0's
        @param callback:  function of two args: ok, payload
        @type callback: ok: bool; payload: string
        @param threshold: detect access_code with up to threshold bits wrong (-1 -> use default)
        @type threshold: int
	"""

	gr.hier_block2.__init__(self, "demod_pkts",
				gr.io_signature(1, 1, gr.sizeof_gr_complex), # Input signature
				gr.io_signature(0, 0, 0))                    # Output signature

        self._demodulator = demodulator
        if access_code is None:
            raise ValueError, "No access_code specified"
        self._access_code = access_code

        if threshold == -1:
            threshold = 0              # FIXME raise exception

        self._rcvd_pktq = gr.msg_queue()          # holds packets from the PHY
        self.correlator = dectv2.correlate_access_code_dect(access_code, threshold)
        #
        self.framer_sink = dectv2.framer_sink_dect(self._rcvd_pktq)
        self.connect(self, self._demodulator, self.correlator, self.framer_sink)
        self._watcher = _queue_watcher_thread(self._rcvd_pktq, callback)





        


    def set_code(self, access_code):

	#return dectv1.dectv1_correlate_access_code_dect.set_access_code(access_code)
	#return dectv1.set_access_code(access_code)
      return self.correlator.set_access_code(access_code)                                
	
    


    def set_code_n(self, access_code):

	#return dectv1.dectv1_correlate_access_code_dect.set_access_code(access_code)
	#return dectv1.set_access_code(access_code)
      return self.correlator.set_access_code_n(access_code)                                

	





class _queue_watcher_thread(_threading.Thread):
    def __init__(self, payload_pktq, callback):
        _threading.Thread.__init__(self)
        self.setDaemon(1)
        self.payload_pktq = payload_pktq
        self.callback = callback
        self.keep_running = True
        self.start()



    def run(self):
        while self.keep_running:
            #print "WAITING FOR PKT",self.payload_pktq.count()
            msg = self.payload_pktq.delete_head()
            #print "msg length =", len(msg.to_string())
            if (len(msg.to_string())!=0):
              ok, payload = unmake_packet(msg.to_string(), int(msg.arg1()),False)
              if self.callback:
                self.callback(ok, payload)
