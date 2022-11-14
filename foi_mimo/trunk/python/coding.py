#
# Copyright 2011 FOI
# 
# This file is part of FOI-MIMO
# 
# FOI-MIMO is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# FOI-MIMO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FOI-MIMO; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import math
from gnuradio import gr, trellis
import foimimo
import gnuradio.gr.gr_threading as _threading

class encoding(gr.hier_block2):
    
    def __init__(self,options,fsm,msgq_limit=2):
        gr.hier_block2.__init__(self,"encoding",
                            gr.io_signature(0,0,0),
                            gr.io_signature2(2,2,gr.sizeof_char,gr.sizeof_char))
        
        self._bytes_per_packet = int(options.size)
        self._encode_fsm = fsm
        
        in_bitspersymbol = int(round(math.log(self._encode_fsm.I())/math.log(2)))
        out_bitspersymbol = int(round(math.log(self._encode_fsm.O())/math.log(2)))
                
        self._crc32_inserter = foimimo.crc32_inserter(self._bytes_per_packet, msgq_limit)
        self._scrambler = foimimo.scrambler_bb(self._bytes_per_packet)
        self._byte2chunk = foimimo.byte_2_chunk(in_bitspersymbol)
        self._encoder = foimimo.trellis_encoder_bb(self._encode_fsm,0)
        self._chunk2byte = foimimo.chunk_2_byte_skip_head(out_bitspersymbol, in_bitspersymbol)
        
                      
        self.connect((self._crc32_inserter,0),(self._scrambler,0), 
                     (self._byte2chunk,0), (self._encoder,0), (self._chunk2byte,0), (self,0))
        self.connect((self._crc32_inserter,1),(self._scrambler,1), 
                     (self._byte2chunk,1), (self._encoder,1), (self._chunk2byte,1),(self,1))
                       
        if options.log:
            self.connect((self._crc32_inserter,0), gr.file_sink(gr.sizeof_char, "payload_with_crc32_b.dat"))
            self.connect((self._crc32_inserter,1), gr.file_sink(gr.sizeof_char, "payload_with_crc32_new_pkt_b.dat"))
            self.connect((self._scrambler,0), gr.file_sink(gr.sizeof_char, "scrambled_payload_b.dat"))
            self.connect((self._scrambler,1), gr.file_sink(gr.sizeof_char, "scrambled_payload_new_pkt_b.dat"))
            self.connect((self._byte2chunk,0), gr.file_sink(gr.sizeof_char, "byte2chunk_b.dat"));
            self.connect((self._byte2chunk,1), gr.file_sink(gr.sizeof_char, "byte2chunk_new_pkt_b.dat"));
            self.connect((self._encoder,0), gr.file_sink(gr.sizeof_char, "encoded_payload_b.dat"))
            self.connect((self._encoder,1), gr.file_sink(gr.sizeof_char, "encoded_payload_new_pkt_b.dat"))
            self.connect(self._chunk2byte, gr.file_sink(gr.sizeof_char, "chunk2byte_skip_head_b.dat")) 
            self.connect((self._chunk2byte,1), gr.file_sink(gr.sizeof_char, "chunk2byte_skip_head_new_pkt_b.dat"))          

    def set_pkt_size(self,pkt_size):
        self._crc32_inserter.set_pkt_size(pkt_size)
        
    def send_pkt(self, payload='',eof=False):
        if eof:
            msg = gr.message(1)
            self._crc32_inserter.msgq().insert_tail(msg)
        else:
            msg = gr.message_from_string(payload)
            self._crc32_inserter.msgq().insert_tail(msg)
            
class decoding(gr.hier_block2):
    def __init__(self,options, fsm, callback = None):
        gr.hier_block2.__init__(self,"decoding",
                                gr.io_signature2(2,2,gr.sizeof_gr_complex,gr.sizeof_char),
                                gr.io_signature(0,0,0))


        self._log = options.log
        self._encode_fsm = fsm
        self.dimensionallity = 2
        
        # QPSK modulation support only
        self._REtable = [0.70699999999999996, -0.70699999999999996]
        self._IMtable = [0.70699999999999996, -0.70699999999999996]
        
        self._bytes_per_packet = int(options.size)
        self._bitspersymbol = int(round(math.log(self._encode_fsm.I())/math.log(2)))
        self.K = int(math.ceil(((self._bytes_per_packet-4)*8.0)/self._bitspersymbol)) # packet size in trellis steps
        self._rcv_queue = gr.msg_queue()
        
        self.c2f = gr.complex_to_float(1)
        self.interleave_constp = gr.interleave(gr.sizeof_float)
        
        self.constant_zero = gr.null_source(gr.sizeof_char)
        self.interleave_newpkt = gr.interleave(gr.sizeof_char)

        self._metrics = foimimo.trellis_metrics_f(self._encode_fsm.O(),self.dimensionallity,
                                          self._REtable, self._IMtable)
        self._viterbi = foimimo.trellis_viterbi_b(self._encode_fsm,self.K,0,-1)
        
        self._chunk2byte = foimimo.chunk_2_byte(self._bitspersymbol)
        self._descrambler = foimimo.descrambler_bb()
        self._crc32_checker = foimimo.crc32_checker_sink(self._bytes_per_packet-4,self._rcv_queue)
        
        self.connect((self,0), self.c2f)
        self.connect((self.c2f,0), (self.interleave_constp,0))
        self.connect((self.c2f,1), (self.interleave_constp,1))
        
        self.connect((self,1),(self.interleave_newpkt,0))
        self.connect(self.constant_zero,(self.interleave_newpkt,1))
 
        self.connect(self.interleave_constp, (self._metrics,0), (self._viterbi,0),
                      (self._chunk2byte,0),(self._descrambler,0), (self._crc32_checker,0))
        self.connect(self.interleave_newpkt, (self._metrics,1), (self._viterbi,1),
                     (self._chunk2byte,1),(self._descrambler,1),(self._crc32_checker,1))
        
        if self._log:
            self.connect((self,1), gr.file_sink(gr.sizeof_char,"demapper_new_pkt_b.dat"));
            self.connect((self._descrambler,0), gr.file_sink(gr.sizeof_char,"descrambled_payload_crc32_b.dat"))
            self.connect((self._descrambler,1), gr.file_sink(gr.sizeof_char,"descrambled_new_pkt_b.dat"))
            self.connect((self.interleave_newpkt),gr.file_sink(gr.sizeof_char,"interleave_new_pkt.dat"))
            self.connect((self._metrics,1), gr.file_sink(gr.sizeof_char,"metrics_new_pkt_b.dat"))
            self.connect((self._viterbi,1), gr.file_sink(gr.sizeof_char, "viterbi_new_pkt_b.dat"))
            self.connect((self._chunk2byte,0),gr.file_sink(gr.sizeof_char,"decoded_payload_crc32_b.dat"))
            self.connect((self._chunk2byte,1), gr.file_sink(gr.sizeof_char, "decoder_new_pkt_b.dat"))
        
        self._watcher = _queue_watcher_thread(self._rcv_queue, callback) 
        
    def set_pkt_size(self,pkt_size):
          self._crc32_checker.set_pkt_size(pkt_size) 
          self._viterbi.set_K(((pkt_size+4)*8)/self._bitspersymbol)
    
class _queue_watcher_thread(_threading.Thread):
    def __init__(self, rcvd_pktq, callback):
        _threading.Thread.__init__(self)
        self.setDaemon(1)
        self.rcvd_pktq = rcvd_pktq
        self.callback = callback
        self.keep_running = True
        self.start()


    def run(self):
        while self.keep_running:
            msg = self.rcvd_pktq.delete_head()
            ok = msg.arg1()
            payload = msg.to_string()
            if self.callback:
                self.callback(ok, payload)
           
        
    
                            
