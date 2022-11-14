#
# Copyright 2011 FOI
#
# Copyright 2005,2006 Free Software Foundation, Inc.
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

# This is a modification of transmit_path.py from GNU Radio.

from gnuradio import gr, gru, blks2,trellis
from gnuradio import eng_notation

import copy
import sys
import math

from foimimo.ofdm import ofdm_mod
from foimimo.ofdm_with_coding import ofdm_mod_with_coding
from foimimo.coding import encoding
# /////////////////////////////////////////////////////////////////////////////
#                              transmit path
# /////////////////////////////////////////////////////////////////////////////

class siso_transmit_path(gr.hier_block2): 
    def __init__(self, options):
        '''
        See below for what options should hold
        '''

	gr.hier_block2.__init__(self, "siso_transmit_path",
				gr.io_signature(0, 0, 0), # Input signature
				gr.io_signature(1, 1, gr.sizeof_gr_complex)) # Output signature

        options = copy.copy(options)    # make a copy so we can destructively modify

        self._verbose      = options.verbose         # turn verbose mode on/off
        self._tx_amplitude = options.tx_amplitude    # digital amplitude sent to USRP
        self._code_rate    = options.code_rate
        
        self.amp = gr.multiply_const_cc(1)
        self.set_tx_amplitude(self._tx_amplitude)
       
        if self._code_rate != "":
            # [ref] Proakis pp. 493-496
            # Rate  Generator in octal
            #  3/4   13  25  61 47
            #  2/3  236 155 337
            #  1/3   13  15  17
            code = {"1/1":[2, 2,[1, 0, 0, 1]],
                    "3/4":[3,4,[0, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 1]],
                    "2/3":[2, 3, [11, 6, 11, 6, 11, 15]], 
                    "1/3":[1, 3, [11, 13, 15]]}
            selected_code = code[self._code_rate]
    
            self._msgq_limit = 2
            self._encode_fsm = trellis.fsm(selected_code[0],selected_code[1],selected_code[2])
            
            self.encoder = encoding(options,self._encode_fsm, self._msgq_limit)
            
            self.ofdm_tx = ofdm_mod_with_coding(options, msgq_limit=4, pad_for_usrp=False)
            # Create and setup transmit path flow graph
            self.connect((self.encoder,0), (self.ofdm_tx,0))
            self.connect((self.encoder,1), (self.ofdm_tx,1))
            self.connect((self.ofdm_tx,0), (self.amp,0), (self,0))
            
        else:
    
            self.ofdm_tx = ofdm_mod(options, msgq_limit=4, pad_for_usrp=False)
            # Create and setup transmit path flow graph
            self.connect(self.ofdm_tx, self.amp, self)



        # Display some information about the setup
        if self._verbose:
            self._print_verbage()
       

    def set_tx_amplitude(self, ampl):
        """
        Sets the transmit amplitude sent to the USRP
        @param: ampl 0 <= ampl < 32768.  Try 8000
        """
        self._tx_amplitude = max(0.0, min(ampl, 32767.0))
        self.amp.set_k(self._tx_amplitude)
        
    def send_pkt(self, payload='', eof=False):
        """
        Calls the transmitter method to send a packet
        """        
        if self._code_rate != "":
            return self.encoder.send_pkt(payload, eof)
        else:
            return self.ofdm_tx.send_pkt(payload, eof)
        
    def add_options(normal, expert):
        """
        Adds transmitter-specific options to the Options Parser
        """
        normal.add_option("-c", "--code-rate", type="string", default="",
                          help="set code rate (3/4, 2/3 or 1/3), empty for uncoded")
        normal.add_option("-a", "--tx-amplitude", type="eng_float", default=1, metavar="AMPL",
                          help="scale digital tx amplitude [default=%default]")
        normal.add_option("-v", "--verbose", action="store_true", default=False)
        expert.add_option("", "--log", action="store_true", default=False,
                          help="Log all parts of flow graph to file (LOTS of data)")

    # Make a static method to call before instantiation
    add_options = staticmethod(add_options)

    def _print_verbage(self):
        """
        Prints information about the transmit path
        """
        print "Tx amplitude     %s"    % (self._tx_amplitude)
        
