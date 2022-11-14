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

# This is a modification of receive_path.py from GNU Radio.

from gnuradio import gr, gru, blks2,trellis
from gnuradio import eng_notation
import copy
import sys
import math

from foimimo.ofdm import ofdm_demod
from foimimo.ofdm_with_coding import ofdm_demod_with_coding
from foimimo.coding import decoding

# /////////////////////////////////////////////////////////////////////////////
#                              receive path
# /////////////////////////////////////////////////////////////////////////////

class siso_receive_path(gr.hier_block2):
    def __init__(self, rx_callback, bad_header_callback, options):

	gr.hier_block2.__init__(self, "siso_receive_path",
				gr.io_signature(1, 1, gr.sizeof_gr_complex), # Input signature
				gr.io_signature(0, 0, 0)) # Output signature


        options = copy.copy(options)    # make a copy so we can destructively modify

        self._verbose     = options.verbose
        self._log         = options.log
        self._rx_callback = rx_callback      # this callback is fired when there's a packet available
        self._bad_header_callback = bad_header_callback # this callback is fired when a packet with bad header is thrown away
        
        self._code_rate    = options.code_rate
        
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
            # receiver
            self._decode_fsm = trellis.fsm(selected_code[0],selected_code[1],selected_code[2])                
            self.ofdm_siso_demod = ofdm_demod_with_coding(options,
                                                 int(round(math.log(self._decode_fsm.I())/math.log(2))),
                                                 int(round(math.log(self._decode_fsm.O())/math.log(2))), 
                                                 bad_header_callback=self._bad_header_callback)
            self.decoder = decoding(options, self._decode_fsm, callback=self._rx_callback)
       
            self.connect((self,0), (self.ofdm_siso_demod,0))
            self.connect((self.ofdm_siso_demod,0), (self.decoder,0))
            self.connect((self.ofdm_siso_demod,1), (self.decoder,1))
        else:
            self.ofdm_siso_demod = ofdm_demod(options, callback=self._rx_callback)

            # Carrier Sensing Blocks
            alpha = 0.001
            thresh = 30   # in dB, will have to adjust
            self.probe = gr.probe_avg_mag_sqrd_c(thresh,alpha)
    
            self.connect(self, self.ofdm_siso_demod)
            self.connect(self.ofdm_siso_demod, self.probe)

        # Display some information about the setup
        if self._verbose:
            self._print_verbage()
        
    def carrier_sensed(self):
        """
        Return True if we think carrier is present.
        """
        #return self.probe.level() > X
        return self.probe.unmuted()

    def carrier_threshold(self):
        """
        Return current setting in dB.
        """
        return self.probe.threshold()

    def set_carrier_threshold(self, threshold_in_db):
        """
        Set carrier threshold.

        @param threshold_in_db: set detection threshold
        @type threshold_in_db:  float (dB)
        """
        self.probe.set_threshold(threshold_in_db)
    
        
    def add_options(normal, expert):
        """
        Adds receiver-specific options to the Options Parser
        """
        normal.add_option("-c", "--code-rate", type="string", default="",
                          help="set code rate (3/4, 2/3 or 1/3), empty for uncoded [default=%default]")
        normal.add_option("-v", "--verbose", action="store_true", default=False)
        expert.add_option("", "--log", action="store_true", default=False,
                          help="Log all parts of flow graph to files (CAUTION: lots of data)")

    # Make a static method to call before instantiation
    add_options = staticmethod(add_options)


    def _print_verbage(self):
        """
        Prints information about the receive path
        """
        pass
