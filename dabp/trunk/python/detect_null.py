#!/usr/bin/env python
#
# Copyright 2005,2006,2007 Free Software Foundation, Inc.
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
# This code fragment was adopted from the gr-dab package written by Andreas Muller

from gnuradio import gr
import dabp
class detect_null(gr.hier_block2):
    """
    @brief Detect Null symbols by looking at the energy of the symbol

    This block outputs a 1 at the last sample of the null symbol where a new DAB frame follows, and zeros otherwise.
    """
    
    def __init__(self, length, debug=False):
        """
        Hierarchical block to detect Null symbols

        @param length length of the Null symbol (in samples)
        @param debug whether to write signals out to files
        """
        gr.hier_block2.__init__(self, "detect_null",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex),    # input signature
                                gr.io_signature(1, 1, gr.sizeof_char))          # output signature


        # get the magnitude squared
        self.ns_c2magsquared = gr.complex_to_mag_squared()
        self.ns_moving_sum = dabp.moving_sum_ff(length)
        self.ns_invert = gr.multiply_const_ff(-1)

        # peak detector on the inverted, summed up signal -> we get the nulls (i.e. the position of the start of a frame)
        self.ns_peak_detect = gr.peak_detector_fb(0.6,0.7,10,0.0001) # mostly found by try and error -> remember that the values are negative!

        # connect it all
        self.connect(self, self.ns_c2magsquared, self.ns_moving_sum, self.ns_invert, self.ns_peak_detect, self)

        if debug:
            self.connect(self.ns_invert, gr.file_sink(gr.sizeof_float, "debug/ofdm_sync_dabp_ns_filter_inv_f.dat"))
            self.connect(self.ns_peak_detect,gr.file_sink(gr.sizeof_char, "debug/ofdm_sync_dabp_peak_detect_b.dat"))

