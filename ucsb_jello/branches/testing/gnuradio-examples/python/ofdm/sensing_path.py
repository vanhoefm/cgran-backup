#!/usr/bin/env python
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

from gnuradio import gr, gru, blks2, window
from gnuradio import usrp
from gnuradio import eng_notation
import copy
import math
# from current dir
import numpy, struct


# linklab, define constants
FFT_SIZE      = 512        # fft size for sensing  
LOW_THRES     = -45        # low power threshold in dB to identify free freq blocks
HIGH_THRES    = -25        # high power threshold in dB to identify busy freq blocks
SMOOTH_LENGTH = 100        # smooth length
EDGE_THRES    = 5          # edge detection threshold in dB


# /////////////////////////////////////////////////////////////////////////////
#                              sensing path
# /////////////////////////////////////////////////////////////////////////////

class sensing_path(gr.hier_block2):
    def __init__(self, options):

	gr.hier_block2.__init__(self, "sensing_path",
				gr.io_signature(1, 1, gr.sizeof_gr_complex), # Input signature
				gr.io_signature(0, 0, 0)) # Output signature


        options = copy.copy(options)    # make a copy so we can destructively modify

        self._verbose        = options.verbose
       
        # linklab, fft size for sensing, different from fft length for tx/rx
        self.fft_size = FFT_SIZE

        # interpolation rate: sensing fft size / ofdm fft size
        self.interp_rate = self.fft_size/options.fft_length

        self._fft_length      = options.fft_length
        self._occupied_tones  = options.occupied_tones
        self.msgq             = gr.msg_queue()

        # linklab , setup the sensing path
        # FIXME: some components are not necessary
        self.s2p = gr.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        mywindow = window.blackmanharris(self.fft_size)
        self.fft = gr.fft_vcc(self.fft_size, True, mywindow)
        power = 0
        for tap in mywindow:
            power += tap*tap
        self.c2mag = gr.complex_to_mag(self.fft_size)
        self.avg = gr.single_pole_iir_filter_ff(1.0, self.fft_size)

        # linklab, ref scale value from default ref_scale in usrp_fft.py
        ref_scale = 13490.0

        # FIXME  We need to add 3dB to all bins but the DC bin
        self.log = gr.nlog10_ff(20, self.fft_size,
                                -10*math.log10(self.fft_size)              # Adjust for number of bins
                                -10*math.log10(power/self.fft_size)        # Adjust for windowing loss
                                -20*math.log10(ref_scale/2))               # Adjust for reference scale

        self.sink = gr.message_sink(gr.sizeof_float * self.fft_size, self.msgq, True)
        self.connect(self, self.s2p, self.fft, self.c2mag, self.avg, self.log, self.sink)

    # linklab, get PSD map to perform smoothing
    def get_psd(self):
        LOOP = 50
        self.msgq.flush()
        
        # linklab, loop to empty the lower layer buffers to avoid detecting old signals
        while self.msgq.count() < 60:
            pass
        self.msgq.flush()

        # linklab, loop until received enough number of observations for smoothing
        nitems = 0
        s = ""
        while nitems < SMOOTH_LENGTH:
            msg = self.msgq.delete_head()  # blocking read of message queue
            nitems += int(msg.arg2())
            s += msg.to_string()

        # linklab, smooth psd by averaging over multiple observations
        itemsize = int(msg.arg1())
        psd_temp = numpy.zeros(self.fft_size)
        for i in range(0, SMOOTH_LENGTH):
            start = itemsize * (SMOOTH_LENGTH - i - 1)
            tmp = s[start:start+itemsize]
            psd_temp += numpy.fromstring(tmp, numpy.float32)/SMOOTH_LENGTH

        # linklab, rearrange psd bins
        psd = psd_temp.copy()
        psd[self.fft_size/2:self.fft_size] = psd_temp[0:self.fft_size/2]
        psd[0:self.fft_size/2] = psd_temp[self.fft_size/2:self.fft_size]

        return psd 

    # linklab, get available subcarriers via edge detection
    def get_avail_carriers(self):
        
        # get PSD map
        psd = self.get_psd()
       
        N = self.fft_size

        # calculate the number of bins on the left that are not allocatable
        zeros_on_left = int(math.ceil((self._fft_length - self._occupied_tones)/2.0)) * self.interp_rate
        
        # init avail_subc_bin to indicate all subcarriers are available
        avail_subc_bin = numpy.ones(self._occupied_tones)
       
        # freq domain smoothing
        for i in range(1, N-1):
            psd[i] = (psd[i-1] + psd[i] + psd[i+1]) / 3


        # identify and mark the rising and dropping edges
        # edge_markers: 1 rising, -1 dropping, 0 no edge
        edge_markers = numpy.zeros(N)
        for i in range(2,N-3):
            diff_forward = psd[i+2] - psd[i];
            diff_backward = psd[i] - psd[i-2];
            diff = psd[i+1] - psd[i-1]
            
            if diff_forward > EDGE_THRES:    # check rising edges
                edge_markers[i] = 1
            elif diff_backward < -EDGE_THRES:# check dropping edges
                edge_markers[i] = -1
    
        # use edge information to mark unavailable subcarriers
        # we treat the left half and right half separately to avoid the central freq artifact
        avail = numpy.zeros(N)

        # the right half subcarriers
        avail[0] = 1
        for i in range(1,N/2):
            if ((avail[i-1] == 1) and (edge_markers[i] == 1)):
                avail[i] = 0
            elif ((avail[i-1] == 0) and (edge_markers[i] == -1)) and edge_markers[i+1] != -1:
                avail[i] = 1
            else:
                avail[i] = avail[i-1]

        # the left half subcarriers
        avail[N-1] = 1
        for j in range(1,N/2):
            i = N - j - 1
            if ((avail[i+1] == 1) and (edge_markers[i] == -1)): 
                avail[i] = 0
            elif ((avail[i+1] == 0) and (edge_markers[i] == 1)) and edge_markers[i-1] != 1:
                    avail[i] = 1
            else:
                avail[i] = avail[i+1]

        # combine edge detection sensing results with energy sensing with HIGH_THRES and LOW_THRES
        for i in range(zeros_on_left, self.fft_size - zeros_on_left - 1):
            
            # map the PSD index i to subcarrier index
            carrier_index = (i - zeros_on_left) / self.interp_rate
            
            # if power very high or detected busy from edge detection, unavailable
            if psd[i] > HIGH_THRES  or avail[i] == 0:
                avail_subc_bin[carrier_index] = 0
            # if power very low, available 
            if psd[i] < LOW_THRES: 
                avail_subc_bin[carrier_index] = 1
       
        return avail_subc_bin

