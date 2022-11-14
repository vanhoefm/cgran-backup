#/usr/bin/env python
#
# Copyright 2006, 2007, 2008 Free Software Foundation, Inc.
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

import math
from numpy import fft
from gnuradio import gr
from gnuradio.blks2impl.ofdm_sync_ml import ofdm_sync_ml
from gnuradio.blks2impl.ofdm_sync_pn import ofdm_sync_pn
from gnuradio.blks2impl.ofdm_sync_pnac import ofdm_sync_pnac
from gnuradio.blks2impl.ofdm_sync_fixed import ofdm_sync_fixed

# linklab, import non-contiguous filetr
from gnuradio.ncofdm_filter import ncofdm_filt

class ofdm_receiver(gr.hier_block2):
    """
    Performs receiver synchronization on OFDM symbols.

    The receiver performs channel filtering as well as symbol, frequency, and phase synchronization.
    The synchronization routines are available in three flavors: preamble correlator (Schmidl and Cox),
    modifid preamble correlator with autocorrelation (not yet working), and cyclic prefix correlator
    (Van de Beeks).
    """
    # linklab, add carrier_map_bin to indicate carrier map, nc_filter to indicate use non-contiguous or not
    def __init__(self, fft_length, cp_length, occupied_tones, snr, ks, carrier_map_bin, nc_filter, logging=False):
        """
	Hierarchical block for receiving OFDM symbols.

	The input is the complex modulated signal at baseband.
        Synchronized packets are sent back to the demodulator.

        @param fft_length: total number of subcarriers
        @type  fft_length: int
        @param cp_length: length of cyclic prefix as specified in subcarriers (<= fft_length)
        @type  cp_length: int
        @param occupied_tones: number of subcarriers used for data
        @type  occupied_tones: int
        @param snr: estimated signal to noise ratio used to guide cyclic prefix synchronizer
        @type  snr: float
        @param ks: known symbols used as preambles to each packet
        @type  ks: list of lists
        @param logging: turn file logging on or off
        @type  logging: bool
	"""

	gr.hier_block2.__init__(self, "ofdm_receiver",
				gr.io_signature(1, 1, gr.sizeof_gr_complex), # Input signature
                                gr.io_signature2(2, 2, gr.sizeof_gr_complex*occupied_tones, gr.sizeof_char)) # Output signature

        # linklab, get ofdm parameters
        self._fft_length = fft_length
        self._occupied_tones = occupied_tones
        self._cp_length = cp_length
        self._nc_filter = nc_filter
        self._carrier_map_bin = carrier_map_bin

        win = [1 for i in range(self._fft_length)]

        # linklab, initialization function
        self.initialize(ks, self._carrier_map_bin)
        
        SYNC = "pn"
        if SYNC == "ml":
            nco_sensitivity = -1.0/fft_length                             # correct for fine frequency
            self.ofdm_sync = ofdm_sync_ml(fft_length, cp_length, snr, self._ks0time, logging)
        elif SYNC == "pn":
            nco_sensitivity = -2.0/fft_length                             # correct for fine frequency
            self.ofdm_sync = ofdm_sync_pn(fft_length, cp_length, self._subcarriers_num, logging)
        elif SYNC == "pnac":
            nco_sensitivity = -2.0/fft_length                             # correct for fine frequency
            self.ofdm_sync = ofdm_sync_pnac(fft_length, cp_length, self._ks0time, logging)
        elif SYNC == "fixed":                                             # for testing only; do not user over the air
            self.chan_filt = gr.multiply_const_cc(1.0)                    # remove filter and filter delay for this
            nsymbols = 18                                                 # enter the number of symbols per packet
            freq_offset = 0.0                                             # if you use a frequency offset, enter it here
            nco_sensitivity = -2.0/fft_length                             # correct for fine frequency
            self.ofdm_sync = ofdm_sync_fixed(fft_length, cp_length, nsymbols, freq_offset, logging)

        # channel filter
        bw = (float(occupied_tones) / float(fft_length)) / 2.0
        tb = bw*0.06 

        chan_coeffs = gr.firdes.low_pass (1.0,                     # gain
                                          1.0,                     # sampling rate
                                          bw+tb,                   # midpoint of trans. band
                                          tb,                      # width of trans. band
                                          gr.firdes.WIN_HAMMING)   # filter type
        self.chan_filt = gr.fft_filter_ccc(1, chan_coeffs)
       
        # Create a delay line, linklab
        self.delay = gr.delay(gr.sizeof_gr_complex, fft_length)

        self.nco = gr.frequency_modulator_fc(nco_sensitivity)         # generate a signal proportional to frequency error of sync block
        self.sigmix = gr.multiply_cc()
        self.sampler = gr.ofdm_sampler(fft_length, fft_length+cp_length)
        self.fft_demod = gr.fft_vcc(fft_length, True, win, True)
        self.ofdm_frame_acq = gr.ofdm_frame_acquisition(self._occupied_tones, self._fft_length,
                                                        self._cp_length, self._ks[0])

        # linklab, check current mode: non-contiguous OFDM or not
        if self._nc_filter:
            print '\nMulti-band Filter Turned ON!'
            # linklab, non-contiguous filter
            self.ncofdm_filt = ncofdm_filt(self._fft_length, self._occupied_tones, self._carrier_map_bin)
            self.connect(self, self.chan_filt, self.ncofdm_filt) 
            self.connect(self.ncofdm_filt, self.ofdm_sync)             # into the synchronization alg.
            self.connect((self.ofdm_sync,0), self.nco, (self.sigmix,1))   # use sync freq. offset output to derotate input signal
            self.connect(self.ncofdm_filt, self.delay, (self.sigmix,0))                 # signal to be derotated
        else :
            print '\nMulti-band Filter Turned OFF!'
            self.connect(self, self.chan_filt)
            self.connect(self.chan_filt, self.ofdm_sync)             # into the synchronization alg.
            self.connect((self.ofdm_sync,0), self.nco, (self.sigmix,1))   # use sync freq. offset output to derotate input signal
            self.connect(self.chan_filt, self.delay, (self.sigmix,0))                 # signal to be derotated

        self.connect(self.sigmix, (self.sampler,0))                   # sample off timing signal detected in sync alg
        self.connect((self.ofdm_sync,1), (self.sampler,1))            # timing signal to sample at

        self.connect((self.sampler,0), self.fft_demod)                # send derotated sampled signal to FFT
        self.connect(self.fft_demod, (self.ofdm_frame_acq,0))         # find frame start and equalize signal
        self.connect((self.sampler,1), (self.ofdm_frame_acq,1))       # send timing signal to signal frame start
        self.connect((self.ofdm_frame_acq,0), (self,0))               # finished with fine/coarse freq correction,
        self.connect((self.ofdm_frame_acq,1), (self,1))               # frame and symbol timing, and equalization

        if logging:
            self.connect(self.chan_filt, gr.file_sink(gr.sizeof_gr_complex, "ofdm_receiver-chan_filt_c.dat"))
            self.connect(self.fft_demod, gr.file_sink(gr.sizeof_gr_complex*fft_length, "ofdm_receiver-fft_out_c.dat"))
            self.connect(self.ofdm_frame_acq,
                         gr.file_sink(gr.sizeof_gr_complex*occupied_tones, "ofdm_receiver-frame_acq_c.dat"))
            self.connect((self.ofdm_frame_acq,1), gr.file_sink(1, "ofdm_receiver-found_corr_b.dat"))
            self.connect(self.sampler, gr.file_sink(gr.sizeof_gr_complex*fft_length, "ofdm_receiver-sampler_c.dat"))
            self.connect(self.sigmix, gr.file_sink(gr.sizeof_gr_complex, "ofdm_receiver-sigmix_c.dat"))
            self.connect(self.nco, gr.file_sink(gr.sizeof_gr_complex, "ofdm_receiver-nco_c.dat"))

    # linklab, initialization
    def initialize(self, ks, carrier_map_bin):
        self._ks = ks
        self._zeros_on_left = int(math.ceil((self._fft_length - self._occupied_tones)/2.0))
        self._ks0 = self._fft_length*[0,]
        self._ks0[self._zeros_on_left : self._zeros_on_left + self._occupied_tones] = self._ks[0]
        
        self._ks0 = fft.ifftshift(self._ks0)
        self._ks0time = fft.ifft(self._ks0)

        # ADD SCALING FACTOR
        self._ks0time = self._ks0time.tolist()

        # linklab, calc # of used subcarriers
        if self._nc_filter:
            self._subcarriers_num = sum(carrier_map_bin) 
        else :
            self._subcarriers_num = self._occupied_tones

    # linklab, reset carrier map
    def reset_carrier_map(self, new_ks, new_carrier_map_bin):
        self.initialize(new_ks, new_carrier_map_bin)
        self.ofdm_frame_acq.reset_known_symbol(self._ks[0])
        if self._nc_filter:
            self.ncofdm_filt.reset_carrier_map(new_carrier_map_bin)
            self.ofdm_sync.reset_carrier_map(self._subcarriers_num)

    # linklab, filter frequency offset compensation
    def filter_fo_comp(self, freq_offset):
        if self._nc_filter:
            self.ncofdm_filt.freq_offset_comp(freq_offset) 
  
    # linklab, get frequency offset estimation
    def get_freq_offset(self):
        int_freq =  self.ofdm_frame_acq.d_coarse_freq
        frac_freq = self.ofdm_sync.get_frac_fo()
        return int_freq, frac_freq

    # linklab, get SINR estimation
    def get_sinr(self):
        freq_sinr =  self.ofdm_frame_acq.d_sinr
        time_sinr = self.ofdm_sync.get_sinr()
        return time_sinr, freq_sinr
