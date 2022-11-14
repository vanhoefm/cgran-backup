#!/usr/bin/env python
#
# Copyright 2009 Institut fuer Nachrichtentechnik / Uni Karlsruhe
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

# Read a file (WAV or DAT) and display a part of the content's power
# spectral density, before and after compression.


from gnuradio import gr,specest,cs
from pylab import *
from numpy import *

from gnuradio.eng_option import eng_option
from optparse import OptionParser

import struct

class compressor(gr.hier_block2):
    """ Stream-based compression. This block adds type conversions to the
    compression block, which usually works with vectors."""
    def __init__(self, cs_sequence, block_out_len, is_binary):
        gr.hier_block2.__init__(self, "compressor",
                gr.io_signature(1, 1, gr.sizeof_gr_complex),
                gr.io_signature(1, 1, gr.sizeof_gr_complex))

        self.stream2vec = gr.stream_to_vector(gr.sizeof_gr_complex, size(cs_sequence))
        self.compress = cs.circmat_vccb(cs_sequence, block_out_len, is_binary)
        self.vec2stream = gr.vector_to_stream(gr.sizeof_gr_complex, block_out_len)
        self.normalize = gr.multiply_const_cc(1.0 / sqrt(size(cs_sequence)))
        self.connect(self, self.stream2vec, self.compress, self.vec2stream, self.normalize,  self)


def parse_options():
        usage = "%prog: [options] filename"
        parser = OptionParser(option_class=eng_option, usage=usage)
        parser.add_option("-L", "--block-length", type="int", default=1024,
                help="Length of compression block length (column count in compression matrix)")
        parser.add_option("-f", "--sequence-file", type="string", default=None,
                help="File with compression sequence.")
        parser.add_option("-C", "--compressed-block-length", type="int", default=128,
                help="Length of compressed block (row count in compression matrix), equals FFT length.")
        parser.add_option("-s", "--sample-type", type="choice", choices=("float", "complex"),
                help="For .dat-files: set input type to float or complex.", default="complex")
        parser.add_option("-m", "--ma-length", type="int", default=8,
                help="length of moving average")
        parser.add_option("-o", "--overlap", type="int", default=0,
                help="number of overlapping samples per segment, leave empty for no overlap.")
        parser.add_option("-N", "--samples", type="intx", default=None,
                help="maximum number of samples to grab from file (you can usually leave this out)")
        parser.add_option("-S", "--shift-fft", action="store_true", default=False,
                help="shift the DC carrier to the middle.")
        parser.add_option("-l", "--linear", action="store_true", default=False,
                help="Plot output on a linear scale instead of a dB scale.")

        (options, args) = parser.parse_args ()
        if len(args) != 1:
            parser.print_help()
            raise SystemExit, 1

        if options.samples is None:
            options.samples = options.compressed_block_length * options.ma_length

        print options.samples
        if options.sample_type == "complex":
            options.sample_size = gr.sizeof_gr_complex
        else:
            options.sample_size = gr.sizeof_float

        return (options, args[0])

def get_cs_sequence(block_len, filename):
    """ Creates a compression sequence, either a random 0/1 sequence, or,
    if a filename is given, a sequence read from a file.
    The second return value is True if zeros should be interpreted as -1.
    For files, this value is autodetected."""
    if filename is None:
        return (random.randint(0, 2, block_len), True)
    else:
        fid = open(filename, mode='rb')

        str_data = fid.read()
        n_bytes = len(str_data)
        sequence = struct.unpack('b' * n_bytes, str_data)

        if str_data.count(chr(255)) > 0:
            is_binary = False
        else:
            is_binary = True

        fid.close()
        return (sequence, is_binary)


def calc_xaxis_vector(fft_len, shift_fft):
    """ Return a vector which can be used as an x-value for plot()"""
    if options.shift_fft:
        return (array(range(fft_len)) - round(fft_len/2)) / round(fft_len/2)
    else:
        return (array(range(fft_len))) / round(fft_len/2)

def main():
    (options, filename) = parse_options()

    fft_len = options.compressed_block_length
    tb = gr.top_block()
    head = gr.head(gr.sizeof_gr_complex, options.samples)

    if filename[-4:].lower() == '.wav':
        src = gr.wavfile_source(filename, False)
        f2c = gr.float_to_complex()
        tb.connect((src, 0), (f2c, 0))
        if src.channels() == 2:
            tb.connect((src, 1), (f2c, 1))
        tb.connect(f2c, head)
    else:
        src = gr.file_source(options.sample_size, filename, False)
        if options.sample_size == gr.sizeof_float:
            f2c = gr.float_to_complex()
            tb.connect(src, f2c, head)
        else:
            tb.connect(src, head)

    (cs_sequence, is_binary) = get_cs_sequence(options.block_length, options.sequence_file)
    block_len = len(cs_sequence)
    CS = compressor(cs_sequence, fft_len, is_binary)
    tb.connect(head, CS)

    welch = specest.welch(fft_len, options.overlap, options.ma_length, options.shift_fft)
    sink = gr.vector_sink_f(fft_len)
    welch_cs = specest.welch(fft_len, 0, options.ma_length, options.shift_fft)
    sink_cs = gr.vector_sink_f(fft_len)

    if options.linear:
        tb.connect(head, welch, sink)
        tb.connect(CS, welch_cs, sink_cs)
        ylabelstr = 'Power Spectrum Density / W/rad'
    else:
        l2dB = gr.nlog10_ff(10, fft_len)
        l2dB_cs = gr.nlog10_ff(10, fft_len)
        tb.connect(head, welch, l2dB, sink)
        tb.connect(CS, welch_cs, l2dB_cs, sink_cs)
        ylabelstr = 'Power Spectrum Density / dBW/rad'

    tb.run()

    psd_est = sink.data()[-fft_len:]
    psd_est_cs = sink_cs.data()[-fft_len:]

    x_axis = calc_xaxis_vector(fft_len, options.shift_fft)

    figure()
    plot(x_axis, psd_est)
    hold(True)
    plot(x_axis, psd_est_cs)
    xlabel('Normalised frequency / pi')
    ylabel(ylabelstr)
    legend(('Uncompressed', 'Compressed to %.2f%%' % (100.0 * fft_len / block_len)))
    grid()
    show()


if __name__ == '__main__':
    main()

