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
# spectral density.


from gnuradio import gr,specest
from pylab import *

from gnuradio.eng_option import eng_option
from optparse import OptionParser


def parse_options():
        usage = "%prog: [options] filename"
        parser = OptionParser(option_class=eng_option, usage=usage)
        parser.add_option("-s", "--sample-type", type="choice", choices=("float", "complex"),
                help="For .dat-files: set input type to float or complex.", default="complex")
        parser.add_option("-F", "--fft-len", type="int", default=1024,
                help="FFT length")
        parser.add_option("-m", "--ma-length", type="int", default=8,
                help="length of moving average")
        parser.add_option("-o", "--overlap", type="int", default=-1,
                help="number of overlapping samples per segment, leave empty for 50% overlap.")
        parser.add_option("-N", "--samples", type="eng_float", default=None,
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
            if options.overlap == -1:
                options.samples = (options.fft_len / 2 + 1) * options.ma_length
            else:
                options.samples = options.fft_len * options.ma_length - options.overlap * (options.ma_length - 1)

        if options.sample_type == "complex":
            options.sample_size = gr.sizeof_gr_complex
        else:
            options.sample_size = gr.sizeof_float

        return (options, args[0])


def main():
    (options, filename) = parse_options()

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

    welch = specest.welch(options.fft_len, options.overlap, options.ma_length, options.shift_fft)
    sink = gr.vector_sink_f(options.fft_len)

    if options.linear:
        tb.connect(head, welch, sink)
        ylabelstr = 'Power Spectrum Density / W/rad'
    else:
        l2dB = gr.nlog10_ff(10, options.fft_len)
        tb.connect(head, welch, l2dB, sink)
        ylabelstr = 'Power Spectrum Density / dBW/rad'

    tb.run()

    psd_est = sink.data()[-options.fft_len:]

    if options.shift_fft:
        x_axis = (array(range(options.fft_len)) - round(options.fft_len/2)) / round(options.fft_len/2)
    else:
        x_axis = (array(range(options.fft_len))) / round(options.fft_len/2)

    figure()
    plot(x_axis, psd_est)
    xlabel('Normalised frequency / pi')
    ylabel(ylabelstr)
    grid()
    show()


if __name__ == '__main__':
    main()

