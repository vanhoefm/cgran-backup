#!/usr/bin/env python
# this is to test the gr-dabp package

from gnuradio import gr
import dabp
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import sys

class test_dabp(gr.top_block):
    """
    @brief Test program for the dab package
    """
    def __init__(self):
        gr.top_block.__init__(self)

        usage = "%prog: [options] samples_file"
        parser = OptionParser(option_class=eng_option, usage=usage)
        parser.add_option("-m", "--dab-mode", type="int", default=1,
            help="DAB mode [default=%default]")
        parser.add_option("-c", "--complex", action="store_true", default=False,
            help="Sample file is float complex (64 bit per sample, 32 I + 32 Q) format")
        parser.add_option("-n", "--config", type="string", default="channel.conf",
            help="DAB+ channel configuration file [default=%default]")
        
        (options, args) = parser.parse_args ()
        
        if len(args)<1:
            print "error: need file with samples"
            sys.exit(1)
        else:
            filename = args[0]
            if options.complex:
                src = gr.file_source(gr.sizeof_gr_complex, filename, True)
            else:
                sinput = gr.file_source(gr.sizeof_short, filename, True)
                src = gr.interleaved_short_to_complex()
                self.connect(sinput, src)

        self.conf = options.config
        
        mode=options.dab_mode
        param = dabp.parameters(mode)
        # null symbol detector
        samples_per_null_sym=param.get_Nnull()
        #print "samples per null symbol : %d" % samples_per_null_sym
        nulldet = dabp.detect_null(samples_per_null_sym)
        # ofdm demod
        demod = dabp.ofdm_demod(mode)
        # FIC/MSC demultiplexer
        demux = dabp.fic_msc_demux(0, mode)
        # depuncturer
        punc = dabp.depuncturer_fic(mode)
        I=punc.getI()
        # viterbi
        vit = dabp.vitdec(I)
        # descrambler
        scram = dabp.scrambler(I)
        # FIB sink
        self.dst = dabp.fib_sink()
        
        nullsink = gr.null_sink(gr.sizeof_char)
        # connect everything
        self.connect(src, nulldet)
        self.connect(src, (demod,0))
        self.connect(nulldet, (demod,1))
        self.connect((demod,0), demux, punc, vit, scram, self.dst)
        self.connect((demod,1), nullsink)

        # debug
        #self.connect(nulldet, gr.file_sink(gr.sizeof_char, "debug_nulldet.dat"))
        #self.connect(vit, gr.file_sink(gr.sizeof_char, "debug_vit.dat"))
        
if __name__=='__main__':
    try:
        tb=test_dabp()
        tb.run()
        tb.dst.print_subch()
        tb.dst.save_subch(tb.conf)
    except KeyboardInterrupt:
        pass

