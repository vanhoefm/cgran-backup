#!/usr/bin/env python
# this is to test the gr-dabp package for FIC
# ./test_dabp_fic_usrp -f carrier_freq -g gain -n channel.conf

from gnuradio import gr
import dabp
from gnuradio import usrp
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import sys

class test_dabp(gr.top_block):
    """
    @brief Test program for the dab package
    """
    def __init__(self):
        gr.top_block.__init__(self)

        parser = OptionParser(option_class=eng_option)
        parser.add_option("-m", "--dab-mode", type="int", default=1,
            help="DAB mode [default=%default]")
        parser.add_option("-f", "--freq", type="eng_float", default=206352000,
            help="Carrier frequency [default=%default]")
        parser.add_option("-g", "--gain", type="eng_float", default=8,
            help="USRP gain (dB) [default=%default]")
        parser.add_option("-n", "--config", type="string", default="channel.conf",
            help="DAB+ channel configuration file [default=%default]")
        
        (options, args) = parser.parse_args ()
        
        if len(args)>0:
            parser.print_help()
            raise SystemExit, 1
        
        self.conf=options.config
        
        ######## set up usrp
        self.u = usrp.source_c(decim_rate=32) # 2M sampling freq
        rx_subdev_spec = usrp.pick_rx_subdevice(self.u)
        #print "u.db(0,0).dbid() = ", self.u.db(0,0).dbid()
        #print "u.db(1,0).dbid() = ", self.u.db(1,0).dbid()
        #print "rx_subdev_spec = ", options.rx_subdev_spec
        
        mux=usrp.determine_rx_mux_value(self.u, rx_subdev_spec)
        #print "mux = ", mux
        self.u.set_mux(mux)

        # determine the daughterboard subdevice we're using
        self.subdev = usrp.selected_subdev(self.u, rx_subdev_spec)
        #print "Using RX d'board %s" % (self.subdev.side_and_name(),)
        input_rate = self.u.adc_freq() / self.u.decim_rate()
        #print "ADC sampling @ ", eng_notation.num_to_str(self.u.adc_freq())
        #print "USB sample rate %s" % (eng_notation.num_to_str(input_rate))

        self.subdev.set_gain(options.gain)
        r = self.u.tune(0, self.subdev, options.freq)
        
        #print "freq range = [%s, %s]" % (eng_notation.num_to_str(self.subdev.freq_min()), eng_notation.num_to_str(self.subdev.freq_max()))
        if not r:
            sys.stderr.write('Failed to set frequency\n')
            raise SystemExit, 1
        
        
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
        self.connect(self.u, nulldet)
        self.connect(self.u, (demod,0))
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
        
        

