from gnuradio import gr
import dabp
import dabp_usrp

class dabp_fic(gr.top_block):
    def __init__(self, mode, freq, gain, conf, samp=None, usrp=True):
        gr.top_block.__init__(self)
        
        if usrp: # set up usrp source
            src = dabp_usrp.setup_usrp(freq, gain)
        else: # set up sample file source
            sinput = gr.file_source(gr.sizeof_short, samp.encode('ascii'), False)
            src = gr.interleaved_short_to_complex()
            self.connect(sinput, src)
            
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
        