from gnuradio import gr
import dabp
import dabp_usrp
import subprocess

class dabp_msc(gr.top_block):
    def __init__(self, mode, freq, gain, conf, subchid, record=None, usrp=True):
        gr.top_block.__init__(self)
        
        if usrp: # set up usrp source
            src = dabp_usrp.setup_usrp(freq, gain)
        else: # set up sample file source
            sinput = gr.file_source(gr.sizeof_short, samp.encode('ascii'), False)
            src = gr.interleaved_short_to_complex()
            self.connect(sinput, src)
            
        # channel configuration
        cc = dabp.channel_conf(conf)
        #sys.stderr.write("Playing Channel: "+cc.get_label(options.subchid)+"\n")
        
        start_addr = cc.get_start_addr(subchid)
        subchsz = cc.get_subchsz(subchid)
        optprot = cc.get_optprot(subchid)
        #print "start_addr=%d, subchsz=%d, optprot=%d" % (start_addr, subchsz, optprot)
        
        param = dabp.parameters(mode)
        
        # null symbol detector
        samples_per_null_sym=param.get_Nnull()
        #print "samples per null symbol : %d" % samples_per_null_sym
        nulldet = dabp.detect_null(samples_per_null_sym)
        # ofdm demod
        demod = dabp.ofdm_demod(mode)
        # FIC/MSC demultiplexer
        demux = dabp.fic_msc_demux(1, mode)
        
        cifsz = param.get_cifsz()
        scs = dabp.subchannel_selector(cifsz,start_addr,subchsz)
        intlv = dabp.time_deinterleaver(subchsz)
        punc = dabp.depuncturer(subchsz,optprot)
        I = punc.getI()
        vit = dabp.vitdec(I)
        scram = dabp.scrambler(I)
        len_logfrm=scram.get_nbytes()
        sync = dabp.super_frame_sync(len_logfrm)
        subchidx = sync.get_subchidx()
        rs = dabp.super_frame_rsdec(subchidx)
        if record == None:
            mplayer = subprocess.Popen(['mplayer', '-af', 'volume=20', '-ac', '+faad', '-rawaudio', 'format=0xff', '-demuxer', '+rawaudio','-'], stdin=subprocess.PIPE)
            dst = dabp.super_frame_sink(subchidx, mplayer.stdin.fileno())
        else:
            dst = dabp.super_frame_sink(subchidx, record.encode('ascii'))
        
        nullsink = gr.null_sink(gr.sizeof_char)
        # connect everything
        self.connect(src, nulldet)
        self.connect(src, (demod,0))
        self.connect(nulldet, (demod,1))
        self.connect((demod,0), demux, scs, intlv, punc, vit, scram, sync, rs, dst)
        self.connect((demod,1), nullsink)
        
