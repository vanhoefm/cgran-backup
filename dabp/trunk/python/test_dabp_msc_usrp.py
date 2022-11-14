#!/usr/bin/env python
# this is to test the gr-dabp package for MSC
# to pipe to mplayer using
# ./test_dabp_msc_usrp -i subchid -n channel.conf -f carrier_freq -g gain | mplayer -ac faad -rawaudio format=0xff -demuxer rawaudio -
# this is to tell mplayer that this is a raw AAC (ADTS) audio stream and to use faad codec

from gnuradio import gr
import dabp
from gnuradio import usrp
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import sys
import subprocess

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
        parser.add_option("-i", "--subchid", type="int", default=-1,
            help="Subchannel ID")
        parser.add_option("-n", "--config", type="string", default="channel3.conf",
            help="DAB+ channel configuration file [default=%default]")
        parser.add_option("-o", "--output", type="string", default="-",
            help="Output AAC file [default=%default]")
        parser.add_option("-p", "--mplayer", action="store_true", default=False,
            help="Play the program by MPlayer [default=%default, stdout]")
        (options, args) = parser.parse_args ()
        
        if len(args)>0:
            parser.print_help()
            raise SystemExit, 1
        
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
        
        
        # channel configuration
        cc = dabp.channel_conf(options.config)
        sys.stderr.write("Playing Channel: "+cc.get_label(options.subchid)+"\n")
        
        start_addr = cc.get_start_addr(options.subchid)
        subchsz = cc.get_subchsz(options.subchid)
        optprot = cc.get_optprot(options.subchid)
        #print "start_addr=%d, subchsz=%d, optprot=%d" % (start_addr, subchsz, optprot)
        
        mode=options.dab_mode
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
        if options.mplayer:
            mplayer = subprocess.Popen(['mplayer', '-ac', 'faad', '-rawaudio', 'format=0xff', '-demuxer', 'rawaudio','-'], stdin=subprocess.PIPE)
            dst = dabp.super_frame_sink(subchidx, mplayer.stdin.fileno())
        else:
            dst = dabp.super_frame_sink(subchidx, options.output)
        
        nullsink = gr.null_sink(gr.sizeof_char)
        # connect everything
        self.connect(self.u, nulldet)
        self.connect(self.u, (demod,0))
        self.connect(nulldet, (demod,1))
        self.connect((demod,0), demux, scs, intlv, punc, vit, scram, sync, rs, dst)
        self.connect((demod,1), nullsink)

        # debug
        #self.connect(nulldet, gr.file_sink(gr.sizeof_char, "debug_nulldet.dat"))
        #self.connect(vit, gr.file_sink(gr.sizeof_char, "debug_vit.dat"))
        
if __name__=='__main__':
    try:
        tb=test_dabp()
        tb.run()
    except KeyboardInterrupt:
        pass

