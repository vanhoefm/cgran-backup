#!/usr/bin/env python

# to pipe to mplayer using
# ./cif2aac.py -f infile | mplayer -ac faad -rawaudio format=0xff -demuxer rawaudio -
# this is to tell mplayer that this is a raw AAC (ADTS) audio stream and to use faad codec

from gnuradio import gr
from gnuradio import dabp
import sys, getopt

class cif2aac(gr.top_block):

    def __init__ (self, argv):
        gr.top_block.__init__(self)
        try:
            opts, args = getopt.getopt(argv, "f:",["infile="])
        except getopt.GetoptError:
            print "please specify input file name using -f or --infile ..."
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-f","--infile"):
                infile=arg
        try:
            src = gr.file_source (8,infile,False)
        except NameError:
            print "please specify input file name using -f or --infile ..."
            sys.exit(2)
        cifsz=55296
        start_addr=0 #36 #72 #108 #144 #240 #288 #336 #384 #432 #504 #576
        subchsz=36 #36 #36 #36 #48 #48 #48 #48 #48 #72 #72 #24
        optprot=2
        #I=32*6*subchsz/6
        #len_logfrm=I/8
        #subchidx=len_logfrm/24
        
        thr = gr.throttle(8,cifsz*4/0.096) # 4 CIFs / DAB frame (96 ms)
        d2f = dabp.d2f ()
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
        dst = dabp.super_frame_sink(subchidx,"-")
        self.connect (src, thr, d2f, scs, intlv, punc, vit, scram, sync, rs, dst)

if __name__ == '__main__':
    try:
        cif2aac(sys.argv[1:]).run()
    except KeyboardInterrupt:
        pass
    

