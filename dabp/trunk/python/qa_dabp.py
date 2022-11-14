#!/usr/bin/env python

from gnuradio import gr, gr_unittest
import dabp

class qa_dabp (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None
    """
    def test_001_cif2aac (self):
        src = gr.file_source (8,"../../../cif_M204.dat",False)
        d2f = dabp.d2f ()
        cifsz=55296
        start_addr=504
        subchsz=72
        numcifs=1000
        optprot=2
        I=32*6*subchsz/6
        len_logfrm=I/8
        subchidx=len_logfrm/24
        scs = dabp.subchannel_selector(cifsz,start_addr,subchsz)
        head = gr.head(4,cifsz*numcifs)
        intlv = dabp.time_deinterleaver(subchsz)
        punc = dabp.depuncturer(subchsz,optprot)
        vit = dabp.vitdec(I)
        scram = dabp.scrambler(I)
        sync = dabp.super_frame_sync(len_logfrm)
        rs = dabp.super_frame_rsdec(subchidx)
        dst = dabp.super_frame_sink(subchidx,"dabp_test.aac")
        self.tb.connect (src, d2f, head)
        self.tb.connect (head, scs, intlv, punc, vit, scram, sync, rs, dst)
        self.tb.run ()
    """
    """
    def test_001_msc_dec (self):
        #import os
        #print 'Blocked waiting for gdb attach (pid=%d)' % (os.getpid(),)
        #raw_input('Press Enter to continue:')
        
        src = gr.file_source (8,"../../../cif_M204.dat",False)
        d2f = dabp.d2f ()
        cifsz=55296
        start_addr=504
        subchsz=72
        numcifs=4*16
        optprot=2
        I=32*6*subchsz/6
        len_logfrm=I/8
        subchidx=len_logfrm/24
        scs = dabp.subchannel_selector(cifsz,start_addr,subchsz)
        head = gr.head(4,cifsz*numcifs)
        intlv = dabp.time_deinterleaver(subchsz)
        punc = dabp.depuncturer(subchsz,optprot)
        vit = dabp.vitdec(I)
        scram = dabp.scrambler(I)
        dst = gr.file_sink(1,"logfrm.dat")
        #self.tb.connect (src, d2f, head)
        #self.tb.connect (head, scs, intlv, punc, vit, scram, dst)
        self.tb.connect(src,d2f,scs,intlv,punc,vit,scram,dst)
        self.tb.run ()
    """
    
    def test_001_super_frame_dec (self):
        import os
        print 'Blocked waiting for gdb attach (pid=%d)' % (os.getpid(),)
        raw_input('Press Enter to continue:')
        
        src = gr.file_source (1,"logfrm.dat",False)
        cifsz=55296
        start_addr=504
        subchsz=72
        optprot=2
        I=32*6*subchsz/6
        len_logfrm=I/8
        subchidx=len_logfrm/24
        sync = dabp.super_frame_sync(len_logfrm)
        rs = dabp.super_frame_rsdec(subchidx)
        dst = dabp.super_frame_sink(subchidx,"dabp_test.aac")
        self.tb.connect (src, sync, rs, dst)
        self.tb.run ()
    """
    def test_001_grbuf (self):
        import os
        print 'Blocked waiting for gdb attach (pid=%d)' % (os.getpid(),)
        raw_input('Press Enter to continue:')
        
        src = gr.file_source (2,"seq.dat",False)
        len=8
        tt = dabp.test_grbuf(len)
        dst = gr.null_sink(2)
        self.tb.connect (src, tt, dst)
        self.tb.run ()
    """

if __name__ == '__main__':
    gr_unittest.main ()

