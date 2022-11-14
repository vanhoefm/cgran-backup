#!/usr/bin/env python

# to pipe to mplayer using
# ./logfrm2aac.py | mplayer -cache 512 -
# -cache 512 is necessary

from gnuradio import gr
from gnuradio import dabp

class logfrm2aac(gr.top_block):

    def __init__ (self):
        gr.top_block.__init__(self)
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
        #dst = dabp.super_frame_sink(subchidx,"test123.aac")
        dst = dabp.super_frame_sink(subchidx,"-")
        self.connect (src, sync, rs, dst)

if __name__ == '__main__':
    try:
        logfrm2aac().run()
    except KeyboardInterrupt:
        pass
    

