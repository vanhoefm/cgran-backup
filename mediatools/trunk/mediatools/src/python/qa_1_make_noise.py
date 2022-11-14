#!/usr/bin/python

from gnuradio import gr,audio_oss;

tb = gr.top_block();
src = gr.file_source(gr.sizeof_float,"/dev/urandom");
sink = audio_oss.sink( 44100, "/dev/dsp", True);

tb.connect(src,sink);
tb.run();





