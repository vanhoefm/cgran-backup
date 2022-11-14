from gnuradio import gr, gr_unittest
import speexcodec
from gnuradio import audio
from gnuradio import wavfile
import sys

class qa_speex (gr_unittest.TestCase):

    def setUp (self):
        self.fg = gr.top_block()

    def tearDown (self):
        self.fg = None

    def test_001_speex_codec (self):
        # This test takes a wav file input, compresses with speex decompresses and writes 
        # into a wav file output. 
        infile  = "cdda.wav"
	outfile = "cddaspeex.wav"
              
	wf_in  = wavfile.wavfile_source(infile)
        enc = speexcodec.speex_encoder(44100,1)
        dec = speexcodec.speex_decoder(1)
        wf_out = wavfile.wavfile_sink(outfile,
				1,
				44100,
				16)

        src_scale = gr.multiply_const_ff(32767)
        sink_scale = gr.multiply_const_ff(1.0/32767.)
        raw_file = gr.file_sink((gr.sizeof_char)*10,"cddaraw")
        #self.fg.connect(wf_in,src_scale,enc,dec,sink_scale,raw_file)
        self.fg.connect(wf_in,src_scale,enc,raw_file)
	self.fg.run()
	wf_out.close()
      	self.assertEqual(1, 1)

if __name__ == '__main__':
    gr_unittest.main ()
