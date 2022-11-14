#!/usr/bin/env python

from gnuradio import gr, gru, usrp, optfir, audio, eng_notation, blks, tait

# Test to see how the filter_delay_fc can be used to introduce a delay into a signal path.
# As it turns out the delay introduced is n samples, where n = (ntaps - 1) / 2.
class app_flow_graph(gr.flow_graph):
    def __init__(self):
	gr.flow_graph.__init__(self)
	
	# test signal
	sig_vec = (0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	src = gr.vector_source_f(sig_vec)

	cfir_decimation = 1
	
	# the introduced delay will be equal to n samples, where n = (ntaps - 1) / 2.
	ntaps = 9
        taps = gr.firdes_hilbert (ntaps)
	# This is what is used to introduce the delay into the signal path.
        delay = gr.filter_delay_fc (taps)
	# convert back to floats for sinking to file.
	c2f = gr.complex_to_float()

	dst1 = gr.file_sink(gr.sizeof_float, "delayed_signal.dat") 
	dst2 = gr.file_sink(gr.sizeof_float, "signal.dat") 
	
	# We compare the original signal to the delayed version.
	# As it turns out the delay works as expected.
	self.connect(src, delay, c2f, dst1)
	self.connect(src, dst2)

	
def main():
    fg = app_flow_graph()
    try:
        fg.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
