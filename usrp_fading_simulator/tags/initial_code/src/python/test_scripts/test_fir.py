#!/usr/bin/env python

from gnuradio import gr, gru, usrp, optfir, audio, eng_notation, blks, tait


class app_flow_graph(gr.flow_graph):
    def __init__(self):
	gr.flow_graph.__init__(self)
	
	# Test fir filter.
	impulse = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	IMP_SRC = gr.vector_source_f(impulse)
	
	taps = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
	PST_DMD_FIR = gr.fir_filter_fff(1, taps)
	
	IMP_FILE_SINK = gr.file_sink (gr.sizeof_float, "fir_impulse.dat")
	
	self.connect(IMP_SRC, PST_DMD_FIR, IMP_FILE_SINK)	

	
def main():
    fg = app_flow_graph()
    try:
        fg.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
