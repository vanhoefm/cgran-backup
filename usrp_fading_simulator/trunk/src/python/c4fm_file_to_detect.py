#!/usr/bin/env python

from gnuradio import gr, gru, usrp, optfir, audio, eng_notation, blks, tait

# Uses a file source, removes DC offset, scales and converts before feeding
# into c4fm detection. The file should contain demodulated C4 FM.

class app_flow_graph(gr.flow_graph):
    def __init__(self):
	gr.flow_graph.__init__(self)
	
	FILE_SOURCE = gr.file_source (gr.sizeof_float, "GNU_C4FM_RxData_MatlabDemod.dat")
	DC_CORRECTOR = tait.DC_corrector_ff(0.95)
	MULTI = gr.multiply_const_ff(32768)
	F_2_S = gr.float_to_short()
	C4FM_Detect = tait.c4fm_detect_s()
	
	self.connect(FILE_SOURCE, DC_CORRECTOR, MULTI, F_2_S, C4FM_Detect)
	
def main():
    fg = app_flow_graph()
    try:
        fg.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
