#!/usr/bin/env python

from gnuradio import gr, eng_notation, blks, tait, audio
from gnuradio.eng_option import eng_option
from gnuradio.wxgui import stdgui, fftsink, waterfallsink, scopesink, form, slider
from optparse import OptionParser
import wx
import sys

class app_flow_graph(stdgui.gui_flow_graph):
	def __init__(self, frame, panel, vbox, argv):
		stdgui.gui_flow_graph.__init__(self)
		
		self.frame = frame
		self.panel = panel
			
		sin_freq = 60000
		#sin_freq = 2970
		ampl = 5
		sample_rate = 5e6
		# Tests the biquad4 signal proccesing block.
		self.SRC = gr.sig_source_f (sample_rate, gr.GR_SIN_WAVE, sin_freq, ampl)
		ADD_DC_OFFSET = gr.add_const_ff(2)
		REMOVE_DC_OFFSET = tait.DC_corrector_ff(0.95)
		
		
		#FS_BEFORE_DC = gr.file_sink (gr.sizeof_float, "before_dc.dat")
		#FS_AFTER_DC = gr.file_sink (gr.sizeof_float, "after_dc.dat")
		
		# We add these throttle blocks so that this demo doesn't
		# suck down all the CPU available.  Normally you wouldn't use these.
		thr1 = gr.throttle(gr.sizeof_float, sample_rate)
		thr2 = gr.throttle(gr.sizeof_float, sample_rate)
		
		self.scope1 = scopesink.scope_sink_f(self, panel, sample_rate=sample_rate, v_scale=1,
		title='Before DC removal')
		self.scope2 = scopesink.scope_sink_f(self, panel, sample_rate=sample_rate, v_scale=1,
		title='After DC removal')
		#self.AUDIO = audio.sink(sample_rate)
		
		self.connect(self.SRC, ADD_DC_OFFSET, REMOVE_DC_OFFSET, thr1, self.scope2)
		self.connect(ADD_DC_OFFSET, thr2, self.scope1)
		#self.connect(REMOVE_DC_OFFSET, self.AUDIO)
		
		#self.connect(self.SRC, ADD_DC_OFFSET, REMOVE_DC_OFFSET, FS_AFTER_DC)
		#self.connect(ADD_DC_OFFSET, FS_BEFORE_DC)
		
		#self.scope = scopesink.scope_sink_f(self, panel, sample_rate=sample_rate)
		#self.connect(self.SRC, self.scope)		
		self._build_gui(vbox)


	def _build_gui(self, vbox):
		vbox.Add(self.scope1.win, 10, wx.EXPAND)
		vbox.Add(self.scope2.win, 10, wx.EXPAND)
	
def main ():
	app = stdgui.stdapp(app_flow_graph, "USRP O'scope", nstatus=1)
	app.MainLoop()

if __name__ == '__main__':
	main ()
