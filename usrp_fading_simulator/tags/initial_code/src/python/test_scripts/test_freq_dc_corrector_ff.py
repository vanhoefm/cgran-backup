#!/usr/bin/env python

#Tests the frequancy response of the dc_corrector_ff block

from gnuradio import gr, eng_notation, blks, tait, audio
from gnuradio.eng_option import eng_option
from gnuradio.wxgui import stdgui, fftsink
from optparse import OptionParser
import wx
import sys

class app_flow_graph(stdgui.gui_flow_graph):
	def __init__(self, frame, panel, vbox, argv):
		stdgui.gui_flow_graph.__init__(self)
		
		self.frame = frame
		self.panel = panel
		frequency = 425.75
			
		USRP = usrp_source_c(self, 	# Flow graph
			None, 			# Daugherboard spec
			250,		     	# IF decimation ratio
			None)	    		# Receiver gain
		USRP.tune(frequency)

		

		ADD_DC_OFFSET = gr.add_const_ff(2)
		REMOVE_DC_OFFSET = tait.DC_corrector_ff(0.95)
		
		
		#FS_BEFORE_DC = gr.file_sink (gr.sizeof_float, "before_dc.dat")
		#FS_AFTER_DC = gr.file_sink (gr.sizeof_float, "after_dc.dat")
		
		self.scope1 = scopesink.scope_sink_f(self, panel, sample_rate=sample_rate, v_scale=1,
		title='Before DC removal')
		self.scope2 = scopesink.scope_sink_f(self, panel, sample_rate=sample_rate, v_scale=1,
		title='After DC removal')
		self.AUDIO = audio.sink(sample_rate)
		
		self.connect(self.SRC, self.AUDIO)
		self.connect(self.SRC, ADD_DC_OFFSET, self.scope1)
		self.connect(ADD_DC_OFFSET, REMOVE_DC_OFFSET, self.scope2)
		
		
		#self.connect(self.SRC, ADD_DC_OFFSET, REMOVE_DC_OFFSET, FS_AFTER_DC)
		#self.connect(ADD_DC_OFFSET, FS_BEFORE_DC)
		
		#self.scope = scopesink.scope_sink_f(self, panel, sample_rate=sample_rate)
		#self.connect(self.SRC, self.scope)		
		self._build_gui(vbox)


	def _build_gui(self, vbox):
		vbox.Add(self.scope1.win, 10, wx.EXPAND)
		vbox.Add(self.scope2.win, 10, wx.EXPAND)
	
class usrp_source_c(gr.hier_block):
    """
    Create a USRP source object supplying complex floats.
    
    Selects user supplied subdevice or chooses first available one.

    Calibration value is the offset from the tuned frequency to 
    the actual frequency.       
    """
    def __init__(self, fg, subdev_spec, decim, gain=None, calibration=0.0):
	self._decim = decim
        self._src = usrp.source_c()
        if subdev_spec is None:
            subdev_spec = usrp.pick_rx_subdevice(self._src)
        self._subdev = usrp.selected_subdev(self._src, subdev_spec)
        self._src.set_mux(usrp.determine_rx_mux_value(self._src, subdev_spec))
        self._src.set_decim_rate(self._decim)

	# If no gain specified, set to midrange
	if gain is None:
	    g = self._subdev.gain_range()
	    gain = (g[0]+g[1])/2.0

        self._subdev.set_gain(gain)
        self._cal = calibration
	gr.hier_block.__init__(self, fg, self._src, self._src)

    def tune(self, freq):
    	result = usrp.tune(self._src, 0, self._subdev, freq+self._cal)
    	# TODO: deal with residual

    def rate(self):
	return self._src.adc_rate()/self._decim
	
def main ():
	app = stdgui.stdapp(app_flow_graph, "USRP O'scope", nstatus=1)
	app.MainLoop()

if __name__ == '__main__':
	main ()
