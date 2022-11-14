#
# Copyright 2008 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

##################################################
# Imports
##################################################
import spec_window
import common
from gnuradio import gr, blks2
from pubsub import pubsub
from constants import *

##################################################
# spec sink block (wrapper for old wxgui)
##################################################
class _spec_sink_base(gr.hier_block2):
	"""
	An spec block with real/complex inputs and a gui window.
	"""

	def __init__(
		self,
		parent,
		baseband_freq=0,
		ref_scale=2.0,
		y_per_div=10,
		y_divs=8,
		ref_level=50,
		sample_rate=1,
		spec_size=512,
		spec_rate=spec_window.DEFAULT_FRAME_RATE,
		average=False,
		avg_alpha=None,
		title='',
		size=spec_window.DEFAULT_WIN_SIZE,
		peak_hold=False,
	):
		#ensure avg alpha
		if avg_alpha is None: avg_alpha = 2.0/spec_rate
		#init
		gr.hier_block2.__init__(
			self,
			"spec_sink",
			gr.io_signature(1, 1, self._item_size),
			gr.io_signature(0, 0, 0),
		)
		#blocks
		msgq = gr.msg_queue(2)
		sink = gr.message_sink(gr.sizeof_float*spec_size, msgq, True)
		#connect
		self.connect(self, sink)
		#controller
		self.controller = pubsub()
		self.controller.subscribe(AVERAGE_KEY, spec.set_average)
		self.controller.publish(AVERAGE_KEY, spec.average)
		self.controller.subscribe(AVG_ALPHA_KEY, spec.set_avg_alpha)
		self.controller.publish(AVG_ALPHA_KEY, spec.avg_alpha)
		self.controller.subscribe(SAMPLE_RATE_KEY, spec.set_sample_rate)
		self.controller.publish(SAMPLE_RATE_KEY, spec.sample_rate)
		#start input watcher
		common.input_watcher(msgq, self.controller, MSG_KEY)
		#create window
		self.win = spec_window.spec_window(
			parent=parent,
			controller=self.controller,
			size=size,
			title=title,
			real=self._real,
			spec_size=spec_size,
			baseband_freq=baseband_freq,
			sample_rate_key=SAMPLE_RATE_KEY,
			y_per_div=y_per_div,
			y_divs=y_divs,
			ref_level=ref_level,
			average_key=AVERAGE_KEY,
			avg_alpha_key=AVG_ALPHA_KEY,
			peak_hold=peak_hold,
			msg_key=MSG_KEY,
		)
		common.register_access_methods(self, self.win)
		setattr(self.win, 'set_baseband_freq', getattr(self, 'set_baseband_freq')) #BACKWARDS
		setattr(self.win, 'set_peak_hold', getattr(self, 'set_peak_hold')) #BACKWARDS

class spec_sink_f(_spec_sink_base):
	_spec_chain = blks2.logpwrspec_f
	_item_size = gr.sizeof_float
	_real = True

class spec_sink_c(_spec_sink_base):
	_spec_chain = blks2.logpwrspec_c
	_item_size = gr.sizeof_gr_complex
	_real = False

# ----------------------------------------------------------------
# Standalone test app
# ----------------------------------------------------------------

import wx
from gnuradio.wxgui import stdgui2

class test_app_block (stdgui2.std_top_block):
    def __init__(self, frame, panel, vbox, argv):
        stdgui2.std_top_block.__init__ (self, frame, panel, vbox, argv)

        spec_size = 256

        # build our flow graph
        input_rate = 20.48e3

        # Generate a complex sinusoid
        #src1 = gr.sig_source_c (input_rate, gr.GR_SIN_WAVE, 2e3, 1)
        src1 = gr.sig_source_c (input_rate, gr.GR_CONST_WAVE, 5.75e3, 1)

        # We add these throttle blocks so that this demo doesn't
        # suck down all the CPU available.  Normally you wouldn't use these.
        thr1 = gr.throttle(gr.sizeof_gr_complex, input_rate)

        sink1 = spec_sink_c (panel, title="Complex Data", spec_size=spec_size,
                            sample_rate=input_rate, baseband_freq=100e3,
                            ref_level=0, y_per_div=20, y_divs=10)
        vbox.Add (sink1.win, 1, wx.EXPAND)

        self.connect(src1, thr1, sink1)

        #src2 = gr.sig_source_f (input_rate, gr.GR_SIN_WAVE, 2e3, 1)
        src2 = gr.sig_source_f (input_rate, gr.GR_CONST_WAVE, 5.75e3, 1)
        thr2 = gr.throttle(gr.sizeof_float, input_rate)
        sink2 = spec_sink_f (panel, title="Real Data", spec_size=spec_size*2,
                            sample_rate=input_rate, baseband_freq=100e3,
                            ref_level=0, y_per_div=20, y_divs=10)
        vbox.Add (sink2.win, 1, wx.EXPAND)

        self.connect(src2, thr2, sink2)

def main ():
    app = stdgui2.stdapp (test_app_block, "spec Sink Test App")
    app.MainLoop ()

if __name__ == '__main__':
    main ()
