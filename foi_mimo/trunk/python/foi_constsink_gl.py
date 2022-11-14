#
# Copyright 2011 FOI
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

# This is a modification of constsink_gl.py from GNU Radio.

##################################################
# Imports
##################################################
from foimimo import foi_const_window
from gnuradio.wxgui import common
from gnuradio import gr, blks2
from gnuradio.wxgui.pubsub import pubsub
from gnuradio.wxgui.constants import *

##################################################
# Constellation sink block (wrapper for old wxgui)
##################################################
class const_sink_c(gr.hier_block2, common.wxgui_hb):
	"""
	A constellation block with a gui window.
	"""

	def __init__(
		self,
		parent,
		title='',
		sample_rate=1,
		size=foi_const_window.DEFAULT_WIN_SIZE,
		frame_rate=foi_const_window.DEFAULT_FRAME_RATE,
		const_size=foi_const_window.DEFAULT_CONST_SIZE,
		symbol_rate=1,
		max_xy_value=4,
	):
		#init
		gr.hier_block2.__init__(
			self,
			"const_sink",
			gr.io_signature(1, 1, gr.sizeof_gr_complex*const_size),
			gr.io_signature(0, 0, 0),
		)
		#blocks
		self.decim = max(1, int(round(sample_rate/const_size/frame_rate)))
		self.decim_block = gr.keep_one_in_n(gr.sizeof_gr_complex*const_size,self.decim)
		msgq = gr.msg_queue(2)
		sink = gr.message_sink(gr.sizeof_gr_complex*const_size, msgq, True)
		#controller
		self.controller = pubsub()
		self.controller.subscribe(SAMPLE_RATE_KEY, self.set_decim)
		self.controller.publish(SAMPLE_RATE_KEY, self.get_decim)
		#initial update
		self.controller[SAMPLE_RATE_KEY] = self.decim
		#start input watcher
		common.input_watcher(msgq, self.controller, MSG_KEY)
		#create window
		self.win = foi_const_window.const_window(
			parent=parent,
			controller=self.controller,
			size=size,
			title=title,
			msg_key=MSG_KEY,
			decim_rate_key=SAMPLE_RATE_KEY,
			x_max=max_xy_value,
			y_max=max_xy_value,
		)
		common.register_access_methods(self, self.win)
		#connect
		self.wxgui_connect(self, self.decim_block, sink)

	def set_decim(self, decim): 
		self.decim_block.set_n(max(1,int(round(decim))))

	def get_decim(self): 
		return self.decim
