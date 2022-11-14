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

# This is a modification of const_window.py from GNU Radio.

##################################################
# Imports
##################################################
from gnuradio.wxgui import plotter
from gnuradio.wxgui import common
import wx
import numpy
import math
from gnuradio.wxgui import pubsub
from gnuradio.wxgui.constants import *
from gnuradio import gr
from gnuradio.wxgui import forms

##################################################
# Constants
##################################################
SLIDER_STEPS = 200
DEFAULT_FRAME_RATE = gr.prefs().get_long('wxgui', 'const_rate', 5)
DEFAULT_WIN_SIZE = (250, 250)
DEFAULT_CONST_SIZE = gr.prefs().get_long('wxgui', 'const_size', 2048)
CONST_PLOT_COLOR_SPEC = (0, 0, 1)
MARKER_TYPES = (
	('Dot Small', 1.0),
	('Dot Medium', 2.0),
	('Dot Large', 3.0),
	('Line Link', None),
)
DEFAULT_MARKER_TYPE = 2.0

DB_DIV_MIN, DB_DIV_MAX = 1, 20

# range of decimation slider
DECIM_RATE_MIN = 200			#frame rate 30 @ 3,125MSps and const size 512 
DECIM_RATE_MAX = 6000			#frame rate 1  @ 3,125MSps and const size 512
DECIM_RATE_SLIDER_STEPS = 4000

##################################################
# Constellation window control panel
##################################################
class control_panel(wx.Panel):
	"""
	A control panel with wx widgits to control the plotter.
	"""
	def __init__(self, parent):
		"""
		Create a new control panel.
		@param parent the wx parent window
		"""
		self.parent = parent
		wx.Panel.__init__(self, parent, style=wx.SUNKEN_BORDER)
		parent[SHOW_CONTROL_PANEL_KEY] = True
		parent.subscribe(SHOW_CONTROL_PANEL_KEY, self.Show)
		control_box = forms.static_box_sizer(
			parent=self, label='Options',
			bold=True, orient=wx.VERTICAL,
		)
		#marker
		control_box.AddStretchSpacer()
		forms.drop_down(
			sizer=control_box, parent=self,
			ps=parent, key=MARKER_KEY, label='Marker',
			choices=map(lambda x: x[1], MARKER_TYPES),
			labels=map(lambda x: x[0], MARKER_TYPES),
		)
		#boxes for frame rate and decimation
		frame_rate_text = forms.static_text(
			sizer=control_box, parent=self, label='Decimation rate',
			converter=forms.float_converter(lambda x: '%.4f'%x),
			ps=parent, key=SAMPLE_RATE_KEY, width=50,
		)
		frame_rate_slider = forms.slider(
			sizer=control_box, parent=self,
			minimum=DECIM_RATE_MIN,
			maximum=DECIM_RATE_MAX,
			num_steps=DECIM_RATE_SLIDER_STEPS,
			ps=parent, key=SAMPLE_RATE_KEY,
		)
		#buttons for size
		control_box.AddStretchSpacer()
		y_ctrl_box = forms.static_box_sizer(
			parent=self, sizer=control_box, label='Axis Options',
			bold=True, orient=wx.VERTICAL,
		)
		#axis scale
		forms.incr_decr_buttons(
			parent=self, sizer=y_ctrl_box, label='Inphase',
			on_incr=self.parent.on_incr_x_level, on_decr=self.parent.on_decr_x_level,
		)
		forms.incr_decr_buttons(
			parent=self, sizer=y_ctrl_box, label='Quadrature',
			on_incr=self.parent.on_incr_y_level, on_decr=self.parent.on_decr_y_level,
		)
		y_ctrl_box.AddSpacer(2)
		#autoscale
		forms.single_button(
			sizer=y_ctrl_box, parent=self, label='Autoscale',
			callback=self.parent.autoscale,
		)
		#run/stop
		control_box.AddStretchSpacer()
		forms.toggle_button(
			sizer=control_box, parent=self,
			true_label='Stop', false_label='Run',
			ps=parent, key=RUNNING_KEY,
		)
		#set sizer
		self.SetSizerAndFit(control_box)

	##################################################
	# subscriber handlers
	##################################################
        def _update_layout(self,key):
          # Just ignore the key value we get
          # we only need to now that the visability or size of something has changed
          self.parent.Layout()
          #self.parent.Fit()  
          
##################################################
# Constellation window with plotter and control panel
##################################################
class const_window(wx.Panel, pubsub.pubsub):
	def __init__(
		self,
		parent,
		controller,
		size,
		title,
		msg_key,
		decim_rate_key,
		x_max,
		y_max,
	):
		pubsub.pubsub.__init__(self)
		#proxy the keys
		self.proxy(MSG_KEY, controller, msg_key)
		self.proxy(SAMPLE_RATE_KEY, controller, decim_rate_key)
		#initialize values
		self[RUNNING_KEY] = True
		self.x_per_div = 10			# +/- 10 in axis scale
		self.y_per_div = 10
		self.x_divs = 6				# nr of grid divs
		self.y_divs = 6
		self[MARKER_KEY] = DEFAULT_MARKER_TYPE
		self.x_max = x_max
		self.y_max = y_max
		
		#init panel and plot
		wx.Panel.__init__(self, parent, style=wx.SIMPLE_BORDER)
		self.plotter = plotter.channel_plotter(self)
		self.plotter.SetSize(wx.Size(*size))
		self.plotter.set_title(title)
		self.plotter.set_x_label('Inphase')
		self.plotter.set_y_label('Quadrature')
		self.plotter.enable_point_label(True)
		self.plotter.enable_grid_lines(True)
		#setup the box with plot and controls
		self.control_panel = control_panel(self)
		main_box = wx.BoxSizer(wx.HORIZONTAL)
		main_box.Add(self.plotter, 1, wx.EXPAND)
		main_box.Add(self.control_panel, 0, wx.EXPAND)
		self.SetSizerAndFit(main_box)
		#register events
		self.subscribe(MSG_KEY, self.handle_msg)
		#initial update
		self.update_grid()

	def handle_msg(self, msg):
		"""
		Plot the samples onto the complex grid.
		@param msg the array of complex samples
		"""
		if not self[RUNNING_KEY]: return
		#convert to complex floating point numbers
		samples = numpy.fromstring(msg, numpy.complex64)
		real = numpy.real(samples)
		imag = numpy.imag(samples)
		self.samples = samples
		self.samples_re = real
		self.samples_im = imag
		#plot
		self.plotter.set_waveform(
			channel=0,
			samples=(real, imag),
			color_spec=CONST_PLOT_COLOR_SPEC,
			marker=self[MARKER_KEY],
		)
		#update the plotter
		self.plotter.update()

	def update_grid(self):
		#grid parameters
		x_max = self.x_max
		y_max = self.y_max
		
		#update the x axis
		self.plotter.set_x_grid(-x_max, x_max, common.get_clean_num(2.0*x_max/self.x_divs))
		#update the y axis
		self.plotter.set_y_grid(-y_max, y_max, common.get_clean_num(2.0*y_max/self.y_divs))
		#update plotter
		self.plotter.update()

	def autoscale(self, *args):
		"""
		Autoscale the fft plot to the last frame.
		Set the dynamic range and reference level.
		"""
		if not len(self.samples): return
		min_level_re, max_level_re = common.get_min_max(self.samples_re)
		min_level_im, max_level_im = common.get_min_max(self.samples_im)
		min_level = min(min_level_re,min_level_im)
		max_level = min(max_level_re,max_level_im)
		#set the range to a clean number of the dynamic range
		self.x_per_div = common.get_clean_num(1+(max_level - min_level)/self.x_divs)
		self.y_per_div = common.get_clean_num(1+(max_level - min_level)/self.y_divs)
		#set the reference level to a multiple of y per div
		self.y_max = self.y_per_div*round(.5+max_level/self.y_per_div)
		self.x_max = self.x_per_div*round(.5+max_level/self.x_per_div)
		
		self.update_grid()

	def on_incr_y_level(self, event):
		self.y_max = self.y_max + self.y_per_div
		self.update_grid()
	def on_decr_y_level(self, event):
		self.y_max = self.y_max - self.y_per_div
		self.update_grid()
	def on_incr_x_level(self, event):
		self.x_max = self.x_max + self.x_per_div
		self.update_grid()
	def on_decr_x_level(self, event):
		self.x_max = self.x_max - self.x_per_div
		self.update_grid()



