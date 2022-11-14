#
# Copyright 2011 Anton Blad.
# 
# This file is part of OpenRD
# 
# OpenRD is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# OpenRD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from PyQt4 import QtGui, QtCore
import math, sip
from gnuradio import gr, qtgui

class signal_viewer_dummy(gr.hier_block2):
	def __init__(self):
		gr.hier_block2.__init__(self, "signal_viewer_dummy",
				gr.io_signature(2, 2, gr.sizeof_gr_complex),
				gr.io_signature(0, 0, 0))

		self.s0 = gr.null_sink(gr.sizeof_gr_complex)
		self.s1 = gr.null_sink(gr.sizeof_gr_complex)

		self.connect((self, 0), self.s0)
		self.connect((self, 1), self.s1)

class number_viewer_dummy:
	def __init__(self):
		pass

	def add_values(self, values):
		pass

	def set_value(self, value):
		pass

class number_viewer:
	def __init__(self, name, average=1, db=False):
		self.d_name = name
		self.d_average = average
		self.d_db = db
		self.d_values = [0]*average
	
	def add_values(self, values):
		values = list(values)
		if(len(values) >= self.d_average):
			self.d_values = values[-self.d_average:]
		else:
			self.d_values = self.d_values[-(self.d_average-len(values)):] + values[:]

		v = sum(self.d_values)/len(self.d_values)
		if self.d_db:
			if v == 0:
				v = -float('inf')
			else:
				v = 10*math.log10(v)

		self.set_display(v)

	def set_value(self, value):
		if self.d_db:
			if v == 0:
				v = -float('inf')
			else:
				v = 10*math.log10(value)
		else:
			v = value

		self.set_display(v)

	def set_display(self, v):
		raise "number_viewer.set_display : abstract function"

class number_viewer_text(number_viewer):
	def __init__(self, name, average=1, db=False):
		number_viewer.__init__(self, name, average, db)

	def set_display(self, v):
		print "%s: %f" % (self.d_name, v)

class signal_viewer_qt(gr.hier_block2, QtGui.QWidget):
	def __init__(self, link_rate, symbol_length, fft_size, desc="%s", disp_mode=0):
		gr.hier_block2.__init__(self, 'signal_viewer',
				gr.io_signature(2, 2, gr.sizeof_gr_complex),
				gr.io_signature(0, 0, 0))
		QtGui.QWidget.__init__(self, None)
		
		self.d_link_rate = link_rate
		self.d_symbol_length = symbol_length
		self.d_fft_size = fft_size
		self.d_disp_mode = disp_mode

		if self.d_disp_mode == 0:
			self.layout = QtGui.QHBoxLayout(self)
		elif self.d_disp_mode == 1:
			self.layout = QtGui.QVBoxLayout(self)

		self.spectrum_sink = qtgui.sink_c(fft_size, gr.firdes.WIN_BLACKMAN_hARRIS,
				0, link_rate, desc % "Spectrum",
				True, True, True, False)
		self.constellation_sink = qtgui.sink_c(fft_size, gr.firdes.WIN_BLACKMAN_hARRIS,
				0, link_rate, desc % "Constellation",
				False, False, False, True)

		self.spectrum_sink_widget = sip.wrapinstance(self.spectrum_sink.pyqwidget(), QtGui.QWidget)
		self.constellation_sink_widget = sip.wrapinstance(self.constellation_sink.pyqwidget(), QtGui.QWidget)

		self.layout.addWidget(self.spectrum_sink_widget)
		self.layout.addWidget(self.constellation_sink_widget)

		self.connect((self, 0), self.spectrum_sink)
		self.connect((self, 1), self.constellation_sink)

class number_viewer_qt(number_viewer, QtGui.QWidget):
	def __init__(self, name, average=1, db=False):
		number_viewer.__init__(self, name, average, db)
		QtGui.QWidget.__init__(self, None)

		self.layout = QtGui.QHBoxLayout(self)

		self.name_label = QtGui.QLabel("%s: " % name)
		self.number_view = QtGui.QLabel()

		self.layout.addWidget(self.name_label)
		self.layout.addWidget(self.number_view)

	def set_display(self, v):
		self.number_view.setNum(v)

