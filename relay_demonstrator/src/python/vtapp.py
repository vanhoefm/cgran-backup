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

import sys
import time
#import cv
from threading import Thread

def num_proto(app):
	if app == 'univideo':
		return 1
	else:
		raise ValueError("Invalid application: %s" % app)

def src(tx, app):
	if app == 'univideo':
		return univideo_src(tx)
	else:
		raise ValueError("Invalid application: %s" % app)

def dest(rx, app):
	if app == 'univideo':
		return univideo_dest(rx)
	else:
		raise ValueError("Invalid application: %s" % app)

class univideo_src(Thread):
	def __init__(self, tx):
		Thread.__init__(self)
		self.tx = tx
		self.d_quit = False
		self.d_fps = 10
		self.d_size = (320,240)

	def quit(self):
		self.d_quit = True

	def set_fps(self, fps):
		self.d_fps = fps

	def set_size(self, size):
		self.d_size = size

	def run(self):
		time.sleep(1)
		self.camera = camera()
		while not self.d_quit:
			data = self.camera.get_image(self.d_size)
			self.tx.send_string(0, data)
			time.sleep(1/self.d_fps)

class univideo_dest(Thread):
	def __init__(self, rx):
		Thread.__init__(self)
		self.rx = rx
		self.d_quit = False
		self.d_size = (320,240)

	def quit(self):
		self.d_quit = True

	def set_size(self, size):
		self.d_size = size

	def run(self):
		self.display = display()
		while not self.d_quit:
			data = self.rx.recv_string(0)
			if data is not None:
				self.display.show_image(data, self.d_size)
			self.display.handle_events()

class camera:
	def __init__(self):
		self.camera = cv.CaptureFromCAM(0)

	def get_image(self, size):
		im = cv.QueryFrame(self.camera)
		imlow = cv.CreateImage(size, im.depth, im.nChannels)
		cv.Resize(im, imlow, cv.CV_INTER_NN)
		return imlow.tostring()

class display:
	def __init__(self):
		cv.NamedWindow("display", 1)
	
	def show_image(self, data, size):
		im = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)
		cv.SetData(im, data)
		cv.ShowImage("display", im)
	
	def handle_events(self):
		cv.WaitKey(10)

