#!/usr/bin/env python
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

import vtapp
import Queue

class dummy_link:
	def __init__(self):
		self.queue = Queue.Queue()

	def send_data(self, proto, data):
		self.queue.put(data)

	def recv_data(self, proto):
		try:
			return self.queue.get(True, 1)
		except Empty:
			return None

link = dummy_link()
src = vtapp.src(link, 'univideo')
dest = vtapp.dest(link, 'univideo')
src.start()
dest.start()

raw_input('Press enter to exit')

src.quit()
dest.quit()

src.join()
dest.join()

