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

from gnuradio import gr
from gnuradio import blks2
from gnuradio import usrp

import sys

usrpmapfn = '/tmp/usrp.id'

def serial_to_idx(serial):
	"""Converts a USRP serial number into an index which can be used in
	calls to usrp.sink_x and usrp.source_x. The index is read from a map
	file generated by "cusrp.py scan"
	
	Arguments:
		serial - USRP serial number
	
	Returns:
		USRP bus index"""

	try:
		f = open(usrpmapfn, 'r')
	except:
		print "USRP map file %s does not exist. Run cusrp.py scan." % usrpmapfn
		sys.exit(1)
	
	list = f.readlines()
	for line in list:
		map = line.split()
		if len(map) == 0:
			break
		elif len(map) == 2:
			which = int(map[0])
			mapserial = map[1]
			if serial == mapserial:
				f.close()
				return which
		else:
			print "USRP map file %s is corrupted. Run cusrp.py scan." % usrpmapfn
			sys.exit(1)
	
	print "USRP with serial %s not found in map file. Make sure it is "\
		"connected and run cusrp.py scan" % serial
	sys.exit(1)

def write_serial_map(map):
	"""Writes the USRP map file used by the USRP serial number handling
	functions.

	Arguments:
		map - list of (which, serial) pairs, where which is an integer and
			serial is a string
	
	Returns:
		Nothing."""
	
	try:
		f = open(usrpmapfn, 'w')
	except:
		print "Could not open USRP map file %s for writing." % usrpmapfn
		sys.exit(1)
	
	for i in map:
		which = i[0]
		serial = i[1]
		f.write("%d %s\n" % (which,serial))
	
	f.close()

class sink_c(gr.hier_block2):
	"""Instantiates a usrp using serial number. The block will have one complex
	input if one side is used, and two complex inputs if both sides are used.

	Arguments:
		serial - the usrp serial number
		interp_rate - interpolation rate
		mode:
			1 - Use 1 TX on side A
			2 - Use 1 TX on side B
			3 - Use 2 TX on side A/B"""

	def __init__(self, serial=0, interp_rate=64, mode=1):

		# Determine number of channels used
		if mode == 1 or mode == 2:
			nchan = 1
		elif mode == 3:
			nchan = 2
		else:
			raise ValueError

		gr.hier_block2.__init__(self, "cusrp sink",
				gr.io_signature(nchan, nchan, gr.sizeof_gr_complex),
				gr.io_signature(0, 0, 0))
		
		# Get bus index from serial map
		which = serial_to_idx(serial)

		# Instantiate the usrp
		self.u = usrp.sink_c(which = which, interp_rate = interp_rate, nchan = nchan)

		# Verify that usrp has correct serial
		if serial != self.u.serial_number():
			print "cusrp: Requested usrp unit had wrong serial number, rescan "\
				"bus with cusrp.py scan"
			raise Exception

		# Set correct mux setting
		sides = []
		if mode == 1:
			if self.u.daughterboard_id(0) == 0x2b:
				self.u.set_mux(0x00000098)
			else:
				self.u.set_mux(0x00000098)
			sides.append(0)
		elif mode == 2:
			if self.u.daughterboard_id(0) == 0x2b:
				self.u.set_mux(0x00009800)
			else:
				self.u.set_mux(0x00009800)
			sides.append(1)
		else:
			if self.u.daughterboard_id(0) == 0x2b:
				self.u.set_mux(0x0000ba98)
			else:
				self.u.set_mux(0x0000ba98)
			sides.append(0)
			sides.append(1)
		
		# Verify that all daughterboards are LFTX or RFX2400
		for side in sides:
			dbid = self.u.daughterboard_id(side)
			if dbid == 0xe:
				self.db = 'lf'
			elif dbid == 0x2b:
				self.db = 'rfx2400'
			else:
				print "cusrp: Found unsupported daughterboard, id: %d" % dbid
				raise Exception

		# Set mid-range gain for all used subdevices
		subdevs = self.get_subdevs(mode)
		for dev in subdevs:
			if dev.dbid() == 0x2b:
				r = dev.gain_range()
				dev.set_gain((r[0]+r[1])/2)
			else:
				r = dev.gain_range()
				dev.set_gain((r[0]+r[1])/2)

		# Enable transmitters for all used RFX2400 subdevices
		for dev in subdevs:
			if dev.dbid() == 0x2b:
				dev.set_enable(True)

		# Connect block inputs to usrp
		if mode == 1 or mode == 2:
			self.connect(self, self.u)
		else:
			l = gr.interleave(gr.sizeof_gr_complex)
			self.connect((self, 0), (l, 0))
			self.connect((self, 1), (l, 1))
			self.connect(l, self.u)

	def tune(self, side, freq):
		"""Tune an LFTX daughterboard

	side:
		1 - A side
		2 - B side
	freq: the frequency in Hz"""

		subdevs = self.get_subdevs(side)
		for dev in subdevs:
			self.u.tune(dev.which(), dev, freq)

	def get_subdevs(self, mode):
		subdevspecs = []
		subdevs = []
		if mode == 1:
			subdevspecs.append((0,0))
		elif mode == 2:
			subdevspecs.append((1,0))
		elif mode == 3:
			subdevspecs.append((0,0))
			subdevspecs.append((1,0))
		else:
			raise ValueError

		for spec in subdevspecs:
			subdevs.append(usrp.selected_subdev(self.u, spec))

		return subdevs

class source_c(gr.hier_block2):
	"""Instantiates a usrp using serial number. The block will have one complex
	output if one side is used, and two complex outputs if both sides are used.

	Arguments:
		serial - the usrp serial number
		decim_rate - decimation rate
		mode:
			1 - Use 1 RX on side A
			2 - Use 1 RX on side B
			3 - Use 2 RX on side A/B"""

	def __init__(self, serial=0, decim_rate=64, mode=1):

		# Determine number of channels used
		if mode == 1 or mode == 2:
			nchan = 1
		elif mode == 3:
			nchan = 2
		else:
			raise ValueError

		gr.hier_block2.__init__(self, "cusrp source",
				gr.io_signature(0, 0, 0),
				gr.io_signature(nchan, nchan, gr.sizeof_gr_complex))
		
		# Get bus index from serial map
		which = serial_to_idx(serial)

		# Instantiate the usrp
		self.u = usrp.source_c(which = which, decim_rate = decim_rate, nchan = nchan)

		# Verify that usrp has correct serial
		if serial != self.u.serial_number():
			print "cusrp: Requested usrp unit had wrong serial number, rescan "\
				"bus with cusrp.py scan"
			raise Exception

		# Set correct mux setting
		sides = []
		if mode == 1:
			if self.u.daughterboard_id(0) == 0x27:
				self.u.set_mux(0x00000001)
			else:
				self.u.set_mux(0x0f0f0f10)
			sides.append(0)
		elif mode == 2:
			if self.u.daughterboard_id(0) == 0x27:
				self.u.set_mux(0x00000023)
			else:
				self.u.set_mux(0x0f0f0f32)
			sides.append(1)
		else:
			if self.u.daughterboard_id(0) == 0x27:
				self.u.set_mux(0x00002301)
			else:
				self.u.set_mux(0x0f0f3210)
			sides.append(0)
			sides.append(1)
		
		# Verify that all daughterboards are LFRX or RFX2400
		for side in sides:
			dbid = self.u.daughterboard_id(side)
			if dbid == 0xf:
				self.db = 'lf'
			elif dbid == 0x27:
				self.db = 'rfx2400'
			else:
				print "cusrp: Found unsupported daughterboard, id: %d" % dbid
				raise Exception

		# Set mid-range gain for all used subdevices
		subdevs = self.get_subdevs(mode)
		for dev in subdevs:
			if dev.dbid() == 0x27:
				r = dev.gain_range()
				dev.set_gain((r[0]+r[1])/2-15)
			else:
				r = dev.gain_range()
				dev.set_gain((r[0]+r[1])/2)

		# Select RX2 receive antenna for all used RFX2400 subdevices
		for dev in subdevs:
			if dev.dbid() == 0x27:
				dev.select_rx_antenna('RX2')

		# Connect usrp to block outputs
		if mode == 1 or mode == 2:
			self.connect(self.u, self)
		else:
			l = gr.deinterleave(gr.sizeof_gr_complex)
			self.connect(self.u, l)
			self.connect((l, 0), (self, 0))
			self.connect((l, 1), (self, 1))

	def tune(self, side, freq):
		"""Tune both the I and Q channel of a LFRX daughterboard

	side:
		1 - A side
		2 - B side
	freq: the frequency in Hz"""

		subdevs = self.get_subdevs(side)
		for dev in subdevs:
			self.u.tune(dev.which(), dev, freq)

	def get_subdevs(self, mode):
		subdevspecs = []
		subdevs = []
		if mode == 1:
			subdevspecs.append((0,0))
			if self.db == 'lf':
				subdevspecs.append((0,1))
		elif mode == 2:
			subdevspecs.append((1,0))
			if self.db == 'lf':
				subdevspecs.append((1,1))
		elif mode == 3:
			subdevspecs.append((0,0))
			subdevspecs.append((1,0))
			if self.db == 'lf':
				subdevspecs.append((0,1))
				subdevspecs.append((1,1))
		else:
			raise ValueError

		for spec in subdevspecs:
			subdevs.append(usrp.selected_subdev(self.u, spec))

		return subdevs

def scanusrp():
	list = []

	for which in range(0,10):
		serial = ''
		try:
			u = usrp.sink_c(which)
			if serial == '':
				serial = u.serial_number()
		except RuntimeError:
			pass

		try:
			u = usrp.source_c(which)
			if serial == '':
				serial = u.serial_number()
		except RuntimeError:
			pass

		if serial != '':
			print "Found USRP (%d) with serial number %s" % (which, u.serial_number())
			list.append((which, serial))

	try:
		write_serial_map(list)
		print "USRP serial map successfully saved"
	except:
		print "Error writing USRP serial map"

def usage():
	print "usage: cusrp.py scan"
	print "         scans the connected USRP units and builds a serial map file"
	sys.exit(0)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		usage()
	
	if sys.argv[1] == 'scan':
		scanusrp()
	else:
		usage()
