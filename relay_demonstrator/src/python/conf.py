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

from openrd import pr
import os
import sys
import ConfigParser
from threading import Thread

class ConfError(Exception):
	pass

class conf:
	def __init__(self, conffile, unitid):
		self.unitid = unitid

		conf = ConfigParser.ConfigParser()
		conf.read(conffile)
		
		self.read_setup(conf)
		self.read_modes(conf)
		self.read_interface(conf)
		self.read_block_size(conf)
		self.read_coding(conf)
		self.read_framing(conf)
		self.read_modulation(conf)

		if self.link() == 'socket':
			self.read_socket(conf, unitid)
		elif self.link() == 'wired':
			self.read_wired(conf, unitid)
		elif self.link() == 'wireless':
			self.read_wireless(conf, unitid)
		else:
			raise ConfError("Invalid link mode %s" % self.link())

		if self.link() == 'socket' or self.link() == 'wired':
			self.read_channel(conf)
	
	def read_setup(self, conf):
		self.d_setup = conf.get('Setup', 'setup')
		self.d_link = conf.get('Setup', 'link')
		self.d_rate = int(conf.get('Setup', 'rate'))

	def read_modes(self, conf):
		source_data = conf.get('Modes', 'source_data')
		if source_data == 'zero':
			self.d_source_data_mode = pr.SRCMODE_ZERO
		elif source_data == 'counter':
			self.d_source_data_mode = pr.SRCMODE_COUNTER
		elif source_data == 'random':
			self.d_source_data_mode = pr.SRCMODE_RANDOM
		elif source_data == 'file':
			self.d_source_data_mode = pr.SRCMODE_FILE
		elif source_data == 'packet':
			self.d_source_data_mode = pr.SRCMODE_PACKET
		else:
			raise ConfError("Invalid source_data in section Modes")

		analysis = conf.get('Modes', 'analysis')
		if analysis == 'none':
			self.d_analysis_mode = pr.AMODE_NONE
		elif analysis == 'ber':
			self.d_analysis_mode = pr.AMODE_BER
		else:
			raise ConfError("Invalid analysis in section Modes")

		if self.d_setup == 'relay':
			relaying = conf.get('Modes', 'relaying')
			if relaying == 'none':
				self.d_relaying_mode = pr.RELAYING_NONE
			elif relaying == 'aaf':
				self.d_relaying_mode = pr.RELAYING_AAF
			elif relaying == 'daf':
				self.d_relaying_mode = pr.RELAYING_DAF
			else:
				raise ConfError("Invalid relaying in section Modes")

		channel = conf.get('Modes', 'channel')
		if channel == 'ideal':
			self.d_channel_mode = pr.CHMODE_IDEAL
		elif channel == 'awgn':
			self.d_channel_mode = pr.CHMODE_AWGN
		elif channel == 'rayleigh':
			self.d_channel_mode = pr.CHMODE_RAYLEIGH
		else:
			raise ConfError("Invalid channel in section Modes")

		if self.d_source_data_mode == pr.SRCMODE_PACKET:
			self.d_application = conf.get('Modes', 'application')

	def read_interface(self, conf):
		control = conf.get('Interface', 'control')
		if control == 'none':
			self.d_control = pr.CNTRLMODE_NONE
		elif control == 'console':
			self.d_control = pr.CNTRLMODE_CONSOLE
		elif control == 'remote':
			self.d_control = pr.CNTRLMODE_REMOTE
		else:
			raise ConfError("Invalid control in section Modes")

		if self.d_setup == 'relay':
			self.d_relayui = conf.get('Interface', 'relayui')

		self.d_destui = conf.get('Interface', 'destui')

	def read_block_size(self, conf):
		self.d_block_size = int(conf.get('Parameters', 'block_size'))

	def read_coding(self, conf):
		coding = conf.get('Parameters', 'coding')
		if coding == 'none':
			self.d_coding = pr.CODING_NONE
		elif coding == 'ldpc':
			self.d_coding = pr.CODING_LDPC
		else:
			raise ConfError("Invalid coding in section Parameters")

	def read_framing(self, conf):
		framing = conf.get('Parameters', 'framing')
		if framing == 'none':
			self.d_framing = pr.FRAMING_NONE
		elif framing == 'simple':
			self.d_framing = pr.FRAMING_SIMPLE
		elif framing == 'gsm':
			self.d_framing = pr.FRAMING_GSM
		else:
			raise ConfError("Invalid framing in section Parameters")

		self.d_frame_size = int(conf.get('Parameters', 'frame_size'))

	def read_modulation(self, conf):
		mod = conf.get('Parameters', 'modulation')
		if mod == 'bpsk':
			self.d_modulation = pr.MODULATION_BPSK
		elif mod == 'qpsk':
			self.d_modulation = pr.MODULATION_QPSK
		else:
			raise ConfError("Invalid modulation in section Parameters")

		self.d_symbol_length = int(conf.get('Parameters', 'symbol_length'))

	def read_socket(self, conf, unitid):
		sec = 'Socket_unit%d' % unitid
		
		numrx = int(conf.get(sec, 'numrx'))
		self.d_rxport = []
		for k in range(numrx):
			self.d_rxport.append(int(conf.get(sec, 'rxport_%d' % k)))

		numtx = int(conf.get(sec, 'numtx'))
		self.d_txhost = []
		self.d_txport = []
		for k in range(numtx):
			self.d_txhost.append(conf.get(sec, 'txhost_%d' % k))
			self.d_txport.append(int(conf.get(sec, 'txport_%d' % k)))

	def read_wired(self, conf, unitid):
		sec = 'Wired_unit%d' % unitid

		self.d_usrp_carrier = int(conf.get('Wired', 'carrier'))
		self.d_usrp_txpower = int(conf.get('Wired', 'txpower'))
		self.d_usrp_serial = conf.get(sec, 'serial')

	def read_wireless(self, conf, unitid):
		sec = 'Wireless_unit%d' % unitid

		self.d_usrp_carrier = int(conf.get('Wireless', 'carrier'))
		self.d_usrp_txpower = int(conf.get('Wireless', 'txpower'))
		self.d_usrp_serial = conf.get(sec, 'serial')

		if self.d_setup == 'relay':
			if self.d_relaying_mode != pr.RELAYING_NONE:
				self.d_usrp_relayoffset = int(conf.get('Wireless', 'relayoffset'))

	def read_channel(self, conf):
		self.d_sd_awgn = float(conf.get('Channel', 'sd_awgn'))
		self.d_sr_awgn = float(conf.get('Channel', 'sr_awgn'))
		self.d_rd_awgn = float(conf.get('Channel', 'rd_awgn'))

		if self.d_channel_mode == pr.CHMODE_RAYLEIGH:
			self.d_sd_speed = float(conf.get('Channel', 'sd_speed'))
			self.d_sr_speed = float(conf.get('Channel', 'sr_speed'))
			self.d_rd_speed = float(conf.get('Channel', 'rd_speed'))

	def link(self):
		return self.d_link

	def link_rate(self):
		return self.d_rate

	def source_data_mode(self):
		return self.d_source_data_mode

	def analysis_mode(self):
		return self.d_analysis_mode

	def relaying_mode(self):
		return self.d_relaying_mode

	def channel_mode(self):
		return self.d_channel_mode

	def control(self):
		return self.d_control

	def relayui(self):
		return self.d_relayui

	def destui(self):
		return self.d_destui

	def application(self):
		return self.d_application

	def block_size(self):
		return self.d_block_size

	def coding(self):
		return self.d_coding

	def framing(self):
		return self.d_framing

	def frame_size(self):
		return self.d_frame_size

	def modulation(self):
		return self.d_modulation

	def symbol_length(self):
		return self.d_symbol_length

	def socket_conn(self):
		return {'rxport' : self.d_rxport, 'txhost' : self.d_txhost, 'txport' : self.d_txport}

	def usrp_serial(self):
		return self.d_usrp_serial

	def usrp_txpower(self):
		return self.d_usrp_txpower

	def usrp_carrier(self):
		return self.d_usrp_carrier

	def usrp_relayoffset(self):
		return self.d_usrp_relayoffset

	def sd_awgn(self):
		return self.d_sd_awgn

	def sr_awgn(self):
		return self.d_sr_awgn

	def rd_awgn(self):
		return self.d_rd_awgn

	def sd_speed(self):
		return self.d_sd_speed

	def sr_speed(self):
		return self.d_sr_speed

	def rd_speed(self):
		return self.d_rd_speed

class control:
	def __init__(self):
		self.d_top = None
		self.d_sdchannel = None
		self.d_srchannel = None
		self.d_rdchannel = None
		self.d_tunesdrx = None
		self.d_tunesrrx = None
		self.d_tunerdrx = None
		self.d_relaycntrl = None
		self.d_srctx = None
		self.d_disp_cmd = False
		self.d_quit = False

	def set_disp_cmd(self, disp_cmd):
		self.d_disp_cmd = disp_cmd

	def set_top(self, top):
		self.d_top = top

	def set_sdchannel(self, sdchannel):
		self.d_sdchannel = sdchannel
	
	def set_srchannel(self, srchannel):
		self.d_srchannel = srchannel
	
	def set_rdchannel(self, rdchannel):
		self.d_rdchannel = rdchannel

	def set_tunesdrx(self, tunesdrx):
		self.d_tunesdrx = tunesdrx

	def set_tunesrrx(self, tunesrrx):
		self.d_tunesrrx = tunesrrx

	def set_tunerdrx(self, tunerdrx):
		self.d_tunerdrx = tunerdrx

	def set_relaycntrl(self, relaycntrl):
		self.d_relaycntrl = relaycntrl

	def set_srctx(self, srctx):
		self.d_srctx = srctx

	def set_sdawgn(self, awgn):
		if self.d_sdchannel is None:
			return

		try:
			self.d_sdchannel.set_noise(awgn)
			if self.d_disp_cmd:
				print "Set SD AWGN %f dBfs" % awgn
		except AttributeError:
			print "Operation not supported"

	def set_srawgn(self, awgn):
		if self.d_srchannel is None:
			return

		try:
			self.d_srchannel.set_noise(awgn)
			if self.d_disp_cmd:
				print "Set SR AWGN %f dBfs" % awgn
		except AttributeError:
			print "Operation not supported"

	def set_rdawgn(self, awgn):
		if self.d_rdchannel is None:
			return

		try:
			self.d_rdchannel.set_noise(awgn)
			if self.d_disp_cmd:
				print "Set RD AWGN %f dBfs" % awgn
		except AttributeError:
			print "Operation not supported"

	def set_sdspeed(self, speed):
		if self.d_sdchannel is None:
			return

		try:
			self.d_sdchannel.set_speed(speed)
			if self.d_disp_cmd:
				print "Set SD speed %f m/s" % speed
		except AttributeError:
			print "Operation not supported"

	def set_srspeed(self, speed):
		if self.d_srchannel is None:
			return

		try:
			self.d_srchannel.set_speed(speed)
			if self.d_disp_cmd:
				print "Set SR speed %f m/s" % speed
		except AttributeError:
			print "Operation not supported"

	def set_rdspeed(self, speed):
		if self.d_rdchannel is None:
			return

		try:
			self.d_rdchannel.set_speed(speed)
			if self.d_disp_cmd:
				print "Set RD speed %f dBfs" % speed
		except AttributeError:
			print "Operation not supported"

	def set_sdrxoff(self, offset):
		if self.d_tunesdrx is None:
			return

		self.d_tunesdrx(offset)

	def set_srrxoff(self, offset):
		if self.d_tunesrrx is None:
			return

		self.d_tunesrrx(offset)

	def set_rdrxoff(self, offset):
		if self.d_tunerdrx is None:
			return

		self.d_tunerdrx(offset)

	def set_relaymode(self, mode):
		if self.d_relaycntrl is None:
			return

		self.d_relaycntrl(mode)

	def set_srcpower(self, power):
		if self.d_srctx is None:
			return

		self.d_srctx.set_txpower(power)

	def quit(self):
		try:
			self.d_top.quit()
		except AttributeError:
			print "Attribute error while trying to quit. Forcing exit.."
			sys.exit(1)

		self.d_quit = True

	def execute(self, line):
		if line == '':
			self.quit()
			return

		t = line.split()

		cmd = t[0]

		if cmd == 'sdawgn':
			self.set_sdawgn(float(t[1]))
		elif cmd == 'srawgn':
			self.set_srawgn(float(t[1]))
		elif cmd == 'rdawgn':
			self.set_rdawgn(float(t[1]))
		elif cmd == 'sdspeed':
			self.set_sdspeed(float(t[1]))
		elif cmd == 'srspeed':
			self.set_srspeed(float(t[1]))
		elif cmd == 'rdspeed':
			self.set_rdspeed(float(t[1]))
		elif cmd == 'sdrxoff':
			self.set_sdrxoff(float(t[1]))
		elif cmd == 'srrxoff':
			self.set_srrxoff(float(t[1]))
		elif cmd == 'rdrxoff':
			self.set_rdrxoff(float(t[1]))
		elif cmd == 'relaymode':
			self.set_relaymode(int(t[1]))
		elif cmd == 'srcpower':
			self.set_srcpower(int(t[1]))
		elif cmd == 'quit':
			self.quit()
		else:
			print "Invalid command '%s'" % cmd

class control_none(control):
	def __init__(self):
		control.__init__(self)
	
	def start(self):
		pass

class control_console(control,Thread):
	def __init__(self):
		control.__init__(self)
		Thread.__init__(self)
	
	def run(self):
		print "console, use \"help\" for help"
		while self.d_quit == False:
			try:
				line = raw_input('> ')
			except EOFError:
				self.quit()

			if self.d_quit == False:
				if line.strip() == 'help':
					self.print_help()
				else:
					self.execute(line)

	def print_help(self):
		print "Available commands:"
		print "sdawgn awgn"
		print "srawgn awgn"
		print "rdawgn awgn"
		print "sdspeed speed"
		print "srspeed speed"
		print "rdspeed speed"
		print "sdrxoff offset"
		print "srrxoff offset"
		print "rdrxoff offset"
		print "relaymode mode"
		print "srcpower power"
		print "quit"

