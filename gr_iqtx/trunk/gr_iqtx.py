#!/usr/bin/env python
##################################################
# Gnuradio Python Flow Graph
# Title: I/Q Baseband Transmitter+RX Monitor with Preemphasis
# Author: Marcus Leech, Science Radio Laboratories
# Description: A simple "baseband copier" for any I+Q input
# Generated: Mon Nov  7 18:57:23 2011
##################################################

from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio import window
from gnuradio.eng_option import eng_option
from gnuradio.gr import firdes
from gnuradio.wxgui import fftsink2
from gnuradio.wxgui import forms
from gnuradio.wxgui import scopesink2
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import SimpleXMLRPCServer
import math
import preemph
import threading
import time
import wx

class gr_iqtx(grc_wxgui.top_block_gui):

	def __init__(self, tlevel=-20, devid="type=usrp1", txant="TX/RX", pemph=0, ton=0, dbspec="A:0", frequency=225.0e6, scale=math.pow(2.0,15), rxant="RX2", frate=5, rfgain=25.0, a1label="Unused", a2label="Unused", a3label="Unused", a4label="Unused", dgain=0, toffset=100e3, afile="/dev/null", atitle="Analysis plugin data", port=8081, rxgain=15.0, srate=int(3.2e6), psiz=128):
		grc_wxgui.top_block_gui.__init__(self, title="I/Q Baseband Transmitter+RX Monitor with Preemphasis")
		_icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
		self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

		##################################################
		# Parameters
		##################################################
		self.tlevel = tlevel
		self.devid = devid
		self.txant = txant
		self.pemph = pemph
		self.ton = ton
		self.dbspec = dbspec
		self.frequency = frequency
		self.scale = scale
		self.rxant = rxant
		self.frate = frate
		self.rfgain = rfgain
		self.a1label = a1label
		self.a2label = a2label
		self.a3label = a3label
		self.a4label = a4label
		self.dgain = dgain
		self.toffset = toffset
		self.afile = afile
		self.atitle = atitle
		self.port = port
		self.rxgain = rxgain
		self.srate = srate
		self.psiz = psiz

		##################################################
		# Variables
		##################################################
		self.rxprobeout = rxprobeout = 0.0
		self.probeout = probeout = 0
		self.CLIPLIM = CLIPLIM = 0.9
		self.xton = xton = False if ton == 0 else True
		self.txmute = txmute = False
		self.tweak = tweak = 0
		self.ttlevel = ttlevel = tlevel
		self.rxlvl = rxlvl = 0 if rxprobeout <= 0.0 else math.log10(rxprobeout)*10.0
		self.prex = prex = False if pemph == 0 else True
		self.gain = gain = rfgain
		self.dgslider = dgslider = dgain
		self.clipval = clipval = "**YES**" if probeout > CLIPLIM else "NO"
		self.a4_value = a4_value = "--"
		self.a3_value = a3_value = "--"
		self.a2_value = a2_value = "--"
		self.a1_value = a1_value = "--"
		self.vgain = vgain = gain if txmute <= 0.0 else 0.0
		self.variable_static_text_0_0_1 = variable_static_text_0_0_1 = rxlvl
		self.variable_static_text_0_0_0 = variable_static_text_0_0_0 = clipval
		self.variable_static_text_0_0 = variable_static_text_0_0 = frequency+tweak
		self.variable_static_text_0 = variable_static_text_0 = srate
		self.tval = tval = math.sqrt(math.pow(10.0,ttlevel/10))
		self.test_tones = test_tones = xton
		self.srl = srl = "This Flowgraph brought to you by Science Radio Laboratories: http://www.science-radio-labs.com"
		self.rgain = rgain = rxgain
		self.pre_on = pre_on = prex
		self.offset = offset = toffset
		self.gval = gval = math.sqrt(math.pow(10.0,dgslider/10))
		self.fsize = fsize = psiz
		self.final_freq = final_freq = frequency+tweak
		self.a4_text = a4_text = a4_value
		self.a3_text = a3_text = a3_value
		self.a2_text = a2_text = a2_value
		self.a1_text_0 = a1_text_0 = atitle
		self.a1_text = a1_text = a1_value
		self.RXPWRLOG = RXPWRLOG = preemph.rxpower_log(rxprobeout)
		self.LOGCLIP = LOGCLIP = preemph.log_clipping(probeout,CLIPLIM,"clipping.log")

		##################################################
		# Blocks
		##################################################
		self.Main = self.Main = wx.Notebook(self.GetWin(), style=wx.NB_TOP)
		self.Main.AddPage(grc_wxgui.Panel(self.Main), "Controls+RX Mon")
		self.Main.AddPage(grc_wxgui.Panel(self.Main), "TX Mon")
		self.Main.AddPage(grc_wxgui.Panel(self.Main), "RX Power history")
		self.Main.AddPage(grc_wxgui.Panel(self.Main), "Analysis plugin data")
		self.Add(self.Main)
		self._txmute_check_box = forms.check_box(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.txmute,
			callback=self.set_txmute,
			label="TX Mute",
			true=1.0,
			false=0.0,
		)
		self.Main.GetPage(0).GridAdd(self._txmute_check_box, 1, 3, 1, 1)
		_tweak_sizer = wx.BoxSizer(wx.VERTICAL)
		self._tweak_text_box = forms.text_box(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_tweak_sizer,
			value=self.tweak,
			callback=self.set_tweak,
			label="Fine TX frequency",
			converter=forms.float_converter(),
			proportion=0,
		)
		self._tweak_slider = forms.slider(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_tweak_sizer,
			value=self.tweak,
			callback=self.set_tweak,
			minimum=-2500,
			maximum=2500,
			num_steps=250,
			style=wx.SL_HORIZONTAL,
			cast=float,
			proportion=1,
		)
		self.Main.GetPage(0).GridAdd(_tweak_sizer, 0, 0, 1, 1)
		self._test_tones_check_box = forms.check_box(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.test_tones,
			callback=self.set_test_tones,
			label="Test Tones On",
			true=1.0,
			false=0.0,
		)
		self.Main.GetPage(0).GridAdd(self._test_tones_check_box, 1, 1, 1, 1)
		self.rxprobe = gr.probe_signal_f()
		_rgain_sizer = wx.BoxSizer(wx.VERTICAL)
		self._rgain_text_box = forms.text_box(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_rgain_sizer,
			value=self.rgain,
			callback=self.set_rgain,
			label="RX RF Gain",
			converter=forms.float_converter(),
			proportion=0,
		)
		self._rgain_slider = forms.slider(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_rgain_sizer,
			value=self.rgain,
			callback=self.set_rgain,
			minimum=0,
			maximum=50.0,
			num_steps=50,
			style=wx.SL_HORIZONTAL,
			cast=float,
			proportion=1,
		)
		self.Main.GetPage(0).GridAdd(_rgain_sizer, 0, 4, 1, 1)
		self._pre_on_check_box = forms.check_box(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.pre_on,
			callback=self.set_pre_on,
			label="Preemphasis On",
			true=True,
			false=False,
		)
		self.Main.GetPage(0).GridAdd(self._pre_on_check_box, 0, 1, 1, 1)
		_offset_sizer = wx.BoxSizer(wx.VERTICAL)
		self._offset_text_box = forms.text_box(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_offset_sizer,
			value=self.offset,
			callback=self.set_offset,
			label="Test tone offset",
			converter=forms.float_converter(),
			proportion=0,
		)
		self._offset_slider = forms.slider(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_offset_sizer,
			value=self.offset,
			callback=self.set_offset,
			minimum=10e3,
			maximum=(srate/2)-1e3,
			num_steps=40,
			style=wx.SL_HORIZONTAL,
			cast=float,
			proportion=1,
		)
		self.Main.GetPage(0).GridAdd(_offset_sizer, 1, 2, 1, 1)
		self.myprobe = gr.probe_avg_mag_sqrd_c(-20, 1)
		self._fsize_chooser = forms.radio_buttons(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.fsize,
			callback=self.set_fsize,
			label="Preemph Filter Length",
			choices=[128, 256, 512],
			labels=[],
			style=wx.RA_HORIZONTAL,
		)
		self.Main.GetPage(0).GridAdd(self._fsize_chooser, 0, 2, 1, 1)
		self.xmlrpc_server_0 = SimpleXMLRPCServer.SimpleXMLRPCServer(("localhost", port), allow_none=True)
		self.xmlrpc_server_0.register_instance(self)
		threading.Thread(target=self.xmlrpc_server_0.serve_forever).start()
		self.wxgui_scopesink2_0 = scopesink2.scope_sink_f(
			self.Main.GetPage(2).GetWin(),
			title="RX Power History",
			sample_rate=4,
			v_scale=0,
			v_offset=0,
			t_scale=225,
			ac_couple=False,
			xy_mode=False,
			num_inputs=1,
			trig_mode=gr.gr_TRIG_MODE_STRIPCHART,
			y_axis_label="Relative RX Power (dB)",
		)
		self.Main.GetPage(2).Add(self.wxgui_scopesink2_0.win)
		self.wxgui_fftsink2_0_0 = fftsink2.fft_sink_c(
			self.Main.GetPage(1).GetWin(),
			baseband_freq=frequency,
			y_per_div=10,
			y_divs=10,
			ref_level=50,
			ref_scale=2.0,
			sample_rate=srate,
			fft_size=1024,
			fft_rate=frate,
			average=True,
			avg_alpha=0.375,
			title="TX Baseband Monitor",
			peak_hold=False,
			size=(650,400),
		)
		self.Main.GetPage(1).Add(self.wxgui_fftsink2_0_0.win)
		self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
			self.Main.GetPage(0).GetWin(),
			baseband_freq=frequency,
			y_per_div=10,
			y_divs=10,
			ref_level=50,
			ref_scale=2.0,
			sample_rate=srate,
			fft_size=1024,
			fft_rate=frate,
			average=True,
			avg_alpha=0.375,
			title="RX Monitor",
			peak_hold=False,
			size=(650,400),
		)
		self.Main.GetPage(0).Add(self.wxgui_fftsink2_0.win)
		self._variable_static_text_0_0_1_static_text = forms.static_text(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.variable_static_text_0_0_1,
			callback=self.set_variable_static_text_0_0_1,
			label="Rel RX Level (dB)",
			converter=forms.float_converter(formatter=preemph.dbf),
		)
		self.Main.GetPage(0).GridAdd(self._variable_static_text_0_0_1_static_text, 3, 3, 1, 1)
		self._variable_static_text_0_0_0_static_text = forms.static_text(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.variable_static_text_0_0_0,
			callback=self.set_variable_static_text_0_0_0,
			label="CLIPPED",
			converter=forms.str_converter(),
		)
		self.Main.GetPage(0).GridAdd(self._variable_static_text_0_0_0_static_text, 3, 5, 1, 1)
		self._variable_static_text_0_0_static_text = forms.static_text(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.variable_static_text_0_0,
			callback=self.set_variable_static_text_0_0,
			label="Channel Freq",
			converter=forms.float_converter(formatter=preemph.freqf),
		)
		self.Main.GetPage(0).GridAdd(self._variable_static_text_0_0_static_text, 3, 2, 1, 1)
		self._variable_static_text_0_static_text = forms.static_text(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.variable_static_text_0,
			callback=self.set_variable_static_text_0,
			label="Sample Rate",
			converter=forms.float_converter(),
		)
		self.Main.GetPage(0).GridAdd(self._variable_static_text_0_static_text, 3, 0, 1, 1)
		self.uhd_usrp_source_0 = uhd.usrp_source(
			device_addr=devid,
			io_type=uhd.io_type.COMPLEX_FLOAT32,
			num_channels=1,
		)
		self.uhd_usrp_source_0.set_subdev_spec(dbspec, 0)
		self.uhd_usrp_source_0.set_samp_rate(srate)
		self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(frequency,srate/1.9), 0)
		self.uhd_usrp_source_0.set_gain(rgain, 0)
		self.uhd_usrp_source_0.set_antenna(rxant, 0)
		self.uhd_usrp_sink_0 = uhd.usrp_sink(
			device_addr=devid,
			io_type=uhd.io_type.COMPLEX_FLOAT32,
			num_channels=1,
		)
		self.uhd_usrp_sink_0.set_subdev_spec(dbspec, 0)
		self.uhd_usrp_sink_0.set_samp_rate(srate)
		self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(frequency+tweak,srate/1.9), 0)
		self.uhd_usrp_sink_0.set_gain(vgain, 0)
		self.uhd_usrp_sink_0.set_antenna(txant, 0)
		_ttlevel_sizer = wx.BoxSizer(wx.VERTICAL)
		self._ttlevel_text_box = forms.text_box(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_ttlevel_sizer,
			value=self.ttlevel,
			callback=self.set_ttlevel,
			label="Test Tone Level",
			converter=forms.float_converter(),
			proportion=0,
		)
		self._ttlevel_slider = forms.slider(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_ttlevel_sizer,
			value=self.ttlevel,
			callback=self.set_ttlevel,
			minimum=-60,
			maximum=0,
			num_steps=50,
			style=wx.SL_HORIZONTAL,
			cast=float,
			proportion=1,
		)
		self.Main.GetPage(0).GridAdd(_ttlevel_sizer, 1, 0, 1, 1)
		self._srl_static_text = forms.static_text(
			parent=self.Main.GetPage(0).GetWin(),
			value=self.srl,
			callback=self.set_srl,
			label="Note",
			converter=forms.str_converter(),
		)
		self.Main.GetPage(0).GridAdd(self._srl_static_text, 2, 0, 1, 5)
		def _rxprobeout_probe():
			while True:
				val = self.rxprobe.level()
				try: self.set_rxprobeout(val)
				except AttributeError, e: pass
				time.sleep(1.0/(1))
		_rxprobeout_thread = threading.Thread(target=_rxprobeout_probe)
		_rxprobeout_thread.daemon = True
		_rxprobeout_thread.start()
		def _probeout_probe():
			while True:
				val = self.myprobe.level()
				try: self.set_probeout(val)
				except AttributeError, e: pass
				time.sleep(1.0/(10))
		_probeout_thread = threading.Thread(target=_probeout_probe)
		_probeout_thread.daemon = True
		_probeout_thread.start()
		self.gr_single_pole_iir_filter_xx_0 = gr.single_pole_iir_filter_ff(1.0/(srate/2.5), 1)
		self.gr_sig_source_x_0_0 = gr.sig_source_c(srate, gr.GR_COS_WAVE, -(srate/2)+offset, tval, 0)
		self.gr_sig_source_x_0 = gr.sig_source_c(srate, gr.GR_COS_WAVE, (srate/2)-offset, tval, 0)
		self.gr_nlog10_ff_0 = gr.nlog10_ff(10, 1, 0)
		self.gr_multiply_const_vxx_2 = gr.multiply_const_vcc((0 if txmute > 0 else 1.0, ))
		self.gr_multiply_const_vxx_1 = gr.multiply_const_vcc((test_tones, ))
		self.gr_multiply_const_vxx_0 = gr.multiply_const_vcc(((1.0/scale)*gval, ))
		self.gr_keep_one_in_n_0 = gr.keep_one_in_n(gr.sizeof_float*1, srate/4)
		self.gr_file_source_0 = gr.file_source(gr.sizeof_gr_complex*1, "/dev/stdin", False)
		self.gr_file_sink_0 = gr.file_sink(gr.sizeof_gr_complex*1, afile)
		self.gr_file_sink_0.set_unbuffered(True)
		self.gr_fft_filter_xxx_0 = gr.fft_filter_ccc(1, (preemph.compute_preemph(pre_on,int(fsize))))
		self.gr_complex_to_mag_squared_0 = gr.complex_to_mag_squared(1)
		self.gr_add_xx_1 = gr.add_vcc(1)
		self.gr_add_xx_0 = gr.add_vcc(1)
		_gain_sizer = wx.BoxSizer(wx.VERTICAL)
		self._gain_text_box = forms.text_box(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_gain_sizer,
			value=self.gain,
			callback=self.set_gain,
			label="TX RF Gain",
			converter=forms.float_converter(),
			proportion=0,
		)
		self._gain_slider = forms.slider(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_gain_sizer,
			value=self.gain,
			callback=self.set_gain,
			minimum=0,
			maximum=50.0,
			num_steps=50,
			style=wx.SL_HORIZONTAL,
			cast=float,
			proportion=1,
		)
		self.Main.GetPage(0).GridAdd(_gain_sizer, 0, 3, 1, 1)
		_dgslider_sizer = wx.BoxSizer(wx.VERTICAL)
		self._dgslider_text_box = forms.text_box(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_dgslider_sizer,
			value=self.dgslider,
			callback=self.set_dgslider,
			label="Digital Gain",
			converter=forms.float_converter(),
			proportion=0,
		)
		self._dgslider_slider = forms.slider(
			parent=self.Main.GetPage(0).GetWin(),
			sizer=_dgslider_sizer,
			value=self.dgslider,
			callback=self.set_dgslider,
			minimum=-30,
			maximum=30,
			num_steps=50,
			style=wx.SL_HORIZONTAL,
			cast=float,
			proportion=1,
		)
		self.Main.GetPage(0).GridAdd(_dgslider_sizer, 0, 5, 1, 1)
		self._a4_text_static_text = forms.static_text(
			parent=self.Main.GetPage(3).GetWin(),
			value=self.a4_text,
			callback=self.set_a4_text,
			label=a4label,
			converter=forms.str_converter(),
		)
		self.Main.GetPage(3).GridAdd(self._a4_text_static_text, 8, 1, 1, 1)
		self._a3_text_static_text = forms.static_text(
			parent=self.Main.GetPage(3).GetWin(),
			value=self.a3_text,
			callback=self.set_a3_text,
			label=a3label,
			converter=forms.str_converter(),
		)
		self.Main.GetPage(3).GridAdd(self._a3_text_static_text, 6, 1, 1, 1)
		self._a2_text_static_text = forms.static_text(
			parent=self.Main.GetPage(3).GetWin(),
			value=self.a2_text,
			callback=self.set_a2_text,
			label=a2label,
			converter=forms.str_converter(),
		)
		self.Main.GetPage(3).GridAdd(self._a2_text_static_text, 4, 1, 1, 1)
		self._a1_text_0_static_text = forms.static_text(
			parent=self.Main.GetPage(3).GetWin(),
			value=self.a1_text_0,
			callback=self.set_a1_text_0,
			label="Note",
			converter=forms.str_converter(),
		)
		self.Main.GetPage(3).GridAdd(self._a1_text_0_static_text, 0, 0, 1, 1)
		self._a1_text_static_text = forms.static_text(
			parent=self.Main.GetPage(3).GetWin(),
			value=self.a1_text,
			callback=self.set_a1_text,
			label=a1label,
			converter=forms.str_converter(),
		)
		self.Main.GetPage(3).GridAdd(self._a1_text_static_text, 2, 1, 1, 1)

		##################################################
		# Connections
		##################################################
		self.connect((self.gr_file_source_0, 0), (self.gr_multiply_const_vxx_0, 0))
		self.connect((self.gr_sig_source_x_0, 0), (self.gr_add_xx_0, 0))
		self.connect((self.gr_sig_source_x_0_0, 0), (self.gr_add_xx_0, 1))
		self.connect((self.gr_add_xx_0, 0), (self.gr_multiply_const_vxx_1, 0))
		self.connect((self.gr_multiply_const_vxx_1, 0), (self.gr_add_xx_1, 1))
		self.connect((self.uhd_usrp_source_0, 0), (self.wxgui_fftsink2_0, 0))
		self.connect((self.gr_fft_filter_xxx_0, 0), (self.gr_add_xx_1, 0))
		self.connect((self.uhd_usrp_source_0, 0), (self.gr_complex_to_mag_squared_0, 0))
		self.connect((self.gr_complex_to_mag_squared_0, 0), (self.gr_single_pole_iir_filter_xx_0, 0))
		self.connect((self.gr_single_pole_iir_filter_xx_0, 0), (self.gr_keep_one_in_n_0, 0))
		self.connect((self.gr_keep_one_in_n_0, 0), (self.rxprobe, 0))
		self.connect((self.gr_nlog10_ff_0, 0), (self.wxgui_scopesink2_0, 0))
		self.connect((self.gr_keep_one_in_n_0, 0), (self.gr_nlog10_ff_0, 0))
		self.connect((self.gr_add_xx_1, 0), (self.gr_multiply_const_vxx_2, 0))
		self.connect((self.gr_multiply_const_vxx_2, 0), (self.uhd_usrp_sink_0, 0))
		self.connect((self.gr_multiply_const_vxx_2, 0), (self.wxgui_fftsink2_0_0, 0))
		self.connect((self.gr_multiply_const_vxx_2, 0), (self.myprobe, 0))
		self.connect((self.uhd_usrp_source_0, 0), (self.gr_file_sink_0, 0))
		self.connect((self.gr_multiply_const_vxx_0, 0), (self.gr_fft_filter_xxx_0, 0))

	def get_tlevel(self):
		return self.tlevel

	def set_tlevel(self, tlevel):
		self.tlevel = tlevel
		self.set_ttlevel(self.tlevel)

	def get_devid(self):
		return self.devid

	def set_devid(self, devid):
		self.devid = devid

	def get_txant(self):
		return self.txant

	def set_txant(self, txant):
		self.txant = txant
		self.uhd_usrp_sink_0.set_antenna(self.txant, 0)

	def get_pemph(self):
		return self.pemph

	def set_pemph(self, pemph):
		self.pemph = pemph
		self.set_prex(False if self.pemph == 0 else True)

	def get_ton(self):
		return self.ton

	def set_ton(self, ton):
		self.ton = ton
		self.set_xton(False if self.ton == 0 else True)

	def get_dbspec(self):
		return self.dbspec

	def set_dbspec(self, dbspec):
		self.dbspec = dbspec

	def get_frequency(self):
		return self.frequency

	def set_frequency(self, frequency):
		self.frequency = frequency
		self.set_variable_static_text_0_0(self.frequency+self.tweak)
		self.wxgui_fftsink2_0_0.set_baseband_freq(self.frequency)
		self.wxgui_fftsink2_0.set_baseband_freq(self.frequency)
		self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.frequency+self.tweak,self.srate/1.9), 0)
		self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.frequency,self.srate/1.9), 0)
		self.set_final_freq(self.frequency+self.tweak)

	def get_scale(self):
		return self.scale

	def set_scale(self, scale):
		self.scale = scale
		self.gr_multiply_const_vxx_0.set_k(((1.0/self.scale)*self.gval, ))

	def get_rxant(self):
		return self.rxant

	def set_rxant(self, rxant):
		self.rxant = rxant
		self.uhd_usrp_source_0.set_antenna(self.rxant, 0)

	def get_frate(self):
		return self.frate

	def set_frate(self, frate):
		self.frate = frate

	def get_rfgain(self):
		return self.rfgain

	def set_rfgain(self, rfgain):
		self.rfgain = rfgain
		self.set_gain(self.rfgain)

	def get_a1label(self):
		return self.a1label

	def set_a1label(self, a1label):
		self.a1label = a1label

	def get_a2label(self):
		return self.a2label

	def set_a2label(self, a2label):
		self.a2label = a2label

	def get_a3label(self):
		return self.a3label

	def set_a3label(self, a3label):
		self.a3label = a3label

	def get_a4label(self):
		return self.a4label

	def set_a4label(self, a4label):
		self.a4label = a4label

	def get_dgain(self):
		return self.dgain

	def set_dgain(self, dgain):
		self.dgain = dgain
		self.set_dgslider(self.dgain)

	def get_toffset(self):
		return self.toffset

	def set_toffset(self, toffset):
		self.toffset = toffset
		self.set_offset(self.toffset)

	def get_afile(self):
		return self.afile

	def set_afile(self, afile):
		self.afile = afile

	def get_atitle(self):
		return self.atitle

	def set_atitle(self, atitle):
		self.atitle = atitle
		self.set_a1_text_0(self.atitle)

	def get_port(self):
		return self.port

	def set_port(self, port):
		self.port = port

	def get_rxgain(self):
		return self.rxgain

	def set_rxgain(self, rxgain):
		self.rxgain = rxgain
		self.set_rgain(self.rxgain)

	def get_srate(self):
		return self.srate

	def set_srate(self, srate):
		self.srate = srate
		self.gr_sig_source_x_0.set_sampling_freq(self.srate)
		self.gr_sig_source_x_0.set_frequency((self.srate/2)-self.offset)
		self.gr_sig_source_x_0_0.set_sampling_freq(self.srate)
		self.gr_sig_source_x_0_0.set_frequency(-(self.srate/2)+self.offset)
		self.wxgui_fftsink2_0_0.set_sample_rate(self.srate)
		self.wxgui_fftsink2_0.set_sample_rate(self.srate)
		self.uhd_usrp_sink_0.set_samp_rate(self.srate)
		self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.frequency+self.tweak,self.srate/1.9), 0)
		self.set_variable_static_text_0(self.srate)
		self.gr_keep_one_in_n_0.set_n(self.srate/4)
		self.gr_single_pole_iir_filter_xx_0.set_taps(1.0/(self.srate/2.5))
		self.uhd_usrp_source_0.set_samp_rate(self.srate)
		self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.frequency,self.srate/1.9), 0)

	def get_psiz(self):
		return self.psiz

	def set_psiz(self, psiz):
		self.psiz = psiz
		self.set_fsize(self.psiz)

	def get_rxprobeout(self):
		return self.rxprobeout

	def set_rxprobeout(self, rxprobeout):
		self.rxprobeout = rxprobeout
		self.set_RXPWRLOG(preemph.rxpower_log(self.rxprobeout))
		self.set_rxlvl(0 if self.rxprobeout <= 0.0 else math.log10(self.rxprobeout)*10.0)

	def get_probeout(self):
		return self.probeout

	def set_probeout(self, probeout):
		self.probeout = probeout
		self.set_LOGCLIP(preemph.log_clipping(self.probeout,self.CLIPLIM,"clipping.log"))
		self.set_clipval("**YES**" if self.probeout > self.CLIPLIM else "NO")

	def get_CLIPLIM(self):
		return self.CLIPLIM

	def set_CLIPLIM(self, CLIPLIM):
		self.CLIPLIM = CLIPLIM
		self.set_LOGCLIP(preemph.log_clipping(self.probeout,self.CLIPLIM,"clipping.log"))
		self.set_clipval("**YES**" if self.probeout > self.CLIPLIM else "NO")

	def get_xton(self):
		return self.xton

	def set_xton(self, xton):
		self.xton = xton
		self.set_test_tones(self.xton)

	def get_txmute(self):
		return self.txmute

	def set_txmute(self, txmute):
		self.txmute = txmute
		self._txmute_check_box.set_value(self.txmute)
		self.set_vgain(self.gain if self.txmute <= 0.0 else 0.0)
		self.gr_multiply_const_vxx_2.set_k((0 if self.txmute > 0 else 1.0, ))

	def get_tweak(self):
		return self.tweak

	def set_tweak(self, tweak):
		self.tweak = tweak
		self._tweak_slider.set_value(self.tweak)
		self._tweak_text_box.set_value(self.tweak)
		self.set_variable_static_text_0_0(self.frequency+self.tweak)
		self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.frequency+self.tweak,self.srate/1.9), 0)
		self.set_final_freq(self.frequency+self.tweak)

	def get_ttlevel(self):
		return self.ttlevel

	def set_ttlevel(self, ttlevel):
		self.ttlevel = ttlevel
		self.set_tval(math.sqrt(math.pow(10.0,self.ttlevel/10)))
		self._ttlevel_slider.set_value(self.ttlevel)
		self._ttlevel_text_box.set_value(self.ttlevel)

	def get_rxlvl(self):
		return self.rxlvl

	def set_rxlvl(self, rxlvl):
		self.rxlvl = rxlvl
		self.set_variable_static_text_0_0_1(self.rxlvl)

	def get_prex(self):
		return self.prex

	def set_prex(self, prex):
		self.prex = prex
		self.set_pre_on(self.prex)

	def get_gain(self):
		return self.gain

	def set_gain(self, gain):
		self.gain = gain
		self._gain_slider.set_value(self.gain)
		self._gain_text_box.set_value(self.gain)
		self.set_vgain(self.gain if self.txmute <= 0.0 else 0.0)

	def get_dgslider(self):
		return self.dgslider

	def set_dgslider(self, dgslider):
		self.dgslider = dgslider
		self._dgslider_slider.set_value(self.dgslider)
		self._dgslider_text_box.set_value(self.dgslider)
		self.set_gval(math.sqrt(math.pow(10.0,self.dgslider/10)))

	def get_clipval(self):
		return self.clipval

	def set_clipval(self, clipval):
		self.clipval = clipval
		self.set_variable_static_text_0_0_0(self.clipval)

	def get_a4_value(self):
		return self.a4_value

	def set_a4_value(self, a4_value):
		self.a4_value = a4_value
		self.set_a4_text(self.a4_value)

	def get_a3_value(self):
		return self.a3_value

	def set_a3_value(self, a3_value):
		self.a3_value = a3_value
		self.set_a3_text(self.a3_value)

	def get_a2_value(self):
		return self.a2_value

	def set_a2_value(self, a2_value):
		self.a2_value = a2_value
		self.set_a2_text(self.a2_value)

	def get_a1_value(self):
		return self.a1_value

	def set_a1_value(self, a1_value):
		self.a1_value = a1_value
		self.set_a1_text(self.a1_value)

	def get_vgain(self):
		return self.vgain

	def set_vgain(self, vgain):
		self.vgain = vgain
		self.uhd_usrp_sink_0.set_gain(self.vgain, 0)

	def get_variable_static_text_0_0_1(self):
		return self.variable_static_text_0_0_1

	def set_variable_static_text_0_0_1(self, variable_static_text_0_0_1):
		self.variable_static_text_0_0_1 = variable_static_text_0_0_1
		self._variable_static_text_0_0_1_static_text.set_value(self.variable_static_text_0_0_1)

	def get_variable_static_text_0_0_0(self):
		return self.variable_static_text_0_0_0

	def set_variable_static_text_0_0_0(self, variable_static_text_0_0_0):
		self.variable_static_text_0_0_0 = variable_static_text_0_0_0
		self._variable_static_text_0_0_0_static_text.set_value(self.variable_static_text_0_0_0)

	def get_variable_static_text_0_0(self):
		return self.variable_static_text_0_0

	def set_variable_static_text_0_0(self, variable_static_text_0_0):
		self.variable_static_text_0_0 = variable_static_text_0_0
		self._variable_static_text_0_0_static_text.set_value(self.variable_static_text_0_0)

	def get_variable_static_text_0(self):
		return self.variable_static_text_0

	def set_variable_static_text_0(self, variable_static_text_0):
		self.variable_static_text_0 = variable_static_text_0
		self._variable_static_text_0_static_text.set_value(self.variable_static_text_0)

	def get_tval(self):
		return self.tval

	def set_tval(self, tval):
		self.tval = tval
		self.gr_sig_source_x_0.set_amplitude(self.tval)
		self.gr_sig_source_x_0_0.set_amplitude(self.tval)

	def get_test_tones(self):
		return self.test_tones

	def set_test_tones(self, test_tones):
		self.test_tones = test_tones
		self.gr_multiply_const_vxx_1.set_k((self.test_tones, ))
		self._test_tones_check_box.set_value(self.test_tones)

	def get_srl(self):
		return self.srl

	def set_srl(self, srl):
		self.srl = srl
		self._srl_static_text.set_value(self.srl)

	def get_rgain(self):
		return self.rgain

	def set_rgain(self, rgain):
		self.rgain = rgain
		self._rgain_slider.set_value(self.rgain)
		self._rgain_text_box.set_value(self.rgain)
		self.uhd_usrp_source_0.set_gain(self.rgain, 0)

	def get_pre_on(self):
		return self.pre_on

	def set_pre_on(self, pre_on):
		self.pre_on = pre_on
		self._pre_on_check_box.set_value(self.pre_on)
		self.gr_fft_filter_xxx_0.set_taps((preemph.compute_preemph(self.pre_on,int(self.fsize))))

	def get_offset(self):
		return self.offset

	def set_offset(self, offset):
		self.offset = offset
		self.gr_sig_source_x_0.set_frequency((self.srate/2)-self.offset)
		self.gr_sig_source_x_0_0.set_frequency(-(self.srate/2)+self.offset)
		self._offset_slider.set_value(self.offset)
		self._offset_text_box.set_value(self.offset)

	def get_gval(self):
		return self.gval

	def set_gval(self, gval):
		self.gval = gval
		self.gr_multiply_const_vxx_0.set_k(((1.0/self.scale)*self.gval, ))

	def get_fsize(self):
		return self.fsize

	def set_fsize(self, fsize):
		self.fsize = fsize
		self._fsize_chooser.set_value(self.fsize)
		self.gr_fft_filter_xxx_0.set_taps((preemph.compute_preemph(self.pre_on,int(self.fsize))))

	def get_final_freq(self):
		return self.final_freq

	def set_final_freq(self, final_freq):
		self.final_freq = final_freq

	def get_a4_text(self):
		return self.a4_text

	def set_a4_text(self, a4_text):
		self.a4_text = a4_text
		self._a4_text_static_text.set_value(self.a4_text)

	def get_a3_text(self):
		return self.a3_text

	def set_a3_text(self, a3_text):
		self.a3_text = a3_text
		self._a3_text_static_text.set_value(self.a3_text)

	def get_a2_text(self):
		return self.a2_text

	def set_a2_text(self, a2_text):
		self.a2_text = a2_text
		self._a2_text_static_text.set_value(self.a2_text)

	def get_a1_text_0(self):
		return self.a1_text_0

	def set_a1_text_0(self, a1_text_0):
		self.a1_text_0 = a1_text_0
		self._a1_text_0_static_text.set_value(self.a1_text_0)

	def get_a1_text(self):
		return self.a1_text

	def set_a1_text(self, a1_text):
		self.a1_text = a1_text
		self._a1_text_static_text.set_value(self.a1_text)

	def get_RXPWRLOG(self):
		return self.RXPWRLOG

	def set_RXPWRLOG(self, RXPWRLOG):
		self.RXPWRLOG = RXPWRLOG

	def get_LOGCLIP(self):
		return self.LOGCLIP

	def set_LOGCLIP(self, LOGCLIP):
		self.LOGCLIP = LOGCLIP

if __name__ == '__main__':
	parser = OptionParser(option_class=eng_option, usage="%prog: [options]")
	parser.add_option("", "--tlevel", dest="tlevel", type="eng_float", default=eng_notation.num_to_str(-20),
		help="Set test-tone level (dB) [default=%default]")
	parser.add_option("", "--devid", dest="devid", type="string", default="type=usrp1",
		help="Set device id [default=%default]")
	parser.add_option("", "--txant", dest="txant", type="string", default="TX/RX",
		help="Set transmit antenna port [default=%default]")
	parser.add_option("", "--pemph", dest="pemph", type="intx", default=0,
		help="Set Preemphasis filter on/off [default=%default]")
	parser.add_option("", "--ton", dest="ton", type="intx", default=0,
		help="Set test-tone on/off [default=%default]")
	parser.add_option("", "--dbspec", dest="dbspec", type="string", default="A:0",
		help="Set daughterboard spec [default=%default]")
	parser.add_option("-f", "--frequency", dest="frequency", type="eng_float", default=eng_notation.num_to_str(225.0e6),
		help="Set frequency [default=%default]")
	parser.add_option("", "--scale", dest="scale", type="eng_float", default=eng_notation.num_to_str(math.pow(2.0,15)),
		help="Set scaling of input values [default=%default]")
	parser.add_option("", "--rxant", dest="rxant", type="string", default="RX2",
		help="Set receive antenna port [default=%default]")
	parser.add_option("", "--frate", dest="frate", type="intx", default=5,
		help="Set FFT Frame Rate [default=%default]")
	parser.add_option("-g", "--rfgain", dest="rfgain", type="eng_float", default=eng_notation.num_to_str(25.0),
		help="Set tx-side analog gain [default=%default]")
	parser.add_option("", "--a1label", dest="a1label", type="string", default="Unused",
		help="Set First analysis label [default=%default]")
	parser.add_option("", "--a2label", dest="a2label", type="string", default="Unused",
		help="Set Second analysis label [default=%default]")
	parser.add_option("", "--a3label", dest="a3label", type="string", default="Unused",
		help="Set Third analysis label [default=%default]")
	parser.add_option("", "--a4label", dest="a4label", type="string", default="Unused",
		help="Set Fourth analysis label [default=%default]")
	parser.add_option("", "--dgain", dest="dgain", type="eng_float", default=eng_notation.num_to_str(0),
		help="Set digital gain adjustment (dB) [default=%default]")
	parser.add_option("", "--toffset", dest="toffset", type="eng_float", default=eng_notation.num_to_str(100e3),
		help="Set test-tone offset [default=%default]")
	parser.add_option("", "--afile", dest="afile", type="string", default="/dev/null",
		help="Set Analysis output filename [default=%default]")
	parser.add_option("", "--atitle", dest="atitle", type="string", default="Analysis plugin data",
		help="Set Title of Analysis Tab [default=%default]")
	parser.add_option("", "--port", dest="port", type="intx", default=8081,
		help="Set XMLRPC Port [default=%default]")
	parser.add_option("", "--rxgain", dest="rxgain", type="eng_float", default=eng_notation.num_to_str(15.0),
		help="Set rx-side analog gain [default=%default]")
	parser.add_option("-r", "--srate", dest="srate", type="intx", default=int(3.2e6),
		help="Set hw sample rate [default=%default]")
	parser.add_option("", "--psiz", dest="psiz", type="intx", default=128,
		help="Set Preemphasis filter size [default=%default]")
	(options, args) = parser.parse_args()
	tb = gr_iqtx(tlevel=options.tlevel, devid=options.devid, txant=options.txant, pemph=options.pemph, ton=options.ton, dbspec=options.dbspec, frequency=options.frequency, scale=options.scale, rxant=options.rxant, frate=options.frate, rfgain=options.rfgain, a1label=options.a1label, a2label=options.a2label, a3label=options.a3label, a4label=options.a4label, dgain=options.dgain, toffset=options.toffset, afile=options.afile, atitle=options.atitle, port=options.port, rxgain=options.rxgain, srate=options.srate, psiz=options.psiz)
	tb.Run(True)

