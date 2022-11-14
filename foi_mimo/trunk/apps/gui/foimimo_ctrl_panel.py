#
# Copyright 2011 FOI
# 
# This file is part of FOI-MIMO
# 
# FOI-MIMO is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# FOI-MIMO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FOI-MIMO; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from grc_gnuradio import wxgui as grc_wxgui
import wx

class ctrlPanel(grc_wxgui.Panel):
    def __init__(self, parent, global_ctrl, main_frame, orient=wx.HORIZONTAL):
        grc_wxgui.Panel.__init__ (self,parent,orient)
        
        # Init variables
        self.main_frame = main_frame
        self.global_ctrl = global_ctrl
        self.options_enabled = True
        self.options = self.global_ctrl.default_options()
        
        # Sizes and positions:
        box_width = 100
        box_size = (box_width,-1)
        
        x_boxline1 = 10
        x_textline1 = x_boxline1 + box_width + 10
        x_checkline1 = x_boxline1
        
        x_boxline2 = x_textline1 + 500
        x_textline2 = x_boxline2 + box_width + 10
        x_checkline2 = x_boxline2
        
        y_row = 30
        y_offs = 10
        y_text_offs = 6
        
        x_button = x_boxline1
        y_button = 500
        
        button_width = 200
        button_size = (button_width,-1)
        y_buttonrow = 45
        
        y_checkbox_offs = 8
        
        # All options:
        # Normal:
        row_nr=0
        self.text_options = wx.StaticText(parent=self, label="Normal settings:", pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_transmission_mode = wx.StaticText(parent=self, label="Transmission mode:", pos=(x_boxline1,y_offs+y_row*row_nr+3))
        self.radio_options_image_mode = wx.RadioButton(parent=self, label="Image", pos=(x_textline1+40,y_offs+y_row*row_nr), style=wx.RB_GROUP)
        self.radio_options_file_mode = wx.RadioButton(parent=self, label="File", pos=(x_textline1+120,y_offs+y_row*row_nr))
        self.radio_options_benchmark_mode = wx.RadioButton(parent=self, label="Benchmark", pos=(x_textline1+190,y_offs+y_row*row_nr))
        row_nr+=1        
        self.text_npixels = wx.StaticText(parent=self, label="- Number of pixels to transmit", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_npixels = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_input_filename = wx.StaticText(parent=self, label="- File containing input data", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_input_filename = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_output_filename_mimo = wx.StaticText(parent=self, label="- File to be written with output data in mimo mode", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_output_filename_mimo = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_output_filename_siso = wx.StaticText(parent=self, label="- File to be written with output data in siso mode", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_output_filename_siso = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_size = wx.StaticText(parent=self, label="- Packet size in bytes (incl 8 byte head & CRC32)", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_size = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_sample_rate = wx.StaticText(parent=self, label="- Sample rate in Sps (in throttle or USRP)", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_sample_rate = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_discontinuous = wx.StaticText(parent=self, label="- Enable discontinous transmission, burst of N packets [Default: continuous]", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_discontinuous = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_tx_amplitude = wx.StaticText(parent=self, label="- Transmitter digital amplitude (0 <= AMPL < 1)", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_tx_amplitude = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_modulation = wx.StaticText(parent=self, label="- Modulation type (bpsk or qpsk)", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_modulation = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr+=1
        self.radio_options_mimo = wx.RadioButton(parent=self, label="MIMO", pos=(x_checkline1,y_offs+y_row*row_nr+y_checkbox_offs), style=wx.RB_GROUP)
        self.radio_options_siso = wx.RadioButton(parent=self, label="SISO", pos=(x_checkline1+70,y_offs+y_row*row_nr+y_checkbox_offs))
        row_nr+=0.8
        self.checkbox_options_write_all = wx.CheckBox(parent=self, label="Write all received packets (not only those that are ok)", pos=(x_checkline1,y_offs+y_row*row_nr+y_checkbox_offs))   
        row_nr+=0.8
        self.checkbox_options_verbose = wx.CheckBox(parent=self, label="Verbose", pos=(x_checkline1,y_offs+y_row*row_nr+y_checkbox_offs))   
        row_nr+=1.2
        self.text_per_avg_nr = wx.StaticText(parent=self, label="- Nr of received packets to make packet error rate average over", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_per_avg_nr = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline1,y_offs+y_row*row_nr))
        row_nr += 1
        self.text_code_rate = wx.StaticText(parent=self, label="- Code rate", pos=(x_textline1,y_offs+y_row*row_nr+y_text_offs))
        self.coding_options = wx.Choice(self,-1,pos=(x_boxline1,y_offs+y_row*row_nr),choices=["3/4","2/3","1/3",""])
        
        #Expert:
        row_nr=0
        self.text_expert = wx.StaticText(parent=self, label="Expert settings:", pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_fft_length = wx.StaticText(parent=self, label="- Number of FFT bins", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_fft_length = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_occupied_tones = wx.StaticText(parent=self, label="- Number of occupied FFT bins", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_occupied_tones = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_cp_length = wx.StaticText(parent=self, label="- Number of bits in the cyclic prefix", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_cp_length = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.checkbox_options_log = wx.CheckBox(parent=self, label="Log all parts of flow graph to files (CAUTION: lots of data)", pos=(x_checkline2,y_offs+y_row*row_nr+y_checkbox_offs))
        
        #USRP:
        row_nr+=2
        self.text_usrp = wx.StaticText(parent=self, label="USRP settings:", pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_usrp_addr0 = wx.StaticText(parent=self, label="- USRP address, first unit", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_usrp_addr0 = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_usrp_addr1 = wx.StaticText(parent=self, label="- USRP address, second unit", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_usrp_addr1 = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_center_freq = wx.StaticText(parent=self, label="- USRP center frequency", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_center_freq = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_gain_rx = wx.StaticText(parent=self, label="- USRP rx gain in dB (0 <= GAIN < 31.5)", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_gain_rx = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))
        row_nr+=1
        self.text_gain_tx = wx.StaticText(parent=self, label="- USRP tx gain in dB (0 <= GAIN < 25.0)", pos=(x_textline2,y_offs+y_row*row_nr+y_text_offs))
        self.textbox_options_gain_tx = wx.TextCtrl(self,size=box_size,style=wx.TE_PROCESS_ENTER,pos=(x_boxline2,y_offs+y_row*row_nr))

        #Buttons:
        self.start_button = wx.Button(parent=self,label="Start (activates settings)",pos=(x_button,y_button),size=button_size)
        self.stop_button = wx.Button(parent=self,label="Stop",pos=(x_button+button_width+10,y_button),size=button_size)
        self.settings_button = wx.Button(parent=self,label="Activate Settings",pos=(x_button,y_button+35),size=button_size)
        self.default_button = wx.Button(parent=self,label="Go back to default settings!",pos=(x_button+button_width+10,y_button+35),size=button_size)

        # Bindings of buttons
        self.default_button.Bind(wx.EVT_BUTTON, self.on_default_settings)
        self.settings_button.Bind(wx.EVT_BUTTON, self.on_activate_settings)
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)

        self.radio_options_mimo.Bind(wx.EVT_RADIOBUTTON, self.on_mimo_options)
        self.radio_options_siso.Bind(wx.EVT_RADIOBUTTON, self.on_mimo_options)
        self.radio_options_file_mode.Bind(wx.EVT_RADIOBUTTON, self.on_transmission_mode)
        self.radio_options_image_mode.Bind(wx.EVT_RADIOBUTTON, self.on_transmission_mode)
        self.radio_options_benchmark_mode.Bind(wx.EVT_RADIOBUTTON, self.on_transmission_mode)
        
        self.Bind(wx.EVT_PAINT,self.on_paint)
        
        # Update default settings
        self.set_default_values()
        self.update_start_buttons()
        if self.global_ctrl.get_running():
            self.enable_settings()
        else:
            self.disable_settings()
        
    def set_default_values(self):
        self.options = self.global_ctrl.default_options()
        #Normal:
        self.textbox_options_size.SetValue(str(self.options.size))
        self.textbox_options_npixels.SetValue(str(self.options.npixels))
        self.textbox_options_input_filename.SetValue(self.options.input_filename)
        self.textbox_options_output_filename_mimo.SetValue(self.options.output_filename_mimo)
        self.textbox_options_output_filename_siso.SetValue(self.options.output_filename_siso)
        self.textbox_options_sample_rate.SetValue(str(self.options.sample_rate))
        self.textbox_options_discontinuous.SetValue(str(self.options.discontinuous))
        self.textbox_options_tx_amplitude.SetValue(str(self.options.tx_amplitude))
        self.textbox_options_modulation.SetValue(self.options.modulation)
        self.radio_options_mimo.SetValue(not self.options.siso)
        self.radio_options_siso.SetValue(self.options.siso)
        self.radio_options_benchmark_mode.SetValue(self.options.benchmark_mode)
        self.radio_options_file_mode.SetValue(self.options.file_mode)
        self.radio_options_image_mode.SetValue(self.options.image_mode)
        self.checkbox_options_verbose.SetValue(self.options.verbose)
        self.checkbox_options_write_all.SetValue(self.options.write_all)
        self.textbox_options_per_avg_nr.SetValue(str(self.options.per_avg_nr))
        self.coding_options.SetSelection(0)
        #Expert:
        self.textbox_options_fft_length.SetValue(str(self.options.fft_length))
        self.textbox_options_occupied_tones.SetValue(str(self.options.occupied_tones))
        self.textbox_options_cp_length.SetValue(str(self.options.cp_length))
        self.checkbox_options_log.SetValue(self.options.log)
        #USRP:
        self.textbox_options_usrp_addr0.SetValue(self.options.usrp_addr0)
        self.textbox_options_usrp_addr1.SetValue(self.options.usrp_addr1)
        self.textbox_options_center_freq.SetValue(str(self.options.center_freq))
        self.textbox_options_gain_rx.SetValue(str(self.options.gain_rx))
        self.textbox_options_gain_tx.SetValue(str(self.options.gain_tx))
        
        self.update_mode_options()

    def set_option_parameters(self):
        #Normal:
        self.options.size = float(self.textbox_options_size.GetValue())
        self.options.npixels = float(self.textbox_options_npixels.GetValue())
        self.options.input_filename = str(self.textbox_options_input_filename.GetValue())
        self.options.output_filename_mimo = str(self.textbox_options_output_filename_mimo.GetValue())
        self.options.output_filename_siso = str(self.textbox_options_output_filename_siso.GetValue())
        self.options.sample_rate = float(self.textbox_options_sample_rate.GetValue())
        self.options.discontinuous = int(self.textbox_options_discontinuous.GetValue())
        self.options.tx_amplitude = float(self.textbox_options_tx_amplitude.GetValue())
        self.options.modulation = self.textbox_options_modulation.GetValue()
        self.options.siso = not self.radio_options_mimo.GetValue()
        self.options.benchmark_mode = self.radio_options_benchmark_mode.GetValue()
        self.options.file_mode = self.radio_options_file_mode.GetValue()
        self.options.image_mode = self.radio_options_image_mode.GetValue()
        self.options.verbose = self.checkbox_options_verbose.GetValue()
        self.options.write_all = self.checkbox_options_write_all.GetValue()
        self.options.per_avg_nr = int(self.textbox_options_per_avg_nr.GetValue())
        self.options.code_rate = self.coding_options.GetStringSelection()
        #Expert:
        self.options.fft_length = int(self.textbox_options_fft_length.GetValue())
        self.options.occupied_tones = int(self.textbox_options_occupied_tones.GetValue())
        self.options.cp_length = int(self.textbox_options_cp_length.GetValue())
        self.options.log = self.checkbox_options_log.GetValue()
        #USRP:
        self.options.usrp_addr0 = str(self.textbox_options_usrp_addr0.GetValue())
        self.options.usrp_addr1 = str(self.textbox_options_usrp_addr1.GetValue())
        self.options.center_freq = float(self.textbox_options_center_freq.GetValue())
        self.options.gain_rx = float(self.textbox_options_gain_rx.GetValue())
        self.options.gain_tx = float(self.textbox_options_gain_tx.GetValue())
        
        self.global_ctrl.set_options(self.options)   
            
    def disable_settings(self):
        #Normal:
        self.textbox_options_size.Disable()
        self.textbox_options_npixels.Disable()
        self.textbox_options_input_filename.Disable()
        self.textbox_options_output_filename_mimo.Disable()
        self.textbox_options_output_filename_siso.Disable()
        self.textbox_options_sample_rate.Disable()
        self.textbox_options_discontinuous.Disable()
        self.textbox_options_tx_amplitude.Disable()
        self.textbox_options_modulation.Disable()
        self.radio_options_mimo.Disable()
        self.radio_options_siso.Disable()
        self.radio_options_benchmark_mode.Disable()
        self.radio_options_file_mode.Disable()
        self.radio_options_image_mode.Disable()
        self.checkbox_options_verbose.Disable()
        self.checkbox_options_write_all.Disable()
        self.textbox_options_per_avg_nr.Disable()
        self.coding_options.Disable()
        #Expert:
        self.textbox_options_fft_length.Disable()
        self.textbox_options_occupied_tones.Disable()
        self.textbox_options_cp_length.Disable()
        self.checkbox_options_log.Disable()
        #USRP:
        self.textbox_options_usrp_addr0.Disable()
        self.textbox_options_usrp_addr1.Disable()
        self.textbox_options_center_freq.Disable()
        self.textbox_options_gain_rx.Disable()
        self.textbox_options_gain_tx.Disable()
        #Buttons:
        self.default_button.Disable()
        self.settings_button.Disable()

        self.options_enabled = False

    def enable_settings(self):
        #Normal:
        self.textbox_options_size.Enable()
        self.textbox_options_sample_rate.Enable()
        self.textbox_options_discontinuous.Enable()
        #self.textbox_options_modulation.Enable() #only qpsk for now...
        self.radio_options_mimo.Enable()
        self.radio_options_siso.Enable()
        self.radio_options_benchmark_mode.Enable()
        self.radio_options_file_mode.Enable()
        self.radio_options_image_mode.Enable()
        self.checkbox_options_verbose.Enable()
        self.checkbox_options_write_all.Enable()
        self.textbox_options_per_avg_nr.Enable()
        self.coding_options.Enable()
        # tx
        if not self.options.usrp_rx:
            self.textbox_options_npixels.Enable()
            self.textbox_options_input_filename.Enable()
            self.textbox_options_tx_amplitude.Enable()
        # rx
        if not self.options.usrp_tx:
            self.textbox_options_output_filename_mimo.Enable()
            self.textbox_options_output_filename_siso.Enable()
        #Expert:
        self.textbox_options_fft_length.Enable()
        self.textbox_options_occupied_tones.Enable()
        self.textbox_options_cp_length.Enable()
        self.checkbox_options_log.Enable()
        #USRP:
        if self.options.usrp_rx or self.options.usrp_tx:
            self.textbox_options_usrp_addr0.Enable()
            self.textbox_options_usrp_addr1.Enable()
            self.textbox_options_center_freq.Enable()
        if self.options.usrp_rx:
            self.textbox_options_gain_rx.Enable()
        if self.options.usrp_tx:
            self.textbox_options_gain_tx.Enable()
              
        #Buttons:
        self.default_button.Enable()
        self.settings_button.Enable()
        #Adjust and disable either of the io mode options:
        self.update_mode_options()
                
        self.options_enabled = True
        
    def update_start_buttons(self):
        if self.global_ctrl.get_running():
            self.start_button.Disable()
            self.stop_button.Enable()
        else:
            self.start_button.Enable()
            self.stop_button.Disable()
            
    def update_mode_options(self):
        options = self.global_ctrl.get_options()
        self.textbox_options_npixels.Disable()
        self.textbox_options_input_filename.Disable()
        self.textbox_options_output_filename_mimo.Disable()
        self.textbox_options_output_filename_siso.Disable()
        if not self.global_ctrl.get_running():
            if self.radio_options_image_mode.GetValue():
                pass
            elif self.radio_options_file_mode.GetValue():
                if not options.usrp_rx: # benchmark or tx
                    self.textbox_options_input_filename.Enable()
                if not self.options.usrp_tx: # benchmark or rx
                    self.textbox_options_output_filename_mimo.Enable()
                    self.textbox_options_output_filename_siso.Enable()
            elif not options.usrp_rx:
                self.textbox_options_npixels.Enable() 
            
    def on_transmission_mode(self,e):
        self.update_mode_options()

    def on_default_settings(self,e):
        self.set_default_values()

    def on_activate_settings(self,e):
        self.set_option_parameters()

    def on_start(self,e):
        self.set_option_parameters()
        self.main_frame.start_tb()
        self.disable_settings()
        self.update_start_buttons()
        
    def on_stop(self,e):
        self.main_frame.stop_tb()
        self.enable_settings()
        self.update_start_buttons()
            
    def on_mimo_options(self,event):
        if self.radio_options_mimo.GetValue():
            self.global_ctrl.set_options_mimo(1)
        elif self.radio_options_siso.GetValue():
            self.global_ctrl.set_options_mimo(0)
        
    def on_paint(self,event):
        running_status = self.global_ctrl.get_running()
        if running_status and self.options_enabled:
            self.disable_settings()
        elif not self.options_enabled and not running_status:
            self.enable_settings()
        
        self.update_start_buttons()
        
        focus = self.global_ctrl.get_focus_view()
        if focus == "ctrl":
            pass
        else:
            self.global_ctrl.set_focus_view("ctrl")
            mimo_status = self.global_ctrl.get_options_mimo()
            if mimo_status:
                self.radio_options_mimo.SetValue(True)
                self.radio_options_siso.SetValue(False)
            else:
                self.radio_options_mimo.SetValue(False)
                self.radio_options_siso.SetValue(True)
