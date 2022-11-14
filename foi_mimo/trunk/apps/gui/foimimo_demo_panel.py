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

import math
from grc_gnuradio import wxgui as grc_wxgui
import wx
import wx.lib.agw.speedmeter as agw_speedmeter
import wx.lib.agw.peakmeter as agw_peakmeter

from foimimo_events import *

class demoPanel(grc_wxgui.Panel):
    def __init__(self, parent, global_ctrl, main_frame, orient=wx.HORIZONTAL):
        grc_wxgui.Panel.__init__ (self,parent,orient)

        # Default settings:
        self.main_frame = main_frame
        self.global_ctrl = global_ctrl
        self.settings_enabled = True
        
        # Filenames:
        self.noImg_filename = "gui/no_image.bmp"
        self.rcvd_img_width =200
        self.rcvd_img_height=150
        # Load no image as default value:
        self.siso_filename = self.noImg_filename
        self.mimo_filename = self.noImg_filename
        
        siso_on_file = "gui/siso_on.png"
        siso_off_file = "gui/siso_off.png"
        mimo_on_file = "gui/mimo_on.png"
        mimo_off_file = "gui/mimo_off.png"
        mimo_siso_img_width = 350 #364
        mimo_siso_img_height = 163 #191
        
        # Boxes and sizers
        self.demoSizer = wx.GridBagSizer(vgap=10,hgap=60)
        self.Add(self.demoSizer)
        
        # Positions and sizes:
        self.x_pos_rcvd_img_siso = 60
        self.x_pos_rcvd_img_mimo = 60+self.rcvd_img_width+40
        self.y_pos_rcvd_img = 70
        
        self.x_pos_antenna_img=600
        self.y_pos_antenna_img=60
        
        self.x_pos_perf_peakmeter=340 #300
        self.y_pos_perf_peakmeter=280 #320
        
        self.min_size_perf_indicator = (200,200)
        self.min_size_rcvd_img = (200,200) # Half width and same height as mimo_siso_on/off
        self.min_size_mimo_option = (400,-1) # Same width as mimo_siso_on/off
        
        # Received images
        self.rcvd_img_mimo = []
        self.rcvd_img_siso = []

        self.text_siso_img = wx.StaticText(parent=self, label="Received SISO image:")
        self.text_mimo_img = wx.StaticText(parent=self, label="Received MIMO image:")

        self.demoSizer.Add(self.text_siso_img, pos=(1,1), span=(1,1),flag=wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_TOP)
        self.demoSizer.Add(self.text_mimo_img, pos=(1,2), span=(1,1),flag=wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_TOP)

        self.demoSizer.SetItemMinSize(self.text_siso_img,self.min_size_rcvd_img)
        self.demoSizer.SetItemMinSize(self.text_mimo_img,self.min_size_rcvd_img)

        # Choise SISO/MIMO and Start/Stop
        text_mimo_config = wx.StaticText(parent=self, label="Current antenna configuration:")
        text_choise_mimo = wx.StaticText(parent=self, label="Choose antenna configuration:")
        self.radio_choise_mimo = wx.RadioButton(parent=self, label="MIMO")
        self.radio_choise_siso = wx.RadioButton(parent=self, label="SISO")
        
        button_size=(100,-1)
        self.start_button = wx.Button(parent=self,label="Start",size=button_size)
        self.stop_button = wx.Button(parent=self,label="Stop",size=button_size)
        
        self.clear_MIMO_button = wx.Button(parent=self,label="Clear",size=button_size)
        self.clear_SISO_button = wx.Button(parent=self,label="Clear",size=button_size)

        self.siso_on = wx.Image(name=siso_on_file, type=wx.BITMAP_TYPE_PNG)
        self.siso_off = wx.Image(name=siso_off_file, type=wx.BITMAP_TYPE_PNG)
        self.mimo_on = wx.Image(name=mimo_on_file, type=wx.BITMAP_TYPE_PNG)
        self.mimo_off = wx.Image(name=mimo_off_file, type=wx.BITMAP_TYPE_PNG)
        
        self.siso_on = wx.BitmapFromImage(self.siso_on, depth=-1)
        self.siso_off = wx.BitmapFromImage(self.siso_off, depth=-1)
        self.mimo_on = wx.BitmapFromImage(self.mimo_on, depth=-1)
        self.mimo_off = wx.BitmapFromImage(self.mimo_off, depth=-1)
 
        self.set_mimo_option()
        self.update_status_img()
        
        self.demoSizer.Add(text_mimo_config,  pos=(1,3), span=(1,2),flag=wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_TOP)
        self.demoSizer.Add(text_choise_mimo,  pos=(3,3), span=(1,2),flag=wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_BOTTOM)
        self.demoSizer.Add(self.radio_choise_mimo, pos=(4,3), span=(1,1),flag=wx.ALIGN_RIGHT|wx.ALIGN_TOP)
        self.demoSizer.Add(self.radio_choise_siso, pos=(4,4), span=(1,1),flag=wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_TOP)
        self.demoSizer.Add(self.start_button, pos=(5,3), span=(1,1),flag=wx.ALIGN_CENTER)
        self.demoSizer.Add(self.stop_button,  pos=(5,4), span=(1,1),flag=wx.ALIGN_CENTER)
        self.demoSizer.Add(self.clear_SISO_button, pos=(2,1), span=(1,1),flag=wx.ALIGN_CENTER)
        self.demoSizer.Add(self.clear_MIMO_button, pos=(2,2), span=(1,1),flag=wx.ALIGN_CENTER)

        self._grid.SetItemMinSize(text_mimo_config,self.min_size_mimo_option)
        self._grid.SetItemMinSize(text_choise_mimo,self.min_size_mimo_option)
        self._grid.SetItemMinSize(self.start_button,button_size)
        self._grid.SetItemMinSize(self.stop_button,button_size)
        
        # Performance
        text_perf = wx.StaticText(parent=self, label="Performance:")
        text_perf_speed = wx.StaticText(parent=self, label="Speed")
        text_perf_rob = wx.StaticText(parent=self, label="Robustness")
        
        # Speed indicator
        self.createSpeedometer()
        
        self.demoSizer.Add(self.speedometer, pos=(5,1), span=(1,1),flag=wx.ALIGN_CENTER)
        self.demoSizer.SetItemMinSize(self.speedometer,self.min_size_perf_indicator)
                
        # Robustnes indicator
        self.update_robustness()
        
        self.demoSizer.Add(text_perf, pos=(4,1), span=(1,2),flag=wx.ALIGN_CENTER)
        self.demoSizer.Add(text_perf_speed,pos=(6,1), span=(1,1),flag=wx.ALIGN_CENTER)
        self.demoSizer.Add(text_perf_rob, pos=(6,2), span=(1,1),flag=wx.ALIGN_CENTER)
        
        # Timer to redraw window
        self.timer = wx.Timer(self)
        self.timer.Start(1000) #1000ms
        self.Bind(wx.EVT_TIMER,self.on_timer)
        
        #Set bindings to buttons
        self.Bind(wx.EVT_PAINT,self.on_paint)
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        self.clear_SISO_button.Bind(wx.EVT_BUTTON, self.on_clear_SISO)
        self.clear_MIMO_button.Bind(wx.EVT_BUTTON, self.on_clear_MIMO)
        self.update_start_buttons()
        
        self.radio_choise_mimo.Bind(wx.EVT_RADIOBUTTON, self.on_antenna_choise)
        self.radio_choise_siso.Bind(wx.EVT_RADIOBUTTON, self.on_antenna_choise)
        
        self.Bind(EVT_PKT,self.on_rcvd_data_event)

    def drawRcvdImg(self):

        # Init:
        x_pos_mimo = self.x_pos_rcvd_img_mimo
        x_pos_siso = self.x_pos_rcvd_img_siso
        y_pos = self.y_pos_rcvd_img
        width = self.rcvd_img_width
        height = self.rcvd_img_height
        dc = wx.PaintDC(self)
        pen_style = wx.SOLID
        
        # Get the data:
        colors_mimo = ''.join(self.rcvd_img_mimo)
        colors_siso = ''.join(self.rcvd_img_siso)
        
        # Draw MIMO image:
        dc.SetBrush(wx.Brush(wx.BLACK,wx.TRANSPARENT))
        dc.SetPen(wx.Pen(wx.BLACK,1,pen_style))
        dc.DrawRectangle(x_pos_mimo-1,y_pos-1,width+2,height+2)
        row = 0
        col = 0
        for j in range(0,len(colors_mimo)-2,3):
            color = wx.Color(ord(colors_mimo[j]),ord(colors_mimo[j+1]),ord(colors_mimo[j+2]))
            dc.SetPen(wx.Pen(color,1,pen_style))
            x = x_pos_mimo + col
            y = y_pos + row
            dc.DrawPoint(x,y)
            if col == width-1:
                if row == height-1: break #for loop end at end of img
                row +=1
                col=0
            else: col +=1

        # Draw SISO image:
        dc.SetBrush(wx.Brush(wx.BLACK,wx.TRANSPARENT))
        dc.SetPen(wx.Pen(wx.BLACK,1,pen_style))
        dc.DrawRectangle(x_pos_siso-1,y_pos-1,width+2,height+2)
        row = 0
        col = 0
        for j in range(0,len(colors_siso)-2,3):
            color = wx.Color(ord(colors_siso[j]),ord(colors_siso[j+1]),ord(colors_siso[j+2]))
            dc.SetPen(wx.Pen(color,1,pen_style))
            x = x_pos_siso + col
            y = y_pos + row
            dc.DrawPoint(x,y)
            if col == width-1:
                if row == height-1: break #for loop end at end of img
                row +=1
                col=0
            else: col +=1
                                
    def drawPeakmeter(self):

        # Init:
        x_pos = self.x_pos_perf_peakmeter
        y_pos = self.y_pos_perf_peakmeter
        value = self.robustnes_value1to5
        dc = wx.PaintDC(self)
        
        # Size position and color:
        nr_bars = 5
        bar_h_add = 35
        bar_w = 30
        bar_h = 35
        bar_x = x_pos
        bar_y = y_pos+nr_bars*bar_h_add
        colors_light = ['#ff0000','#ff8800','#ffff00','#88ff00','#00ff00']
        colors_dark = ['#bbbbbb','#bbbbbb','#bbbbbb','#bbbbbb','#bbbbbb']
        
        for i in range(1, nr_bars+1):
            if i > value:
                dc.SetBrush(wx.Brush(colors_dark[i-1])) #dark
                dc.DrawRectangle(bar_x, bar_y, bar_w, bar_h)
            else:
                dc.SetBrush(wx.Brush(colors_light[i-1])) #light
                dc.DrawRectangle(bar_x, bar_y, bar_w, bar_h)
            bar_h += bar_h_add
            bar_x += bar_w+10
            bar_y -= bar_h_add
        
    def createSpeedometer(self):
        extrastyle = agw_speedmeter.SM_DRAW_HAND | agw_speedmeter.SM_DRAW_SECONDARY_TICKS | agw_speedmeter.SM_DRAW_SECTORS | agw_speedmeter.SM_DRAW_MIDDLE_TEXT
        self.speedometer = agw_speedmeter.SpeedMeter(self, extrastyle=extrastyle)
        self.speedometer.SetAngleRange(-math.pi / 6, 7 * math.pi / 6)     
        intervals = range(0, 2201, 2200/10)
        self.speedometer.SetIntervals(intervals)
        colours = [wx.BLACK] * 10
        self.speedometer.SetIntervalColours(colours)
        ticks = [str(interval) for interval in intervals]
        self.speedometer.SetTicks(ticks)
        self.speedometer.SetTicksColour(wx.WHITE)
        self.speedometer.SetNumberOfSecondaryTicks(5)
        self.speedometer.SetTicksFont(wx.Font(7, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.speedometer.SetBackgroundColour(wx.BLACK)
        self.speedometer.SetMiddleText("kbps")
        self.speedometer.SetMiddleTextColour(wx.WHITE)
        self.speedometer.SetMiddleTextFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.speedometer.SetHandColour(wx.Colour(255, 50, 0))
        self.speedometer.DrawExternalArc(False)
        self.speedometer.SetSpeedBackground(wx.Colour(0xF0,0xEB,0xE2)) # Background color in this specific Ubuntu theme
        self.update_speed()
            
    def disable_settings(self):
        self.radio_choise_mimo.Disable()
        self.radio_choise_siso.Disable()
        self.settings_enabled = False
            
    def enable_settings(self):
        self.radio_choise_mimo.Enable()
        self.radio_choise_siso.Enable()
        self.settings_enabled = True

    def set_mimo_option(self):
        if self.radio_choise_mimo.GetValue():
            self.global_ctrl.set_options_mimo(1)
        elif self.radio_choise_siso.GetValue():
            self.global_ctrl.set_options_mimo(0)

    def update_speed(self):
        # Calculate speed in kbps
        options = self.global_ctrl.get_options()
        if options.code_rate == "":
            coderate = 1.0
        if options.code_rate == "3/4":
            coderate = 3.0/4.0
        if options.code_rate == "2/3":
            coderate = 2.0/3.0
        if options.code_rate == "1/3":
            coderate = 1.0/3.0
        overhead_pkt = 8    # 8 bytes per pkt (4 bytes header + 4 bytes CRC)
        pkt_size_tot = options.size*(1/coderate)
        if options.modulation == 'bpsk':
            modulation = 1
        else:               # We only have bpsk and qpsk right now...
            modulation = 2
        informationbits_per_symbol = options.occupied_tones*modulation
        samples_per_symbol = options.fft_length+options.cp_length
        seconds_per_symbol = float(samples_per_symbol)/float(options.sample_rate)
        informationbits_per_second = float(informationbits_per_symbol)/seconds_per_symbol
        overhead_ratio = float(overhead_pkt)/float(pkt_size_tot)
        speed_kbps = (informationbits_per_second*(1-overhead_ratio)*coderate)/1000
        
        # Update speedmeter
        if self.global_ctrl.get_running():
            self.speedometer.SetSpeedValue(speed_kbps)
        else:
            self.speedometer.SetSpeedValue(0)

    def update_robustness(self):
        options = self.global_ctrl.get_options()
        if self.global_ctrl.get_running():
            if self.global_ctrl.get_options_mimo():
                if options.code_rate == "":
                    self.robustnes_value1to5 = 2
                if options.code_rate == "3/4":
                    self.robustnes_value1to5 = 3
                if options.code_rate == "2/3":
                    self.robustnes_value1to5 = 4
                if options.code_rate == "1/3":
                    self.robustnes_value1to5 = 5
                self.drawPeakmeter()
            else:
                if options.code_rate == "":
                    self.robustnes_value1to5 = 1
                if options.code_rate == "3/4":
                    self.robustnes_value1to5 = 2
                if options.code_rate == "2/3":
                    self.robustnes_value1to5 = 3
                if options.code_rate == "1/3":
                    self.robustnes_value1to5 = 4
                self.drawPeakmeter()
        else:
            self.robustnes_value1to5 = 0
            self.drawPeakmeter()
        
    def update_start_buttons(self):
        if self.global_ctrl.get_running():
            self.start_button.Disable()
            self.stop_button.Enable()
        else:
            self.start_button.Enable()
            self.stop_button.Disable()
    
    def update_status_img(self):
        dc = wx.PaintDC(self)
        pos_x = self.x_pos_antenna_img
        pos_y = self.y_pos_antenna_img
        
        if not self.global_ctrl.get_options_mimo():
            if self.global_ctrl.get_running():
                status_img = self.siso_on
            else:                                                
                status_img = self.siso_off
        elif self.global_ctrl.get_options_mimo():            
            if self.global_ctrl.get_running():              
                status_img = self.mimo_on
            else:                                                
                status_img = self.mimo_off
        dc.DrawBitmap(status_img,pos_x,pos_y)     
        
    def update_rcvd_img(self):
        options = self.global_ctrl.get_options()      
    
    def on_start(self,e):
        self.set_mimo_option()
        self.main_frame.start_tb()
        self.update_speed()
        self.update_robustness()
        self.disable_settings()
        self.update_start_buttons()
        self.update_status_img()
        self.update_rcvd_img()

    def on_stop(self,e):
        self.main_frame.stop_tb()
        self.update_speed()
        self.update_robustness()
        self.enable_settings()
        self.update_start_buttons()
        self.update_status_img()
    
    def on_clear_SISO(self,e):
        self.global_ctrl.reset_statistics()
        self.rcvd_img_siso =[]
        
    def on_clear_MIMO(self,e):
        self.global_ctrl.reset_statistics()
        self.rcvd_img_mimo =[]
        
    def on_antenna_choise(self,e):
        self.set_mimo_option()
        self.update_status_img()
    
    def on_paint(self,event):
        self.drawPeakmeter()
        self.drawRcvdImg()

        running_status = self.global_ctrl.get_running()
        if running_status and self.settings_enabled:
            self.disable_settings()
        elif not self.settings_enabled and not running_status:
            self.enable_settings()
        
        self.update_speed()
        self.update_robustness()
        self.update_status_img()
        self.update_rcvd_img()
        self.update_start_buttons()
        
        focus = self.global_ctrl.get_focus_view()
        if focus == "demo":
            pass
        else:
            self.global_ctrl.set_focus_view("demo")
            mimo_status = self.global_ctrl.get_options_mimo()
            if mimo_status:
                self.radio_choise_mimo.SetValue(True)
                self.radio_choise_siso.SetValue(False)
            else:
                self.radio_choise_mimo.SetValue(False)
                self.radio_choise_siso.SetValue(True)

    def on_timer(self,e):
        self.Refresh() #clears area and calls evt_paint
        
    def on_rcvd_data_event(self,e):
        new_data = e.get_data()
        if self.global_ctrl.get_options_mimo():
            self.rcvd_img_mimo = self.rcvd_img_mimo + [new_data]
        else:
            self.rcvd_img_siso = self.rcvd_img_siso + [new_data]
        
    def reset_rcvd_img(self):
        if self.global_ctrl.get_options_mimo():
            self.rcvd_img_mimo = []
        else:
            self.rcvd_img_siso = []
