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

class txPanel(grc_wxgui.Panel):
    def __init__(self, parent, global_ctrl, main_frame, orient=wx.HORIZONTAL):
        grc_wxgui.Panel.__init__ (self,parent,orient)

        # Default settings:
        self.main_frame = main_frame
        self.global_ctrl = global_ctrl
        self.settings_enabled = True
        
        # Filenames and img sizes:
        imageFile = "gui/demo_image.gif"
        self.img_width =810
        self.img_height=610

        self.img_xpos = 5
        self.img_ypos = 5
        self.imagePos = wx.Point(self.img_xpos,self.img_ypos)
                
        self.recWidth = 200
        self.recHeight = 150
        self.rect_x = 850
        self.rect_y = 80
        
        self.antenna_checkbox_xpos = 850
        self.antenna_checkbox_ypos = 300
        
        self.start_buttons_xpos = 850
        self.start_buttons_ypos = 350
        
        # Big image
        try:
            self.big_image = wx.Image(imageFile, wx.BITMAP_TYPE_ANY)
            self.big_image.Resize((self.img_width,self.img_height),self.imagePos,0,0,0)             
            self.big_bitmap = self.big_image.ConvertToBitmap()
            self.big_static_bitmap = wx.StaticBitmap(self,-1,self.big_bitmap,self.imagePos)
            
            self.big_static_bitmap.Bind(wx.EVT_LEFT_DOWN, self.buttonPressed)
            self.big_static_bitmap.Bind(wx.EVT_MOTION, self.mouseMotion)
            
        except IOError:
            print "File not found"
            raise SystemExit
        
        # Small image, init black:
        self.bild = list()
        self.flat_send_data=''
        for v in range(self.recHeight):
            for h in range(self.recWidth):
                self.bild.append((0,0,0))
                self.flat_send_data = self.flat_send_data + chr(0 & 0xff) + chr(0 & 0xff) + chr(0 & 0xff)
        self.main_frame.set_data_to_send(self.flat_send_data)

        # Choise SISO/MIMO and Start/Stop
        self.radio_choise_mimo = wx.RadioButton(parent=self, label="MIMO",pos=(self.antenna_checkbox_xpos,self.antenna_checkbox_ypos))
        self.radio_choise_siso = wx.RadioButton(parent=self, label="SISO",pos=(self.antenna_checkbox_xpos+80,self.antenna_checkbox_ypos))
        
        button_size=(100,-1)
        self.start_button = wx.Button(parent=self,label="Start",size=button_size,pos=(self.start_buttons_xpos,self.start_buttons_ypos))
        self.stop_button = wx.Button(parent=self,label="Stop",size=button_size,pos=(self.start_buttons_xpos+110,self.start_buttons_ypos))
        self.set_mimo_option()
        
        # Set bindings to buttons
        self.Bind(wx.EVT_PAINT,self.on_paint)
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        self.update_start_buttons()
        
        self.radio_choise_mimo.Bind(wx.EVT_RADIOBUTTON, self.on_antenna_choise)
        self.radio_choise_siso.Bind(wx.EVT_RADIOBUTTON, self.on_antenna_choise)


    def mouseMotion(self,e):
        # Draw rectangle on mouse position
        p = e.GetPosition()
        
        dc = wx.PaintDC(self)
        dc.SetBrush(wx.Brush((0,0,0),wx.TRANSPARENT))
        dc.DrawBitmap(self.big_bitmap,self.imagePos.x,self.imagePos.y)
        
        if (p.x > self.imagePos.x and p.y > self.imagePos.y and
             p.x +self.recWidth < self.big_image.GetWidth() and 
             p.y + self.recHeight <self.big_image.GetHeight()):
            dc.DrawRectangle(p.x + self.imagePos.x, p.y + self.imagePos.y, 
                             self.recWidth,self.recHeight)
        
    def buttonPressed(self,e):
        # Draw small image
        p = e.GetPosition()
        self.bild = list()
        
        for v in range(self.recHeight):
            for h in range(self.recWidth):
                self.bild.append((self.big_image.GetRed(p.x+h,p.y+v),
                                  self.big_image.GetGreen(p.x+h,p.y+v),
                                  self.big_image.GetBlue(p.x+h,p.y+v)))
        dc = wx.PaintDC(self)
        for v in range(self.recHeight):
            for h in range(self.recWidth):
                dc.SetPen(wx.Pen(self.bild[v*self.recWidth+h]))
                dc.DrawPoint(self.rect_x+h,self.rect_y+v)

        self.flat_send_data = ""
        flat_bild = sum(self.bild,())	# flat_bild=(r,g,b,r,g,b,...) because bild=((r,g,b),(r,g,b),...)
        for c in flat_bild:		# flat_send_data=['r','g','b','r','g','b',...]
            self.flat_send_data = self.flat_send_data + chr(c & 0xff)
            
        self.main_frame.set_data_to_send(self.flat_send_data)
                    
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
        
    def update_start_buttons(self):
        if self.global_ctrl.get_running():
            self.start_button.Disable()
            self.stop_button.Enable()
        else:
            self.start_button.Enable()
            self.stop_button.Disable()  
    
    def on_start(self,e):
        self.set_mimo_option()
        #self.main_frame.set_data_to_send(self.flat_send_data) #<- now done at every mouseclick
        self.main_frame.start_tb()
        self.disable_settings()
        self.update_start_buttons()

    def on_stop(self,e):
        self.main_frame.stop_tb()
        self.enable_settings()
        self.update_start_buttons()
        
    def on_antenna_choise(self,e):
        self.set_mimo_option()
    
    def on_paint(self,event):
        running_status = self.global_ctrl.get_running()
        if running_status and self.settings_enabled:
            self.disable_settings()
        elif not self.settings_enabled and not running_status:
            self.enable_settings()

        self.update_start_buttons()
        
        focus = self.global_ctrl.get_focus_view()
        if focus == "tx":
            pass
        else:
            self.global_ctrl.set_focus_view("tx")
            mimo_status = self.global_ctrl.get_options_mimo()
            if mimo_status:
                self.radio_choise_mimo.SetValue(True)
                self.radio_choise_siso.SetValue(False)
            else:
                self.radio_choise_mimo.SetValue(False)
                self.radio_choise_siso.SetValue(True)
                
        # Draw empty rectangle box around small image
        dc = wx.PaintDC(self)
        dc.SetBrush(wx.Brush((0,0,0),wx.TRANSPARENT))
        dc.SetPen(wx.Pen(wx.BLACK,1,wx.SOLID))
        dc.DrawRectangle(self.rect_x-1,self.rect_y-1,self.recWidth+2,self.recHeight+2)
        
        # Draw empty rectangle box around big image
        dc = wx.PaintDC(self)
        dc.SetBrush(wx.Brush((0,0,0),wx.TRANSPARENT))
        dc.SetPen(wx.Pen(wx.BLACK,1,wx.SOLID))
        dc.DrawRectangle(self.img_xpos-1,self.img_ypos-1,self.img_width+2,self.img_height+2)

        # Draw small image
        dc = wx.PaintDC(self)
        for v in range(self.recHeight):
            for h in range(self.recWidth):
                dc.SetPen(wx.Pen(self.bild[v*self.recWidth+h]))
                dc.DrawPoint(self.rect_x+h,self.rect_y+v)
