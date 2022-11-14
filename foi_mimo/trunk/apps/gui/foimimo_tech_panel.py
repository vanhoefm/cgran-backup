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

from foimimo import foi_constsink_gl

from gnuradio import gr
from gnuradio.wxgui import fftsink2
from grc_gnuradio import wxgui as grc_wxgui
import wx
import wx.lib.plot as plot

from foimimo_events import *
    
class techPanel(grc_wxgui.Panel):
    def __init__(self, parent, global_ctrl, main_frame, orient=wx.HORIZONTAL):
        grc_wxgui.Panel.__init__ (self,parent,orient)

        # Variables and stuff
        self.global_ctrl = global_ctrl
        self.options = self.global_ctrl.get_options()
        self.per_update_times  = 0

        # Create new notebook
        self.gui_notebook_tech = wx.Notebook(self.GetWin(), style=wx.NB_TOP)
        self.Add(self.gui_notebook_tech)

        # Create panels
        self.per_panel = grc_wxgui.Panel(self.gui_notebook_tech, wx.HORIZONTAL)
        self.fft_panel = grc_wxgui.Panel(self.gui_notebook_tech, wx.HORIZONTAL)
        self.constellation_panel = grc_wxgui.Panel(self.gui_notebook_tech, wx.HORIZONTAL)

        # Add pages to the notebook     
        self.gui_notebook_tech.AddPage(self.per_panel, "Packet error rate")
        self.gui_notebook_tech.AddPage(self.fft_panel, "Frequency spectrum")
        self.gui_notebook_tech.AddPage(self.constellation_panel, "Constellation plot")
                
        # Create sinks for the fft and constellation plots
        # to be connected in the top block flowgraph later
        self.make_sinks()
        self.add_sinks()
        
        # Plot PER
        self.nr_per_values = 200
        self.plotter = plot.PlotCanvas(self.per_panel)
        self.plotter.SetInitialSize(size=(1000,500))
        self.per_data = [0]
        self.draw_per_line()

        # Bind events
        self.Bind(EVT_PER,self.on_per_event)
        self.Bind(wx.EVT_PAINT,self.on_paint)

    def on_per_event(self,e):
        per_val = e.get_per_val()
        self.update_per(per_val)

    def on_paint(self,e):
        self.draw_per_line()
            
    def reset_per_data(self):
        self.per_data = []
        print "PER RESET"
            
    def update_per(self,value):
        self.per_update_times +=1 
        per_value = value
        self.per_data.append(per_value)
        if len(self.per_data) > self.nr_per_values: self.per_data.pop(0)
        # Updated per, let's paint every 60 times
        if self.per_update_times >= 60:
            self.per_update_times = 0
            event = wx.PyCommandEvent(wx.EVT_PAINT.typeId, self.GetId())
            wx.PostEvent(self.GetEventHandler(),event)
    
    def draw_per_line(self):
        line_data = []
        for i in range(1,len(self.per_data)+1):
            line_data.append((i,self.per_data[i-1])) 
        line = plot.PolyLine(line_data,colour='blue',width=1)
        gc = plot.PlotGraphics([line],'Average packet error rate over time','time (~ received packets)','packet error rate')
        self.plotter.Draw(gc, xAxis=(1,self.nr_per_values), yAxis=(0,100)) 

    def add_sinks(self):
        # FFT sinks
        self.fft_panel.GridAdd(self.wxgui_fftsink_chanfilt0.win,1,1)
        self.fft_panel.GridAdd(self.wxgui_fftsink_chanfilt1.win,1,2)
        self.fft_panel.Layout()
        # Constellation sinks
        self.constellation_panel.GridAdd(self.wxgui_constellationsink_fft0.win,1,1)
        self.constellation_panel.GridAdd(self.wxgui_constellationsink_fft1.win,1,2)
        self.constellation_panel.GridAdd(self.wxgui_constellationsink_frameacq.win,2,1)
        self.constellation_panel.GridAdd(self.wxgui_constellationsink_framesink.win,2,2)
        self.constellation_panel.Layout()
            
    def delete_sinks(self):    
        # Clear FFT sinks
        self.fft_panel._grid.Clear(deleteWindows=False)
        # Clear constellation sinks
        self.constellation_panel._grid.Clear(deleteWindows=False)
                                                   
    def make_sinks(self):
        options = self.global_ctrl.get_options()

        # Sink parameters    
        gui_fftsize = options.fft_length
        gui_fc = 0                        # basband..
        gui_rate = options.sample_rate
        gui_nchan = 1                     # viewing one channel
        # Window size:
        gui_fft_winsize=(400,400)
        gui_const_winsize=(250,250)
        # Constellation sizes:
        constsize_fft = options.fft_length
        constsize_frame_acq = options.occupied_tones
        constsize_frame_sink = options.occupied_tones

        if self.global_ctrl.get_options_mimo():
            max_xy_value_fft=15
            max_xy_value_frame_acq=150
            max_xy_value_frame_sink=150
        else:
            max_xy_value_fft=8
            max_xy_value_frame_acq=2
            max_xy_value_frame_sink=2

        # Create sinks
        self.wxgui_fftsink_chanfilt0 = fftsink2.fft_sink_c(
            self.fft_panel.GetWin(),
            baseband_freq=gui_fc,
            y_per_div=10,
            y_divs=10,
            ref_level=50,
            ref_scale=2.0,
            sample_rate=gui_rate,
            fft_size=gui_fftsize,
            fft_rate=30,
            average=False,
            avg_alpha=None,
            title="Output channel filter 0",
            peak_hold=False,
            size=gui_fft_winsize,
        )
        self.wxgui_fftsink_chanfilt1 = fftsink2.fft_sink_c(
            self.fft_panel.GetWin(),
            baseband_freq=gui_fc,
            y_per_div=10,
            y_divs=10,
            ref_level=50,
            ref_scale=2.0,
            sample_rate=gui_rate,
            fft_size=gui_fftsize,
            fft_rate=30,
            average=False,
            avg_alpha=None,
            title="Output channel filter 1",
            peak_hold=False,
            size=gui_fft_winsize,
        )
        
        self.wxgui_constellationsink_fft0 = foi_constsink_gl.const_sink_c(
            self.constellation_panel.GetWin(),
            title="Output FFT 0",
            sample_rate=gui_rate,
            frame_rate=5,
            const_size=constsize_fft,
            symbol_rate=gui_rate/4.,
            size=gui_const_winsize,
            max_xy_value=max_xy_value_fft,
        )
        self.wxgui_constellationsink_fft1 = foi_constsink_gl.const_sink_c(
            self.constellation_panel.GetWin(),
            title="Output FFT 1",
            sample_rate=gui_rate,
            frame_rate=5,
            const_size=constsize_fft,
            symbol_rate=gui_rate/4.,
            size=gui_const_winsize,
            max_xy_value=max_xy_value_fft,
        )
        self.wxgui_constellationsink_frameacq = foi_constsink_gl.const_sink_c(
            self.constellation_panel.GetWin(),
            title="Output frame aqc.",
            sample_rate=gui_rate,
            frame_rate=5,
            const_size=constsize_frame_acq,
            symbol_rate=gui_rate/4.,
            size=gui_const_winsize,
            max_xy_value=max_xy_value_frame_acq,
        )
        self.wxgui_constellationsink_framesink = foi_constsink_gl.const_sink_c(
            self.constellation_panel.GetWin(),
            title="Output frame sink",
            sample_rate=gui_rate,
            frame_rate=5,
            const_size=constsize_frame_sink,
            symbol_rate=gui_rate/4.,
            size=gui_const_winsize,
            max_xy_value=max_xy_value_frame_sink,
        )        
        
        self.all_sinks = [(self.wxgui_fftsink_chanfilt0, "fft_channel_filter_0"),
                          (self.wxgui_fftsink_chanfilt1, "fft_channel_filter_1"),
                          (self.wxgui_constellationsink_fft0, "constellation_fft_0"),
                          (self.wxgui_constellationsink_fft1, "constellation_fft_1"),
                          (self.wxgui_constellationsink_frameacq, "constellation_frame_acq"),
                          (self.wxgui_constellationsink_framesink, "constellation_frame_sink")]

        return self.all_sinks            
