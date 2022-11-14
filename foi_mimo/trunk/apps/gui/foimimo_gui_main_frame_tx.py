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

import gnuradio.gr.gr_threading as _threading

from grc_gnuradio import wxgui as grc_wxgui
import wx

# import local gui classes
from foimimo_tx_panel import txPanel
from foimimo_ctrl_panel import ctrlPanel

class GUI_thread(_threading.Thread):
    def __init__(self, app):
        _threading.Thread.__init__(self)
        self.setDaemon(1)
        self.app = app
        self.start()

    def run(self):
        self.app.MainLoop()

class main_frame(wx.Frame):
    def __init__(self, global_ctrl, app):
        wx.Frame.__init__(self, None, -1, "Benchmark GUI Notebook")
        
        # Global variables init
        self.global_ctrl = global_ctrl
        
        # Variables needed when starting top block
        self.options = self.global_ctrl.get_options()
        
        # Other variables to init
        self.data_to_send = []
                            
        # Menus and bars
        nstatus = 2
        self.CreateStatusBar (nstatus)
        self.create_menus()
        self.Bind (wx.EVT_CLOSE, self.on_close)
        
        # Window layout and size
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Panels and notebooks
        self.main_panel = grc_wxgui.Panel(self,wx.VERTICAL)
        self.gui_notebook_main = wx.Notebook(self.main_panel.GetWin(), style=wx.NB_TOP,size=(1180,630))
        self.main_panel.Add(self.gui_notebook_main)
        self.vbox.Add(self.main_panel,0)

        self.tx_panel = txPanel(self.gui_notebook_main,self.global_ctrl,self,wx.HORIZONTAL)
        self.ctrl_panel = ctrlPanel(self.gui_notebook_main,self.global_ctrl,self,wx.HORIZONTAL)
        self.gui_notebook_main.AddPage(self.tx_panel, "Demo view")
        self.gui_notebook_main.AddPage(self.ctrl_panel, "Control view")
        
        self.StatusBar.SetStatusText('Status: off')
        
        self.SetSizerAndFit(self.vbox)
        self.SetAutoLayout(True)
        self.Show(True)
        
    def start_tb(self):
        self.global_ctrl.start_flowgraph(send_data=self.data_to_send)
        if self.global_ctrl.get_options_mimo():
            self.StatusBar.SetStatusText('Status: running in MIMO mode')
        else:
            self.StatusBar.SetStatusText('Status: running in SISO mode')
            
    def stop_tb(self):
        self.global_ctrl.stop_flowgraph()
        self.StatusBar.SetStatusText('Status: off')
    
    def set_data_to_send(self, data):
        self.data_to_send = data
                                        
    def create_menus(self):
        filemenu = wx.Menu ()
        main_menubar = wx.MenuBar()
        main_menubar.Append(filemenu, "&File")
        self.SetMenuBar(main_menubar)
        
        menu_exit = filemenu.Append(wx.ID_EXIT, 'E&xit', 'Exit')
        self.Bind(wx.EVT_MENU, self.on_exit, menu_exit)
                
    def on_exit(self, event):
        self.global_ctrl.stop_flowgraph()
        self.global_ctrl.exit()
        ok1 = self.DestroyChildren()
        ok2 = self.Destroy()
        
    def on_close(self,event):
        self.on_exit(wx.EVT_MENU)
