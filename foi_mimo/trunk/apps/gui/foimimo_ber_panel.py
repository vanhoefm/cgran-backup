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
import wx.lib.sheet

class berPanel(grc_wxgui.Panel):
    def __init__(self, parent, global_ctrl, orient=wx.HORIZONTAL):
        grc_wxgui.Panel.__init__ (self,parent,orient)
        self.global_ctrl = global_ctrl
        
        self.text_meas_specific = wx.StaticText(self, label="Measurement specific input => ")
        self.text_tx_pkt = wx.StaticText(self, label="Nr of transmitted packets:")
        self.textbox_tx_pkt = wx.TextCtrl(self,size=(80,-1),style=wx.TE_PROCESS_ENTER)
        self.text_comments = wx.StaticText(self, label="Comments:")
        self.textbox_comments = wx.TextCtrl(self,size=(200,-1),style=wx.TE_PROCESS_ENTER)
        self.calc_button = wx.Button(self,-1,"Calc new BER")
        self.Bind(wx.EVT_BUTTON, self.on_calc_button, self.calc_button)
        
        self.text_general = wx.StaticText(self, label="General input => ")
        self.text_tx_ampl = wx.StaticText(self, label="Tx ampl scale:")
        self.textbox_tx_ampl = wx.TextCtrl(self,size=(40,-1),style=wx.TE_PROCESS_ENTER)
        self.text_tx_gain = wx.StaticText(self, label="Tx gain:")
        self.textbox_tx_gain = wx.TextCtrl(self,size=(40,-1),style=wx.TE_PROCESS_ENTER)
        self.text_tx_ext_gain = wx.StaticText(self, label="Tx external gain:")
        self.textbox_tx_ext_gain = wx.TextCtrl(self,size=(40,-1),style=wx.TE_PROCESS_ENTER)
        self.textbox_tx_ext_gain.SetValue('0')
        self.text_general_comments = wx.StaticText(self, label="General comments:")
        self.textbox_general_comments = wx.TextCtrl(self,size=(200,-1),style=wx.TE_PROCESS_ENTER)
        
        self.save_button = wx.Button(self,-1,"Save file")
        self.Bind(wx.EVT_BUTTON, self.on_save_button, self.save_button)
        self.savefile_prefix = '1'
        self.savefile_name = "ber_save_"
        self.savefile_type = ".ber"
        self.text_filename = wx.StaticText(self, label="Savefile name: "+self.savefile_name)
        self.textbox_savefile_prefix = wx.TextCtrl(self,size=(40,-1),style=wx.TE_PROCESS_ENTER)
        self.textbox_savefile_prefix.SetValue(self.savefile_prefix)
        self.text_filetype = wx.StaticText(self, label=self.savefile_type)
        
        self.clear_button = wx.Button(self,-1,"Clear")
        self.Bind(wx.EVT_BUTTON, self.on_clear_button, self.clear_button)
        
        self.sheet_ber = berSheet(self) 
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.meas_ctrl_box = wx.BoxSizer(wx.HORIZONTAL)
        self.general_ctrl_box = wx.BoxSizer(wx.HORIZONTAL)
        self.clearsave_box = wx.BoxSizer(wx.HORIZONTAL)
        self.control_box = wx.BoxSizer(wx.VERTICAL)
        flags = wx.ALIGN_LEFT|wx.ALL|wx.ALIGN_CENTRE_VERTICAL
        
        self.meas_ctrl_box.Add(self.text_meas_specific,0,border=3,flag=flags)
        self.meas_ctrl_box.Add(self.text_tx_pkt,0,border=3,flag=flags)
        self.meas_ctrl_box.Add(self.textbox_tx_pkt,0,border=3,flag=flags)
        self.meas_ctrl_box.Add(self.text_comments,0,border=3,flag=flags)
        self.meas_ctrl_box.Add(self.textbox_comments,0,border=3,flag=flags)
        self.meas_ctrl_box.Add(self.calc_button,0,border=3,flag=flags)
        
        self.general_ctrl_box.Add(self.text_general,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.text_tx_ampl,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.textbox_tx_ampl,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.text_tx_gain,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.textbox_tx_gain,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.text_tx_ext_gain,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.textbox_tx_ext_gain,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.text_general_comments,0,border=3,flag=flags)
        self.general_ctrl_box.Add(self.textbox_general_comments,0,border=3,flag=flags)
        
        self.clearsave_box.Add(self.clear_button,0,border=3,flag=flags)
        self.clearsave_box.Add(self.save_button,0,border=3,flag=flags)
        self.clearsave_box.Add(self.text_filename,0,border=3,flag=flags)
        self.clearsave_box.Add(self.textbox_savefile_prefix,0,border=3,flag=flags)
        self.clearsave_box.Add(self.text_filetype,0,border=3,flag=flags)
        
        self.control_box.Add(self.general_ctrl_box,0)
        self.control_box.Add(self.meas_ctrl_box,0)
        
        self.vbox.Add(self.control_box,0)
        self.vbox.Add(self.sheet_ber,1)
        self.vbox.Add(self.clearsave_box,0)
        
        self.SetSizer(self.vbox)
        
    def on_calc_button(self,e):
        options = self.global_ctrl.get_options()
        (npkt_rcvd,npkt_right,nr_bad_headers) = self.global_ctrl.get_pkt_statistics()
        (nbit_rcvd,nbit_right) = self.global_ctrl.get_bit_statistics()
            
        tx_pkt = int(self.textbox_tx_pkt.GetValue())
        not_rcvd_pkt = tx_pkt - npkt_rcvd
        tx_bit = tx_pkt*(options.size-8-2)*8
        ber = 1
        ber_rcvd = 1
        if nbit_rcvd > 0:
            ber_rcvd = 1.0 - float(nbit_right)/float(nbit_rcvd)
            ber = float(tx_bit-nbit_right)/float(tx_bit)
            
        per = 1
        per_rcvd = 1
        if npkt_rcvd > 0:
            per_rcvd = 1.0 - float(npkt_right)/float(npkt_rcvd)
            per = float(tx_pkt-npkt_right)/float(tx_pkt)
        
        comment =str(self.textbox_comments.GetValue())
        
        data = [ber,ber_rcvd,per,per_rcvd,npkt_right,npkt_rcvd-npkt_right,not_rcvd_pkt,nr_bad_headers,tx_pkt,comment]
        self.sheet_ber.insertRowWithData(data)
        self.global_ctrl.reset_statistics()
    
    def on_save_button(self,e):
        self.savefile_prefix = self.textbox_savefile_prefix.GetValue()
        out_file = open(self.savefile_name+self.savefile_prefix+self.savefile_type,mode="ab")
        
        options = self.global_ctrl.get_options()
        out_file.write('==== General ====')
        out_file.write('\nPacket size: ')
        out_file.write(str(options.size))
        out_file.write('\nSample rate: ')
        out_file.write(str(options.sample_rate))
        out_file.write('\nCenter freq: ')
        out_file.write(str(options.center_freq))
        out_file.write('\nCode rate: ')
        out_file.write(options.code_rate)
        out_file.write('\nFFT size: ')
        out_file.write(str(options.fft_length))
        out_file.write('\nOccupied tones: ')
        out_file.write(str(options.occupied_tones))
        out_file.write('\nCP size: ')
        out_file.write(str(options.cp_length))
        out_file.write('\n==== TX ====')
        out_file.write('\nAmpl scale: ')
        ampl = self.textbox_tx_ampl.GetValue()
        out_file.write(ampl)
        out_file.write('\nGain: ')
        gain_tx = self.textbox_tx_gain.GetValue()
        out_file.write(gain_tx)
        out_file.write('\nExternal gain: ')
        gain_ext = self.textbox_tx_ext_gain.GetValue()
        out_file.write(gain_ext)
        out_file.write('\n==== RX ====')
        out_file.write('\nGain: ')
        out_file.write(str(options.gain_rx))
        out_file.write('\n==== General comments ====\n')
        general_comments = self.textbox_general_comments.GetValue()
        out_file.write(general_comments)
        out_file.write('\n==== Measurements ====')
        out_file.write('\nBER, BER(rcvd), PER, PER(rcvd), Rcvd OK, Rcvd not OK, Not rcvd, Bad headers, Nr transmitted pkts, Comments, ')
        out_file.write('\n')
        
        rows = self.sheet_ber.GetNumberRows()
        cols = self.sheet_ber.GetNumberCols()
        for r in range(0, rows):
            for c in range(0, cols):
                cell_value = self.sheet_ber.GetCellValue(r, c)
                out_file.write(cell_value)
                out_file.write(', ')
            out_file.write('\n')
        out_file.close()
    
    def on_clear_button(self,e):
        self.sheet_ber.SetNumberRows(0)
    
class berSheet(wx.lib.sheet.CSheet):
    def __init__(self,parent):
        wx.lib.sheet.CSheet.__init__(self,parent)
        
        self.SetNumberRows(0)
        self.SetNumberCols(10)
        
        col_size = 100
        
        self.SetColLabelValue(0,'BER')
        self.SetColLabelValue(1,'BER(rcvd)')
        self.SetColLabelValue(2,'PER')
        self.SetColLabelValue(3,'PER(rcvd)')
        self.SetColLabelValue(4,'Rcvd OK')
        self.SetColLabelValue(5,'Rcvd not OK')
        self.SetColLabelValue(6,'Not rcvd')
        self.SetColLabelValue(7,'Bad headers')
        self.SetColLabelValue(8,'Nr sent')
        self.SetColLabelValue(9,'Comments')
        
        for col in range(self.GetNumberCols()):
            self.SetColSize(col,col_size)
    def insertRowWithData(self,data):
        self.InsertRows()
        row = 0
        for col,d in enumerate(data):
            self.SetCellValue(row,col,str(d))
            
