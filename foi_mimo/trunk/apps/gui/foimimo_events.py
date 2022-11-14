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

import wx

myEVT_PER = wx.NewEventType()
EVT_PER = wx.PyEventBinder(myEVT_PER,1)

class per_event(wx.PyCommandEvent):
    def __init__(self,evtType,id):
        wx.PyCommandEvent.__init__(self,evtType,id)
        self.per_value = None
    def set_per_val(self,val):
        self.per_value = val
    def get_per_val(self):
        return self.per_value

myEVT_PKT = wx.NewEventType()
EVT_PKT = wx.PyEventBinder(myEVT_PKT,1)

class rcvd_data_event(wx.PyCommandEvent):
    def __init__(self,evtType,id):
        wx.PyCommandEvent.__init__(self,evtType,id)
        self.pkt_data = None
    def set_data(self,data):
        self.pkt_data = data
    def get_data(self):
        return self.pkt_data
