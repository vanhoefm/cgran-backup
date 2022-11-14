#!/usr/bin/env python
#
# Copyright 2005,2006,2007 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#
# This code fragment was adopted from the gr-dab package written by Andreas Muller

class channel_conf:
    """
    @brief Read and parse the channel configuration file
    """
    def __init__(self, filename):
        f = open(filename,'r')
        self.subch=[]
        line = f.readline()
        while line != "":
            if line[0]=='#' or line[0]=='\n':
                pass
            else:
                tmplist = str.split(line,',')
                if len(tmplist)>=7:
                    subchid = int(tmplist[0])
                    sid = int(tmplist[1])
                    label = tmplist[2]
                    start_addr = int(tmplist[3])
                    subchsz = int(tmplist[4])
                    option = int(tmplist[5])
                    protect_lev = int(tmplist[6])
                    
                    optprot = option*4+protect_lev
                    entry = [subchid, sid, label, start_addr, subchsz, optprot]
                    self.subch.append(entry)
                
            line = f.readline()
        f.close()
    
    def get_info(self, subchid):
        for entry in self.subch:
            if entry[0]==subchid or subchid<0:
                return entry[1:]
                
    def get_sid(self, subchid):
        for entry in self.subch:
            if entry[0]==subchid or subchid<0:
                return entry[1]
                
    def get_label(self, subchid):
        for entry in self.subch:
            if entry[0]==subchid or subchid<0:
                return entry[2]
                
    def get_start_addr(self, subchid):
        for entry in self.subch:
            if entry[0]==subchid or subchid<0:
                return entry[3]
                
    def get_subchsz(self, subchid):
        for entry in self.subch:
            if entry[0]==subchid or subchid<0:
                return entry[4]
                
    def get_optprot(self, subchid):
        for entry in self.subch:
            if entry[0]==subchid or subchid<0:
                return entry[5]
    
