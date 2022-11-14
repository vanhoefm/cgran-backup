#!/usr/bin/env python
#
# Copyright 2005,2006 Free Software Foundation, Inc.
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

import numpy, sys
import random

# define constants 
SYNC_RESVD = 40
MAX_BLOCK_NUM = 3 


def find_new_carrier_map(carrier_map, avail_subc_bin, demand, strategy, gbsize, id):

    """
    find new carrier map based on strategy:
    1: non-contiguous best fit
    2: contiguous best fit
    """

    # reserve leftmost 40 subcarriers for synchronization
    resvd_len = SYNC_RESVD
   
    
    # calc available blocks by adding guard band of gbsize
    ori_avail_subc_bin = add_guard_band(avail_subc_bin, gbsize)
    avail_subc_bin =[]
    avail_subc_bin.extend(ori_avail_subc_bin)

    # calc used block
    used_subc_bin = subc_str2bin(carrier_map)
    used_bin_num = get_bin_num(used_subc_bin)

    # calc common block by comparing available subcarriers and current used subcarriers
    comn_subc_bin = numpy.zeros(len(avail_subc_bin))
    for i in range(len(comn_subc_bin)):
        comn_subc_bin[i] = used_subc_bin[i] and avail_subc_bin[i]
    comn_bin_num = get_bin_num(comn_subc_bin)    
    
    # calc the number of conlicting subcarriers 
    conf_bin_num = used_bin_num - comn_bin_num
    if used_bin_num != 0:
        conf_percent = float(conf_bin_num)/(float(used_bin_num)) * 100
    else:
        conf_percent = 100
    if not conf_bin_num == 0:
        print "\nWarning!!! %d (%0.1f %%) Sub-carrier(s) Conflict!!!   Moving to New Availble Spectrum!!!\n" % (conf_bin_num, conf_percent)
    
    # disable reserved block for SYNC, save the sensing results in resvd_sbc_bin 
    resvd_subc_bin = []
    for i in range(resvd_len):
        resvd_subc_bin.append(avail_subc_bin[i])
        avail_subc_bin[i] = 0
   
    # get the list of available spectrum blocks in data and SYNC carriers
    # for bestfit and worstfit, sort the blocks
    if strategy in [1,2,5]:
        avail_block = get_subc_block_sorted(avail_subc_bin)
        resvd_block = get_subc_block_sorted(resvd_subc_bin)
    else:
        avail_block = get_subc_block_unsorted(avail_subc_bin)
        resvd_block = get_subc_block_unsorted(resvd_subc_bin)


    # check avail_block
    if len(avail_block) == 0:
        print "\nWarning!!! No spectrum block available now!!!\n"
        new_carrier_map = subc_bin2str(avail_subc_bin)
        return new_carrier_map
    
    
    candidate_blocks = get_candidate_blocks(demand, avail_block, resvd_block, ori_avail_subc_bin, strategy)

    # select available block
    new_subc_bin = numpy.zeros(len(avail_subc_bin))
    for i in range(len(candidate_blocks)):
        length = candidate_blocks[i][0]
        index = candidate_blocks[i][1]
        if index == resvd_len:
            direction = 0
        elif index + length == len(avail_subc_bin):
            direction = 1
        else:
            direction = random.randint(0,1)
        if index <= resvd_len and index+length == resvd_len and new_subc_bin[resvd_len] == 1:
            direction = 1

        fill_bits = min(demand,length)

        for j in range(fill_bits):
            if direction == 0:
                new_subc_bin[index + j] = 1                  #add from left to right
            else:
                new_subc_bin[index + length - 1 - j] = 1     #add from right to left

        demand = demand - length
        if demand <= 0:
            break

    # return new carrier map for coordination
    new_carrier_map = subc_bin2str(new_subc_bin)
    print "new_carrier_map is", new_carrier_map
    return new_carrier_map



# static allocation based on link id, allocate equal amount of spectrum from left to right
def new_static_carrier_map(id, demand, avail_subc_bin):
    new_subc_bin = numpy.zeros(len(avail_subc_bin))
    length = len(avail_subc_bin)/4-4
    index = (id-1) * len(avail_subc_bin)/4
    if index >= len(avail_subc_bin)/2:
        index = index + 4
    for j in range(min(demand, length)):
        new_subc_bin[index + j] = 1                  #add from left to right

    # return 
    new_carrier_map = subc_bin2str(new_subc_bin)
    return new_carrier_map



# get candidate spectrum blocks for allocation, according to strategy
def get_candidate_blocks(demand, avail_block, resvd_block, ori_avail_subc_bin, strategy):
    ori_demand = demand
    
    
    # check strategy
    if not strategy in [1, 2]:
        strategy = 1
    
    avail_bin_num = 0
    
    # calculate the number of available subcarriers in the data section
    for i in range(len(avail_block)):
        avail_bin_num += avail_block[i][0]
    resvd_bin_num = 0
    
    # calculate the number of available subcarriers in the reserved section (SYNC)
    for i in range(len(resvd_block)):
        resvd_bin_num += resvd_block[i][0]
    
    to_be_used_block = []
    search_block = []
    
    print demand, avail_bin_num, resvd_bin_num
    
    # if demand > total available subcarriers, use all of them
    if demand >= avail_bin_num + resvd_bin_num:
        to_be_used_block.extend(avail_block)
        to_be_used_block.extend(resvd_block)
        demand = demand - avail_bin_num - resvd_bin_num 
        search_block = []

    # if demand > available subcarriers in the data section, use all available subcarriers in the 
    # data section, and use part of the reserved blocks
    elif demand > avail_bin_num:
        to_be_used_block.extend(avail_block)
        search_block.extend(resvd_block)
        demand = demand - avail_bin_num

    # only use the available subcarriers in the data section
    else:
        search_block = avail_block
    
    accum_block_len = 0
    have_found = 0


    # set the max number of blocks for each link
    if strategy == 1: # contiguous ofdm or contiguous coarse ofdm
        max_block_num = MAX_BLOCK_NUM
    else: 
        max_block_num = 1


    # get the blocks in search_block list according to strategy
    for i in range(len(search_block)):
        accum_block_len += search_block[i][0]
        # if the block is not able to fullfil the demand or worstfit/firstfit is the strategy
        if accum_block_len < demand or strategy in [4, 5, 6]:
            to_be_used_block.append(search_block[i])
        else:
            accum_block_len -= search_block[i][0]
            for j in range(i, len(search_block)):
                if search_block[j][0] < demand-accum_block_len:
                    to_be_used_block.append(search_block[j-1])
                    have_found = 1
                break
            if not have_found:
                to_be_used_block.append(search_block[i])
                break


    while len(to_be_used_block)>max_block_num:
        dump = to_be_used_block.pop()
    
    return to_be_used_block

# lei, add guard band accoding to gbsize
def add_guard_band(avail_subc_bin, gbsize):
    new_avail_subc_bin = avail_subc_bin.copy()
    last = int(avail_subc_bin[0])
    for i in range(0, len(avail_subc_bin)):
        current = int(avail_subc_bin[i])
        if current != last:
            if current == 0:
                for j in range(0, gbsize):
                    if i-j-1 >= 0:
                        new_avail_subc_bin[i-j-1] = 0
            else:
                for j in range(0, gbsize):
                    if i+j < len(avail_subc_bin):
                        new_avail_subc_bin[i+j] = 0
        last = current

    #
    for i in range(0, len(new_avail_subc_bin)/2):
        if int(new_avail_subc_bin[2*i]) == 0 or int(new_avail_subc_bin[2*i+1]) == 0:
            new_avail_subc_bin[2*i] = 0
            new_avail_subc_bin[2*i + 1] = 0

    return new_avail_subc_bin

def subc_bin2str(subc_bin):
    subc_str = ""
    for i in range(0, len(subc_bin)):
        if (i+1) % 4 == 0:
            tmp = int(subc_bin[i-3]*8 + subc_bin[i-2]*4 + subc_bin[i-1]*2 + subc_bin[i])
            subc_str = subc_str + hex(tmp).replace('0x',"")
    return subc_str

def subc_str2bin(subc_map):
    occupied_tones = len(subc_map)*4
    subc_bin = numpy.zeros(occupied_tones)
    for i in range(occupied_tones):
        if((int(subc_map[i/4],16) >> (3-i%4)) & 1 ==0):
            subc_bin[i] = 0
        else :
            subc_bin[i] = 1
    return subc_bin

def get_bin_num(subc_bin):
    bin_num = 0
    for i in subc_bin:
        if int(i) == 1:
            bin_num += 1
    return bin_num



# return a list of the availalbe spectrum blocks in sorted order
def get_subc_block_sorted(subc_bin):
    block_i = []
    block = []
    length = 0
    if (subc_bin[0] == 1):
        index = 0
        block_i.append(index)
    for i in range(1, len(subc_bin) -1):
        if subc_bin[i-1] == 0 and subc_bin[i] == 1:
            index = i
            block_i.append(index)
        if subc_bin[i-1] == 1 and subc_bin[i] == 0:
            length = i - index
            block_i.insert(0,length)
            block.append(block_i)
            block_i = []
    if (subc_bin[len(subc_bin) - 1] == 1) :
        length = len(subc_bin) - index
        block_i.insert(0,length)
        block.append(block_i)
    block.sort()
    block.reverse()
    return block


# return a list of the availalbe spectrum blocks without sorting
def get_subc_block_unsorted(subc_bin):
    block_i = []
    block = []
    length = 0
    if (subc_bin[0] == 1):
        index = 0
        block_i.append(index)
    for i in range(1, len(subc_bin) -1):
        if subc_bin[i-1] == 0 and subc_bin[i] == 1:
            index = i
            block_i.append(index)
        if subc_bin[i-1] == 1 and subc_bin[i] == 0:
            length = i - index
            block_i.insert(0,length)
            block.append(block_i)
            block_i = []
    if (subc_bin[len(subc_bin) - 1] == 1) :
        length = len(subc_bin) - index
        block_i.insert(0,length)
        block.append(block_i)
    
    return block

