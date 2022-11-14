from gnuradio import eng_notation
from gnuradio.eng_option import eng_option

import random
import struct
import sys


def extract_7_2_3_2_Static_information(str1):

	print "Slot Number %d" % (int(str1[4:8],2)), "-%d" % (int(str1[4:8],2)+12)
	print "Number of Transceivers %d" % (int(str1[11:13],2)+1)
        print "Extended RF carrier information 1=yes: ",str1[13:14]
	print "RF carriers available " , str1[14:24]
	print "Carrier Number %d" % (int(str1[26:32],2))
        print "Primary receiver Scan Carrier Number %d" % (int(str1[34:40],2))
def extract_7_2_3_7_Multiframe_number(str1):

	print "Spare bits " , (str1[4:16])
	print "multiframe number " , (str1[16:40])
        

def extract_Q_system_information(str1):

  print "entering extract Q ", str1[:4],
  for ch in str1:
        if not ch in ('0', '1'):
		print "Nothing no 1 or 0 I can play with"
		return
# Q type packet header decoding (4 bits a8-a11)

  if str1[:4] == "0000" or str1[:4] == "0001":
	print "Static system info"
	extract_7_2_3_2_Static_information(str1)
  if str1[:4] == "0010":
	print "Extended RF carrier information part 1"
  if str1[:4] == "0011":
	print "Fixed part capabilities"
  if str1[:4] == "0100":
	print "Extended fixed part capabilities part 1"
  if str1[:4] == "0101":
	print "Secondary access rights identities"
  if str1[:4] == "0110":
	print "Multi-frame number"
	extract_7_2_3_7_Multiframe_number(str1)
  if str1[:4] == "0111":
	print "Escape"
  if str1[:4] == "1000":
	print "Obsolete not to be used"
  if str1[:4] == "1001":
	print "Extended RF carriers part 2"
  if str1[:4] == "1011":
	print "Transmit information"
  if str1[:4] == "1100":
	print "Extended fixed part capabilities part 2"

  

