#!/usr/bin/env python
# coding: utf-8

# Execution INFOs
# sudo PYTHONPATH=/usr/local/lib/python2.6/dist-packages/ GR_SCHEDULER=STS nice -n -20 ./Gen2_Listener_1-Antenna.py
# /usr/local/lib/python2.6/dist-packages/ --> substitute with your python installation directory
# nice -n -20                             --> high priority for the process
# GR_SCHEDULER=STS or TPB                 --> GNUradio single-thread scheduler or thread-per-block scheduler


from gnuradio import gr, gru, window, listener
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from string import split
from string import strip
from string import atoi
import time
import os
import sys

# Acknowledgement: I would like to acknowledge Michael Buettner since my work on the RFID Listener development started from his CGRAN project "Gen2 RFID Reader"

'''
Settings of important parameters
This is a DEMO version of the RFID Listener presented at Mobicom 2010. In order to detect the preamble and correctly decode the messages transmitted by the tag some preliminary considerations must be taken into account. According to the Gen2 protocol the reader chooses the up-link parameters and communicates these choices to the tags using the opening symbols of each packet (reader’s commands preamble). For these reason, decoding reader commands is required to access the up-link parameters and best tune-up in real-time both the matched filter and the “Tag Decoder” block (ref. Figure 7 of the paper). This version doesn't automatically tune-up the parameters. So you need to run the application, detect the parameters that your Reader uses according to the output of the Reader Decoder log file and finally tune up BLF and Miller (M) values.
'''
up_link_freq      = 250  	# in khz. it's the BLF (Backscatter Link Frequency of the Tag) in Dense Reader Mode --> set it according to the Reader and mode 				# you use. The value is determined as DR/TRcal. Both DR and TRCal parameters are outputted by the Reader Decoder block.
miller            = 4           # it's the number of Miller subcarriers that tag adopt in uplink. It's value is the M parameter, outputted by the Reader Decoder 					# block. 
dec_rate          = 8       	# decimation factor at USRP
sw_dec            = 2		# software decimation
samples_per_pulse = 8		# samples per pulse of the tag signal. It's calculated according as (1/(2*up_link_freq*1000))/((dec_rate*sw_dec)/64e6)
pulse_width       = 12  	# in microsec. equal to Tari/2 --> necessary for the Reader Matched Filter 
interp_factor     = 2		# interpolation factor for tag signal, utilized by clock recovery block	



# FIND DAUGHTERBOARD	           
def pick_subdevice(u):
    """
    The user didn't specify a subdevice on the command line.
    If there's a daughterboard on A, select A.
    If there's a daughterboard on B, select B.
    Otherwise, select A.
    """
    if u.db(0, 0).dbid() >= 0:       # dbid is < 0 if there's no d'board or a problem
        return (0, 0)
    if u.db(0, 0).dbid() >= 0:
        return (1, 0)
    return (0, 0)



# Block connections      				                                                                       
class my_top_block(gr.top_block):
    def __init__(self, freq, samp_freq_READER, u, tag_monitor, reader_monitor_cmd_gate, matched_filter_READER, matched_filter_TAG, delay_TAG, delay_fft, sw_decimator):
        gr.top_block.__init__(self)
               
        # ASK demodulator and MUX
	to_mag_mux = listener.to_mag_mux()       

	# Clock recovery block
        cr = listener.clock_recovery(samples_per_pulse, interp_factor)
        
	# Null sink. Change if you want to save samples on file.
	file_sink = gr.null_sink(gr.sizeof_float*1)
	
	# Enable real-time scheduling	
        r = gr.enable_realtime_scheduling()
    	if r != gr.RT_OK:
            print "Warning: failed to enable realtime scheduling"

	# Carrier tracking blocks
	track_fft_size = 64
    	find_CW = listener.find_CW(track_fft_size, freq, samp_freq_READER, reader_monitor_cmd_gate)
        mywindow = window.blackmanharris(track_fft_size)
    	fft = gr.fft_vcc(track_fft_size, True, mywindow)
    	c2mag = gr.complex_to_mag_squared(track_fft_size)	
    	s2v = gr.stream_to_vector(gr.sizeof_gr_complex*1, track_fft_size)

        # Create flow-graph
	self.connect(u, to_mag_mux)
	self.connect((to_mag_mux,0), matched_filter_READER, (reader_monitor_cmd_gate,0))	#READER DECODING CHAIN
        self.connect((to_mag_mux,1), matched_filter_TAG, delay_TAG, (reader_monitor_cmd_gate,1))	#TAG DECODING CHAIN
	self.connect(reader_monitor_cmd_gate, cr, tag_monitor, file_sink)

	# Connect carrier tracking blocks
	self.connect((to_mag_mux,2), delay_fft, sw_decimator, s2v, fft, c2mag, find_CW)
		
	

# Main program       
def main(self):
    
    which_usrp = 0		
    fpga = "std_2rxhb_2tx.rbf"									# FPGA file
    freq = 866.5e6 										# Center frequency of 2 MHz European RFID Band
    rx_gain = 20   									      									          
    samp_freq_READER = (64 / dec_rate) * 1e6						
    us_per_sample_READER = float (1 / (64.0 / (dec_rate*sw_dec)))		
    samp_freq_TAG = (64 / dec_rate) * 1e6						
    us_per_sample_TAG = float (1 / (64.0 / (dec_rate*sw_dec) / samples_per_pulse))	
        
    u = usrp.source_c(which_usrp, dec_rate, fpga_filename= fpga)        			# Create USRP source
    rx_subdev_spec = pick_subdevice(u)					
    u.set_mux(usrp.determine_rx_mux_value(u, rx_subdev_spec))  		
    rx_subdev = usrp.selected_subdev(u, rx_subdev_spec)
    rx_subdev.set_gain(rx_gain)						 
    rx_subdev.set_auto_tr(False)					
    rx_subdev.select_rx_antenna('RX2')								# RX Antenna on RX2 Connector of USRP 
 
    r = u.tune(0, rx_subdev, freq)								# Tuning Daughterboard @ Center Frequency
    if not r:
        print "Couldn't set rx frequency"
    
    
    # FILE SOURCE for offline tests (comment previous lines and remove comments from next line)
    # u = gr.file_source(gr.sizeof_gr_complex*1, "file.out", True)
        
    print ""
    print "*************************************************************"
    print "************* Single-Antenna RFID Listener ******************"
    print "*************************************************************\n"

    print "USRP central frequency: %s Hz" % str(freq)
    print "Sampling Frequency for READER: "+ str(samp_freq_READER) + " Hz" + " --- microsec. per Sample: " + str(us_per_sample_READER)
    print "Sampling Frequency for TAG: "+ str(samp_freq_TAG) + " Hz" + " --- microsec. per Sample: " + str(us_per_sample_TAG)
    
    # READER MATCHED FILTER 
    num_taps_READER = pulse_width * (1 / ( 1 / (64 / float(dec_rate)))) 
    taps = []
    for i in range(0,int(num_taps_READER)):
        taps.append(float(1))
    matched_filter_READER = gr.fir_filter_fff(sw_dec, taps)

    # TAG MATCHED FILTER
    num_taps_TAG = samp_freq_TAG/(2*up_link_freq*1000)  
    taps = []
    for i in range(0,int(num_taps_TAG)):
        taps.append(float(1))
    matched_filter_TAG = gr.fir_filter_fff(sw_dec, taps)

    # Tag Decoding Block --> the boolean value in input indicate if EPC real-time video output is enabled or not 
    tag_monitor = listener.tag_monitor(True, miller)
    
    # Reader Decoding Block --> as input we have sample duration and a boolean value which indicate if real-time video output is enabled or not 
    reader_monitor_cmd_gate = listener.reader_monitor_cmd_gate(us_per_sample_READER, False)
    
    # Dealy block for TAG and READER chain synchronization 
    delay_TAG = gr.delay(gr.sizeof_float*1,int(num_taps_READER-num_taps_TAG))

    # Delay block and software decimator for carrier tracking chain synchronization
    delay_fft = gr.delay(gr.sizeof_gr_complex*1,int(num_taps_READER))	
    sw_decimator = gr.keep_one_in_n(gr.sizeof_gr_complex*1, sw_dec)	

    # Create python flow-graph
    tb = my_top_block(freq, samp_freq_READER, u, tag_monitor, reader_monitor_cmd_gate, matched_filter_READER, matched_filter_TAG, delay_TAG, delay_fft, sw_decimator)
    
    # Run the flow-graph
    tb.start()
    
    while 1:
	c = raw_input("PRESS 'q' for STOPPING the application")
        if c == "Q" or c == "q":
	 	break

    # Stop the flow-graph
    tb.stop()
        
    # GETTING READER LOG MONITOR
    log_READER = reader_monitor_cmd_gate.get_log()
    print "Log for READER has %s Entries" % str(log_READER.count())
  
    c = raw_input("PRESS 'f' to write READER LOG on file, 'q' to QUIT, 's' for skipping to TAG\n")
    if c == "q":
		print "\n Shutting Down...\n"
		return
    if c == "f":
		print "\n Writing to file...\n"
		self.actual_command = ""
		reader_file = open("reader_log.out","w")
    		reader_file.close()
		reader_file = open("reader_log.out","a")
		i = log_READER.count()    		
		for k in range(0, i):
    			decode_reader_log_msg(log_READER.delete_head_nowait(),reader_file,self)
        		k = k + 1
		reader_file.close()
    if c == "s":
		print "\n Skipping to TAG...\n"
		
    # GETTING TAG LOG MONITOR
    log_TAG = tag_monitor.get_log()
    print "Log for TAG has %s Entries" % (str(log_TAG.count()))
  
    c = raw_input("PRESS 'f' to write TAG LOG on file, 'q' to QUIT\n")
    if c == "q":
		print "\n Shutting Down...\n"
		return
    if c == "f":
		print "\n Writing to file...\n"
		tag_file = open("tag_log.out","w")
    		tag_file.close()
		tag_file = open("tag_log.out","a")
		i = log_TAG.count();    		
		for k in range(0, i):
    			decode_tag_log_msg(log_TAG.delete_head_nowait(),tag_file)
        		k = k + 1
    		tag_file.close()


# Decode Tag Messages
def decode_tag_log_msg(msg,tag_file):
    LOG_RN16, LOG_EPC, LOG_ERROR, LOG_OKAY = range(4)

    fRed = chr(27) + '[31m'
    fBlue = chr(27) + '[34m'
    fReset = chr(27) + '[0m'

    if msg.type() == LOG_RN16:
	fields = split(strip(msg.to_string()), " ")
        rn16 = fields[0].split(",")[0]      
	rssi = strip(fields[0].split(",")[1])  
	if msg.arg2() == LOG_ERROR:
            print "%s\t    %s RN16 w/ Error: %s%s" %(fields[-1],fRed, rn16, fReset)
            tag_file.write("%s\tRN16: %s    \tRSSI: %s  \tERROR\n" % (fields[-1],rn16,rssi))
	else:
            print "%s\t     RN16: %s" %(fields[-1], rn16)
            tag_file.write("%s\tRN16: %s    \tRSSI: %s\n" % (fields[-1],rn16,rssi))

    if msg.type() == LOG_EPC:
        fields = split(strip(msg.to_string()), " ")
        epc = fields[0].split(",")[0]
        rssi = strip(fields[0].split(",")[1])
        epc = epc[16:112]
        tmp = atoi(epc,2)
        if msg.arg2() == LOG_ERROR:
            print "%s\t    %s EPC w/ Error: %024X%s" %(fields[-1],fRed, tmp, fReset)
            tag_file.write("%s\tEPC: %024X\tRSSI: %s  \tCRC_ERROR\n" % (fields[-1],tmp,rssi))
	else:
            print "%s\t    %s EPC: %024X%s" %(fields[-1],fBlue, tmp, fReset)
            tag_file.write("%s\tEPC: %024X\tRSSI: %s\n" % (fields[-1],tmp,rssi))
      

        
# Decode Reader Messages
def decode_reader_log_msg(msg,reader_file,father):
    READER_MSG, PWR_DWN, PWR_UP, INVALID_RTCAL, INVALID_TRCAL  = range(5)
    OKAY, ERROR = range(2)

    command = ""
            
    fields = split(msg.to_string(), "/")
    item = None
    if msg.type() == READER_MSG:
        if msg.arg2() == ERROR:
            print 'ERROR of decoding NEXT MESSAGE \n',
        for i in range(len(fields)):
            item = fields[i]
           
            if item[0:3] == "CMD":
                cmd = split(item, ":")[1]
            if item[0:4] == "Tari":
                tari = split(item,":")[1]
            if item[0:2] == "PW":
                pw = split(item,":")[1]
            if item[0:5] == "RTCal":
                rtcal = split(item,":")[1]
            if item[0:5] == "TRCal":
                trcal = split(item,":")[1]
            if item[0:11] == "COMMAND_LEN":
                cmd_len = split(item, ":")[1]
	    if item[0:11] == "READER_FREQ":
		frequency = split(item, ":")[1]                
	    if item[0:8] == "MAX_RSSI":
		max_RSSI = split(item, ":")[1] 
	    if item[0:8] == "AVG_RSSI":
		avg_RSSI = split(item, ":")[1]
	    if item[0:4] == "TIME":
		time = split(item, ":")[1]  
        
        if cmd[:2] == "00":
            if len(cmd) == 4:
                print "-->READER MESSAGE @ %.1f MHz<-- : QRep Session %s -- \n" % (float(frequency),cmd[2:])
		if (father.actual_command=="ACK"):
			reader_file.write("%.1f MHz\t%s\tQRep\n" % (float(frequency),time))
                command = "QRep"
		father.actual_command="QRep"
		print "\tCommand Lenght: %s microsec. \n\tInter-arrival: %s microsec.\n\n" % (cmd_len, msg.arg1())
                
            else:
                print "-->READER MESSAGE @ %.1f MHz<-- ERROR in decoding QREP : QRep Invalid: %s \n" % (float(frequency),cmd)
		
        elif cmd[:2] == "01":
            if len(cmd) == 18:
                print "-->READER MESSAGE @ %.1f MHz<-- : ACK: %s  -- \n" % (float(frequency),cmd[2:])
		reader_file.write("%.1f MHz\t%s\tACK %s  \n" % (float(frequency),time,cmd[2:]))
                command = "ACK"
		father.actual_command = "ACK"
		print "\tCommand Lenght: %s microsec. \n\tInter-arrival: %s microsec.\n\n" % (cmd_len, msg.arg1())
            else:
                print "-->READER MESSAGE @ %.1f MHz<-- ERROR in decoding ACK : ACK Invalid: %s \n" % (float(frequency),cmd),
		
        elif cmd[:4] == "1000":
            if len(cmd) == 22:
                dr = cmd[4]
                m = cmd[5:7]
                trext = cmd[7]
                sel = cmd[8:10]
                session = cmd[10:12]
                target = cmd[12]
                q = cmd[13:17]
                
                print "-->READER MESSAGE @ %.1f MHz<-- : QUERY \n" % float(frequency)
		#c = raw_input("'d' for detail, 'c' to continue  \n")
		#if c == "d":
		#	print "\tTari: %s \n\tPW: %s \n\tRTCal: %s \n\tTRCal: %s \n\tDR: %s \n\tMiller: %s \n\tTRExt: %s \n\tSel: %s \n\tSession: %s \n\tTarget: %s \n\tQ: %s \n\t# Slots: %s \n" % (tari, pw, rtcal, trcal, dr, m, trext, sel, session, target, q, int(math.pow(2, bin_to_dec(q))))
		#	print "\tCommand Lenght: %s microsec. \n\tInter-arrival: %s microsec.\n\n" % (cmd_len, msg.arg1())		
		#if c == "c":
			#print "CRC: %s" % cmd[17:22]
                command = "QUERY"
                
            else:
                print "-->READER MESSAGE @ %.1f MHz<-- ERROR in decoding QUERY : Query_Len Incorrect: %s" % (float(frequency),cmd),

        elif cmd[:4] == "1001":
            if len(cmd) == 9:
                print "-->READER MESSAGE @ %.1f MHz<-- : QAdj Session %s -- Adj: %s" % (float(frequency),cmd[5:7], cmd[7:]),
                command = "QAdj"
		print "\tCommand Lenght: %s microsec. \n\tInter-arrival: %s microsec.\n\n" % (cmd_len, msg.arg1())
            else:
                print "-->READER MESSAGE @ %.1f MHz<-- ERROR in decoding QAdj : QAdj_Len Incorrect %s" % (float(frequency),cmd),

        elif cmd[:8] == "11000000":
            print "-->READER MESSAGE @ %.1f MHz<-- : NAK" % float(frequency)
	    reader_file.write("%.1f MHz\t%s\tNAK\n" % (float(frequency),time))
            command = "NAK"
	    father.actual_command = "NAK"
	    print "\tCommand Lenght: %s microsec. \n\tInter-arrival: %s microsec.\n\n" % (cmd_len, msg.arg1())

        elif cmd[:4] == "1010":
            target = cmd[4:7]
            action = cmd[7:10]
            bank = cmd[10:12]
            print "-->READER MESSAGE @ %.1f MHz<-- : SELECT\n" % float(frequency)
	    #c = raw_input("'d' per DETTAGLI , 'c' per CONTINUARE  \n")
	    #if c == "d":
		#print "\tTarget: %s \n\tAction: %s \n\tBank: %s \n" % (target, action, bank)
		#print "\tCommand Lenght: %s microsec. \n\tInter-arrival: %s microsec.\n\n" % (cmd_len, msg.arg1())		
	    #if c == "c":
		#print "CRC: %s" % cmd[17:22]
            command = "SELECT"
	
        else:
            print "*****-->Unknown Command @ %.1f MHz<--*****: %s" % (float(frequency),cmd),

                        
    if msg.type() == PWR_DWN:
        print "POWER DOWN @ %s microsec. @ %s MHz\n" % (msg.arg1(), msg.arg2())
	
    if msg.type() == PWR_UP:
        print "POWER UP @ %s microsec. @ %s MHz\n" % (msg.arg1(), msg.arg2())
	        
    if msg.type() == INVALID_TRCAL:
        print "-->READER MESSAGE @ %.1f MHz<-- ERROR : Invalid TRCal: %s\n" % (float(frequency),msg.arg1())
   
    
# Convert binary values to decimal values
def bin_to_dec(bits):
    dec = 0
    for i in range(1, len(bits) + 1):
        dec = dec + int(bits[-i]) * math.pow(2, i - 1)
    return dec


# Single Antenna Gen2 RFID Listener  
if __name__ == '__main__':
    main (my_top_block)
