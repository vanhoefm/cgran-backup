#!/usr/bin/env python

#
# Decoder of IEEE 802.15.4 RADIO Packets.
#
# Modified by: Thomas Schmid, Leslie Choong, Mikhail Tadjikov, Angelo Coluccia
#
  
from gnuradio import gr, eng_notation
from gnuradio import usrp
from gnuradio.ucla_blks import ieee802_15_4_pkt
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import struct, sys, math

def pick_subdevice(u):
    """
    The user didn't specify a subdevice on the command line.
    If there's a daughterboard on A, select A.
    If there's a daughterboard on B, select B.
    Otherwise, select A.
    """
    if u.db(0, 0).dbid() >= 0:       # dbid is < 0 if there's no d'board or a problem
        return (0, 0)
    if u.db(1, 0).dbid() >= 0:
        return (1, 0)
    return (0, 0)

class stats(object):
    def __init__(self):
        self.npkts = 0
        self.nright = 0
        
    
class oqpsk_rx_graph (gr.top_block):
    def __init__(self, options, rx_callback):
        gr.top_block.__init__(self)
        print "cordic_freq = %s" % (eng_notation.num_to_str (options.cordic_freq))


        # ----------------------------------------------------------------

        self.data_rate = options.data_rate
        self.samples_per_symbol = 2
        self.usrp_decim = int (64e6 / self.samples_per_symbol / self.data_rate)
        self.fs = self.data_rate * self.samples_per_symbol
        payload_size = 128             # bytes

        print "data_rate = ", eng_notation.num_to_str(self.data_rate)
        print "samples_per_symbol = ", self.samples_per_symbol
        print "usrp_decim = ", self.usrp_decim
        print "fs = ", eng_notation.num_to_str(self.fs)

        u = usrp.source_c (0, self.usrp_decim)
        if options.rx_subdev_spec is None:
            options.rx_subdev_spec = pick_subdevice(u)
        u.set_mux(usrp.determine_rx_mux_value(u, options.rx_subdev_spec))

        subdev = usrp.selected_subdev(u, options.rx_subdev_spec)
	subdev.select_rx_antenna('RX2')

        print "Using RX d'board %s" % (subdev.side_and_name(),)

        u.tune(0, subdev, options.cordic_freq)
        u.set_pga(0, options.gain)
        u.set_pga(1, options.gain)

        self.u = u

        self.packet_receiver = ieee802_15_4_pkt.ieee802_15_4_demod_pkts(self,
                                                                callback=rx_callback,
                                                                sps=self.samples_per_symbol,
                                                                symbol_rate=self.data_rate,
                                                                threshold=-1)

        self.squelch = gr.pwr_squelch_cc(50, 1, 0, True)
        self.connect(self.u, self.squelch, self.packet_receiver)

def main ():

    db_rssi={} #dictionary
    db_dist={} #dictionary
    db_alpha=[] #list
    db_P0=[] #list

    def rx_callback(ok, payload):
        st.npkts += 1
        if ok:
            st.nright += 1

        (pktno,) = struct.unpack('!H', payload[0:2])
	(rssi,)=struct.unpack('<I', payload[-4:])  #little endian, unsigned int C++ (4 byte) appended to the payload
	if len(payload)>30:
		(MAC_long,)=struct.unpack('<q', payload[17:25]) #little endian, long (8 byte)
		(dest,)=struct.unpack('!H', payload[5:7])
		(MAC_addr,)=struct.unpack('!H', payload[7:9])

		if pktno==0x6188: 
			#Hello messages have Frame Control 0x6188

			z=db_rssi.get(MAC_long)
			if z is None:
			    node_list=[]
			    node_list.append(10 * math.log10(rssi))
			    db_rssi[MAC_long]=node_list
			    db_dist[MAC_long]=input("Insert the distance from %s (ID %s) " % (hex(MAC_long)[:-1], hex(MAC_addr))) 
				#from hex(MAC_long) a spurious last character that emerges after the conversion is removed
			else:
			    z.append(10 * math.log10(rssi))
			    db_rssi[MAC_long]=z
	

			print "ok = %5r  pktno = %4d  len(payload) = %4d  %d/%d  RSSI=%d" % (ok, pktno, len(payload),
				                                                    st.nright, st.npkts, rssi)
			#print "  payload: " + str(map(hex, map(ord, payload)))

			L=len(db_rssi)
			print str(L) + " anchors found"
			if L>=3:
			   a=b=c=d=0        
			   for key in db_rssi:
			       u=sum(db_rssi[key][-10:])/10
			       v=10*math.log10(db_dist[key])
			       a+= u*v
			       b+=u
			       c+=v
			       d+=v*v
			   alpha=(L*a-b*c)/(c**2-L*d)
			   P0=(b+alpha*c)/L
	
			   db_alpha.append(alpha)
			   db_P0.append(P0)
			   print "calibration (alpha, P0) %s %s" % (sum(db_alpha[-10:])/10, sum(db_P0[-10:])/10)

			print " ------------------------"
			sys.stdout.flush()

        
    parser = OptionParser (option_class=eng_option)
    parser.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
                      help="select USRP Rx side A or B (default=first one with a daughterboard)")
    parser.add_option ("-c", "--cordic-freq", type="eng_float", default=2480000000,
                       help="set rx cordic frequency to FREQ", metavar="FREQ")
    parser.add_option ("-r", "--data-rate", type="eng_float", default=2000000)
    parser.add_option ("-f", "--filename", type="string",
                       default="rx.dat", help="write data to FILENAME")
    parser.add_option ("-g", "--gain", type="eng_float", default=0,
                       help="set Rx PGA gain in dB [0,20]")
    
    (options, args) = parser.parse_args ()

    st = stats()

    tb = oqpsk_rx_graph(options, rx_callback)
    tb.start()

    tb.wait()

if __name__ == '__main__':
    # insert this in your test code...
    #import os
    #print 'Blocked waiting for GDB attach (pid = %d)' % (os.getpid(),)
    #raw_input ('Press Enter to continue: ')
    
    main ()
