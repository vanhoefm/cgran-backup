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

from numpy import *
from numpy.linalg import inv
import copy

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

    db_rssi={}
    db_anchors={} 
    file_ML=open("ML_positions.txt","w")
    file_ML_quant=open("ML_quant_positions.txt","w")
    file_lat=open("lat_positions.txt","w")

    def norm(x,y):
	dis=0
	for i in range(3):
	    dis+=(x[i]-y[i])**2
	return math.sqrt(dis)

    def rx_callback(ok, payload):
        st.npkts += 1
        if ok:
            st.nright += 1

        (pktno,) = struct.unpack('!H', payload[0:2])
	(rssi,)=struct.unpack('<I', payload[-4:])  #little endian, unsigned int C++ (4 byte) passato in coda al payload
	if len(payload)>30:
		(MAC_long,)=struct.unpack('<q', payload[17:25]) #little endian, long (8 byte)
		(dest,)=struct.unpack('!H', payload[5:7])
		(MAC_addr,)=struct.unpack('!H', payload[7:9])

		if pktno==0x6188: 
			#Hello messages have Frame Control 0x6188

			z=db_rssi.get(MAC_long)
			if z is None:  #found a new anchor
			    node_list=[]
			    node_list.append(10 * math.log10(rssi))
			    db_rssi[MAC_long]=node_list
			    print "new anchor %s found (ID %s)" % (hex(MAC_long)[:-1], hex(MAC_addr))
				#from hex(MAC_long) a spurious last character that emerges after the conversion is removed
			    c_x=float(input("Insert x coordinate "))
			    c_y=float(input("Insert y coordinate "))
			    c_z=float(input("Insert z coordinate "))
			    db_anchors[MAC_long]=array([c_x, c_y, c_z])
			    
			    if len(db_anchors)>=3: #computes matrices for lateration
				global nodes, A, At, b, nodeN, nodeN_key
				nodes=copy.deepcopy(db_anchors)
				ult=nodes.popitem()
				nodeN=ult[1]
				nodeN_key=ult[0]
				A=zeros((len(nodes),3))
				b=zeros((len(nodes),1))
				i=-1
				for node in nodes.keys():
				    i+=1
				    A[i,0]=2*(nodes[node][0]-nodeN[0])
				    A[i,1]=2*(nodes[node][1]-nodeN[1])
				    A[i,2]=2*(nodes[node][2]-nodeN[2])
				    b[i]=nodes[node][0]**2-nodeN[0]**2 + nodes[node][1]**2-nodeN[1]**2 + nodes[node][2]**2-nodeN[2]**2
				At=transpose(A)

			else:
			    z.append(10 * math.log10(rssi))
			    db_rssi[MAC_long]=z

			print "ok = %5r  pktno = %4d  len(payload) = %4d  %d/%d  RSSI=%d" % (ok, pktno, len(payload),
		                                                            st.nright, st.npkts, rssi)
			#print "  payload: " + str(map(hex, map(ord, payload)))
		
			file_txt=open("measurements.txt","w")
			alpha=0.25
			P0=33
			P_avg={}
			dist={}
			for key in db_rssi:
		     		P_avg[key]=sum(db_rssi[key][-10:])/10
		     		dist[key]=10 ** ((P0-P_avg[key])/10/alpha)
		     		print "distance from %s: %s" % (hex(key)[:-1], dist[key])

		     		file_txt.write("Measurements from node "+hex(key)[:-1]+"\n")
		     		for elem in db_rssi[key]:
		     			file_txt.write(str(elem)+" ")
		     		file_txt.write("\n")

			file_txt.close()

			if len(db_anchors)>= 3:  #at least 3 anchors are needed for localization

			   p=[1, 1, 1]
			   p_prec=[-1, -1, -1]
			   for k in range(100):
			     p_prec[0]=p[0]
			     p_prec[1]=p[1]
			     p_prec[2]=p[2]	
			     for key in db_rssi:
				 p[0]-=0.1*20*alpha/math.log(10)* (P_avg[key]-P0+alpha*10*math.log10(norm(p,db_anchors[key]))) *(p[0]-db_anchors[key][0])/ (norm(p,db_anchors[key]) ** 2)
				 p[1]-=0.1*20*alpha/math.log(10)* (P_avg[key]-P0+alpha*10*math.log10(norm(p,db_anchors[key]))) *(p[1]-db_anchors[key][1])/ (norm(p,db_anchors[key]) ** 2)
				 p[2]-=0.1*20*alpha/math.log(10)* (P_avg[key]-P0+alpha*10*math.log10(norm(p,db_anchors[key]))) *(p[2]-db_anchors[key][2])/ (norm(p,db_anchors[key]) ** 2)
			   print "ML position estimate: "+str(p)
			   file_ML.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+"\n")




			   p=[1, 1, 1]
			   p_prec=[-1, -1, -1]
			   for k in range(100):
			     p_prec[0]=p[0]
			     p_prec[1]=p[1]
			     p_prec[2]=p[2]	
			     for key in db_rssi:
				 p[0]-=0.1*20*alpha/math.log(10)* (round(P_avg[key])-P0+alpha*10*math.log10(norm(p,db_anchors[key]))) *(p[0]-db_anchors[key][0])/ (norm(p,db_anchors[key]) ** 2)
				 p[1]-=0.1*20*alpha/math.log(10)* (round(P_avg[key])-P0+alpha*10*math.log10(norm(p,db_anchors[key]))) *(p[1]-db_anchors[key][1])/ (norm(p,db_anchors[key]) ** 2)
				 p[2]-=0.1*20*alpha/math.log(10)* (round(P_avg[key])-P0+alpha*10*math.log10(norm(p,db_anchors[key]))) *(p[2]-db_anchors[key][2])/ (norm(p,db_anchors[key]) ** 2)
			   print "ML position estimate quantized: "+str(p)
			   file_ML_quant.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+"\n")






			   c=zeros((len(nodes),1))
			   i=-1
			   for node in nodes:
			      i+=1
			      c[i]=b[i]+dist[nodeN_key]**2-dist[node]**2 	
			   p_lat=transpose(dot(dot(inv(dot(At,A)),At),c))
			   print "position by lateration: " +str(p_lat)
			   file_lat.write(str(p_lat[0][0])+" "+str(p_lat[0][1])+" "+str(p_lat[0][2])+"\n")

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
