1. Copyright Information
=====================
This software is part of GNU Radio. It's a free software. You can distribute it and/or modify it under the terms of GNU General Public License as published by the Free Software Foundation (version 3 and later).


2. User Instructions
====================

The high layer implementation of Jello (and Papyrus) is at: gnuradio-examples/python/ofdm/ including some extra test scripts. For each script, use option -h to get all the options supported. We now explain some of these test scripts. 

2.1 Measure Frequency Offset
=============================

The frequency offset between sender and receiver USRP boards, if large enough, can disrupt OFDM transmissions. We recommend to manually measure and compensate the frequency offset, using the following method. 

To measure the frequency offset between USRPs, you need to use one USRP as a reference, whch will send a single-frequency signal at 2.381GHz. 

        $ usrp_siggen.py -f 2.381G -a 10000

usrp_siggen.py is a build-in script in the GNU Radio package. 

For every other USRP, use usrp_fft.py -f 2.381G to emulate a spectrum analyzer. You can get the frequency offset by measuring the frequency difference between the peak power frequency on the PSD map and the central frequency (2.381GHz).

        $ usrp_fft.py -f 2.381G -d 128

NOTE: Please store the measured frequency offset of each link in gnuradio-executes/python/ofdm/freq_offset.dat. The format of each line in freq_offset.dat is:
        USRPBoardId FrequencyOffset(kHz)

To get the USRP board ID, run lsusrp.py on each PC which lists the USRP board ID for all USRPs attached to that PC. For each USRP, write the USPRBoardId and FrequencyOffset(kHz) in freq_offset.dat. In both the Single Link Experiment (see 2.4)  and  Multi-link Experiment (see 2.5), Jello's protocol ssma.py will load freq_offset.dat and automatically compensate the frequency offset.


2.2 Benchmark Test
==================

We provide a benchmark program to test Papyrus's non-contiguous OFDMA transmissions. You will need one USRP sender and one USRP receiver. 

The sender executes:
    
    $ ./benchmark_ofdm_tx.py -f 2.381G --fft 256 --occ 240 --carr 000000000000000ffffffffffffff000000000000ffffffffffffff00000  
    
The receiver executes:
    
    $ ./benchmark_ofdm_rx.py -f 2.381G --fft 256 --occ 240 --carr 000000000000000ffffffffffffff000000000000ffffffffffffff00000  

    Note:
            --fft 256 specifies the fft size as 256;
            --occ 240 specifies the number of occupied subcarriers as 240;
            --carr specifies the subcarriers in use in hex format,  for example: "f3" means the subcarrier usage is 11110011.

The sender continuously sends packets at a constant rate according to the number of used subcarriers. Note that this benchmark script does not automatically compensate frequency offset, so we recommend to manually adjust central frequency using the -f parameter, for example, for a USRP with frequency offset of -4kHz, it shall set -f 2.381004kHz.

After receiving each packet, the receiver will output the received packet number, the status of each packet, and the packet loss rate.


2.3 Generate Traffic Trace
==========================

We provide a tool to generate traffic traces to drive your wireless experiments. The traffic traces are stored in the ./traffic folder. The filename indicates the link index, e.g. trace_2 is the traffic trace for link 2. Each line in a traffic trace file represents a traffic changing event. The format is: 

     TrafficStartTime TrafficDuration DemandValue (in the unit of subcarriers)

For example, 889.951 5.085 34 in trace 1 indicates that link 1 has demand of 34 subcarriers starting at 889.951 second and will last for 5.085 seconds. 


The tool is a python script called generate_traffic.py. It generates sample traffic traces that follow the random uniform distribution between [min, max]. For example, the following command generates a set of traffic trace with random traffic demands between 30 to 50 subcarriers with an average traffic changing period of 10 seconds.

    $ ./generate_traffic.py --min 30 --max 50 --period 10


2.4 Single Link Experiment
==========================

To demonstrate Jello's coordination protocol, we will need at least one sender USRP and one receiver USRP. 

The sender node executes: 

    $ ./ssma.py -f 2.381G -s --fft 256 --occ 240

The receiver node executes:

    $ ./ssma.py -f 2.381G -r --fft 256 --occ 240

Both programs at the sender and receiver will print out details of the coordination status, packet reception rate and spectrum sensing results. 

Since traffic demands change over time, each link will change their subcarrier usage adapting to its demand. To examine frequency sharing between links, you can active another USRP sender using benchmark_ofdm_tx.py. The link running ssma.py can sense the transmissions of the new USRP sender and avoid using the subcarriers used by that interferer USRP. This refers to the per-session FDMA we discussed in our NSDI 2010 paper. 


2.5 Multi-link Experiment
============================

For multi-link experiments, each link needs to have a unique link ID by specifying the --id parameter. 

For example, if there are two links. The first pair of GNU Radio nodes should execute the following commands for the sender and receiver:

    sender1  $ ./ssma.py -f 2.381G -s --fft 256 --occ 240 --id 1 
    receiver1$ ./ssma.py -f 2.381G -r --fft 256 --occ 240 --id 1 
    
The second pair of GNU Radio nodes should execute the following commands for the sender and receiver:

    sender2  $ ./ssma.py -f 2.381G -s --fft 256 --occ 240 --id 2 
    receiver2$ ./ssma.py -f 2.381G -r --fft 256 --occ 240 --id 2 

ssma.py can output the detailed running information, including the spectrum sensing results, the link coordination status, and behaviors of the BACKOFF and the SYNC mechanisms. 


3. Author Information
=====================

This software is developed by the Intelligent Networking Laboratory (LINK), University of California, Santa Barbara.


4. Contact Information
======================

    Email: papyrus.linklab@gmail.com

    Website: http://www.cs.ucsb.edu/~htzheng/papyrus/

    LINK lab Website: http://link.cs.ucsb.edu

