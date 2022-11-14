This directory contains examples for transmitting and receiving 802.11
style bpsk.  The following are brief descriptions of the python scrips
in this directory.

bbn_80211b.py:
  This module instantiates the gr-blocks that do the encoding and
decoding.  For both the transmitter and receiver the user can choose
whether to use a raised-cosine baseband pulse, or the 802.11 Barker
baseband pulse.  Currently, the transmitter does not work very well
with the Barker baseband pulse, however given a sufficient signal to
noise ratio (and processing power), the receiver can correctly decode
802.11 1 Mbps packets.  Also, once in a while, the receiver can
correctly decode an 802.11 2 Mbps (qpsk) packet, however qpsk
tranmission is not yet supported (although it is possible to transmit
and receive at 2 Mbps using bpsk and 2 bits per symbol).

bbn_80211b_pkt.py:
  This module is wraps the classes in bbn_80211b.py.  The user should
instantiate the classes in this file, and not instantiate the classes
in bbn_80211b.py directly.  The bbn_80211b_mod_pkts class instantiates
the transmit packet queue and provides the send_pkt function.  The
send_pkt function takes a payload, prepends the 802.11 long preamble
and plcp header, appeands the payload crc, and queues the packet for
transmission.  The bbn_80211b_demod_pkts class instantiates the
receive queue and creates a thread to watch for received packets.

bbn_80211b_transmit_path.py:
  This file contains a class that connects an instance of
bbn_80211b_mod_pkts to the usrp.

bbn_80211b_test.py:
  This class instantiates the transmitter and receiver and simulates
an additive white Gaussian noise channel.  Its sends a few packets
across the simulated channel, and prints out their contents when they
are received.

bbn_80211_rx.py:
bbn_80211_tx.py:
  These python scripts respectively receive and send packets over the
usrp.  There are several different ways of running these scripts.  The
transmit script uses a raised cosine baseband pulse, and the receive
script can be configured to look for either a raised cosine, or a
Barker pulse.  To send some packets between two usrps at 500 kBps, on
one machine start the receiver:

    ./bbn_80211_rx.py

  and on the other machine run the transmitter:

    ./bbn_80211_tx.py

  If it works, the receiver will output a "PKT" message for each
packet it receives.  By default, both the transmitter and receiver use
8 samples per symbol, run at 4 megasamples per second, and use 2.4Mhz
as the center frequency.  The receiver configures the usrp to use 8
bits for each I and Q sample, and the transmitter uses 16 bits for
each I and Q sample.  To send data at 1 Mpbs, you can configure the
receiver to use 4 samples per symbol:

    ./bbn_80211_rx.py --spb 4

  and do the same for the transmitter:

    ./bbn_80211_tx.py --spb 4

  On a fast machine, the receiver program should be able to capture
real 802.11 1 Mbps packets and a few 2 Mbps packets.  Running in this
mode seems to max out the cpu on our 1.6 Ghz Pentium M.  Despite the
fact that the symbol rate of 802.11 bpsk and qpsk is only 1
megasymbols per second, the energy in the 802.11 signal is spread over
11 Mhz (at baseband) using an 11-chip Barker spreading code.
Fortunately, under high signal-to-noise ratio conditions, it is not
necessary to sample the entire 11 Mhz bandwidge to recover the
symbols.  We have succeeded in recovering 802.11 packets using an 8
Mhz sampling rate.  To try to catch 802.11 1 Mpbs packets, use the
following command:

    ./bbn_80211b_rx.py -d 8 -f 2412e6 -b

  The "-d 8" tells the receiver to run at 8 megasamples per second
(instead of 4), and  "-f 2412e6" means to listen on 802.11 channel 1.
(The 802.11 channels are 5 Mhz apart, so channel 2 is at 2427, channel
3 is at 2422 etc..)  The "-b" means to look for the Barker code.  A
message will be printed out for each received packet that passes the
crc check.  Use the "-p" option to disable crc checking.  Note that
when crc checking is enabled the crc is sripped off the packet, so
disabling crc checking causes the received packet to be 4 bytes
longer.

bbn_80211_tap.py:

  This script creates a tap interface that can used to send and
receive packets.  On Linux a standard ethernet tap (called gr0) is
created, and on NetBSD an 802.11 (called tap0) interface is created.
Most of the options in the other scripts are available for the tap
script.  However, the tap script has a "-z" option for transmit only
(disable the reciver), and the "-t" option to enable the transmitter
(while leaving the receiver enabled).  If you use both -r and -t
options at the same time the transmitter will be enabled and the
reciver will be diabled.  To run the tap in 802.11 channel 6 without
transmitting, use the following command:

    ./bbn_80211b_tap.py

The defaults are shown in this command

    ./bbn_80211b_tap.py -d 8 -f 2437e6 -b
