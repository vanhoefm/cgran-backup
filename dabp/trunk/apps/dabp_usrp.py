import sys
from gnuradio import usrp

def setup_usrp(freq, gain):
    ######## set up usrp
    u = usrp.source_c(decim_rate=32) # 2M sampling freq
    rx_subdev_spec = usrp.pick_rx_subdevice(u)
    #print "u.db(0,0).dbid() = ", self.u.db(0,0).dbid()
    #print "u.db(1,0).dbid() = ", self.u.db(1,0).dbid()
    #print "rx_subdev_spec = ", options.rx_subdev_spec
    
    mux=usrp.determine_rx_mux_value(u, rx_subdev_spec)
    #print "mux = ", mux
    u.set_mux(mux)

    # determine the daughterboard subdevice we're using
    subdev = usrp.selected_subdev(u, rx_subdev_spec)
    #print "Using RX d'board %s" % (self.subdev.side_and_name(),)
    input_rate = u.adc_freq() / u.decim_rate()
    #print "ADC sampling @ ", eng_notation.num_to_str(self.u.adc_freq())
    #print "USB sample rate %s" % (eng_notation.num_to_str(input_rate))

    subdev.set_gain(gain)
    r = u.tune(0, subdev, freq)
    
    #print "freq range = [%s, %s]" % (eng_notation.num_to_str(self.subdev.freq_min()), eng_notation.num_to_str(self.subdev.freq_max()))
    if not r:
        sys.stderr.write('Failed to set frequency\n')
        raise SystemExit, 1
    return u
    