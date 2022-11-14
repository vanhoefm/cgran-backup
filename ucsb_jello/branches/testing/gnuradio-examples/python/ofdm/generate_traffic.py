#!/usr/bin/env python

# Script for generating traffic traces

from optparse import OptionParser
from gnuradio.eng_option import eng_option
from random import *

LINK_NUM = 4
SEED = 1
SIM_TIME = 2000
TRAFFIC_MIN = 30
TRAFFIC_MAX = 70
TRAFFIC_MEAN = (TRAFFIC_MAX + TRAFFIC_MIN)/2
PERIOD = 10

PATH = "./traffic"

def main():

    # add options
    parser = OptionParser (option_class=eng_option)
    add_options(parser)
    (options, args) = parser.parse_args ()
    
    time = options.time
    traffic_min = options.min
    traffic_max = options.max
    traffic_mean = (traffic_min + traffic_max)/2
    period = options.period    

    seed(options.seed)
    
    print "Generating traffic trace files for %d seconds with seed = %d" % (time, options.seed)
    print "Mean traffic: %d\nMin traffic: %d\nMax traffic: %d" % (traffic_mean, traffic_min, traffic_max)
    print "Average period = %d " % (period) 
    for link in range(0, LINK_NUM):
        t = 0
        filename = "%s/trace_%d" % (PATH, link+1) 
        print "Link%d: %s" % (link, filename)
        FILE = open(filename, 'w')
        while t < time:
            demand = gen_traffic(traffic_min, traffic_max)
            duration = gen_duration(period)
            line = "%8.3f\t%8.3f\t%d" % (t, duration, demand)
            FILE.writelines(line+"\n")
            t += duration
        FILE.close()
        
# generate traffic demand
def gen_traffic(min, max):
    demand = 2 * randint(min/2, max/2) # round demand to even number
    return demand

# generate duration
def gen_duration(period):
    duration = uniform(period/2, period*3/2) 
    return duration 

def add_options(parser):
    parser.add_option("-t", "--time", type="int", default=SIM_TIME,
                      help="Set experiment running time (sec)",  metavar="TIME")
    parser.add_option("", "--seed", type="int", default=SEED,
                      help="Set seed for random number generator",  metavar="SEED")
    parser.add_option("", "--min", type="int", default=TRAFFIC_MIN,
                      help="Minimum traffic demand",  metavar="MIN")
    parser.add_option("", "--max", type="int", default=TRAFFIC_MAX,
                      help="Maximum traffic demand",  metavar="MAX")
    parser.add_option("", "--period", type="int", default=PERIOD,
                      help="Average traffic changing period",  metavar="PERIOD")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass



