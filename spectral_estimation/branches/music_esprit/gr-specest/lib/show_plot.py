#!/usr/bin/env python

import pylab
import numpy

def main():
    files = ['qa_specesti_esprit_armadillo.dat', 'qa_specesti_music_armadillo.dat', 'qa_specesti_esprit_fortran.dat']
    grphs = list()
    for f in files:
        tmp = numpy.loadtxt(f)
        print 'Read file %s ... length %u' % (f,len(tmp))
        x = pylab.arange(start=-0.5, stop=0.5, step=1.0/float(len(tmp)))
        grph = pylab.plot(x, 10*pylab.log10(tmp), label=f)
        grphs.append(grph)
    pylab.legend()
    pylab.xlim(-0.5, 0.5)
    pylab.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
