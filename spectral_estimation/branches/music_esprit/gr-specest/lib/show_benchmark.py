#!/usr/bin/env python

import pylab
import numpy

def main():
    data = numpy.loadtxt('qa_specesti_esprit_armadillo_bench.dat')
    pylab.plot(data[:,0], data[:,2]/data[:,1], label='ESPRIT Armadillo')
    pylab.plot(data[:,0], data[:,3]/data[:,1], label='ESPRIT Armadillo (Spectrum)')
    data = numpy.loadtxt('qa_specesti_esprit_fortran_bench.dat')
    pylab.plot(data[:,0], data[:,2]/data[:,1], label='ESPRIT Fortran')
    pylab.plot(data[:,0], data[:,3]/data[:,1], label='ESPRIT Fortran (Spectrum)')
    data = numpy.loadtxt('qa_specesti_music_armadillo_bench.dat')
    pylab.plot(data[:,0], data[:,2]/data[:,1], label='MUSIC Armadillo')
    pylab.plot(data[:,0], data[:,3]/data[:,1], label='MUSIC Armadillo (Spectrum)')
    pylab.legend()
    pylab.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
