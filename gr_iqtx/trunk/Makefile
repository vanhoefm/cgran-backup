files=Makefile gr_iqtx.py gr_iqtx.grc preemph.py gr_iqtx cadence.c
CFLAGS= -O3 -msse -msse2 -mfpmath=sse -ffast-math
LDFLAGS= -lm -msse -msse2 -mfpmath=sse -ffast-math
all: cadence tarfile install
CLEANFILES=*.o *.pyc cadence

clean:
	rm -f $(CLEANFILES)

cadence: cadence.o
install: cadence
	-mkdir $(HOME)/bin
	cp -p gr_iqtx.py $(HOME)/bin
	cp -p gr_iqtx $(HOME)/bin
	cp -p preemph.py $(HOME)/bin
	cp -p cadence $(HOME)/bin

tarfile: cadence
	tar czvf gr_iqtx.tar.gz $(files)
