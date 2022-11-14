INSTALL_FILES=meteor_detector.py meteor_detector meteor_helper.py meteor_bb_analyser.py \
    fft_raw2asc meteor_bb_analyser meteor_det_analyser meteor_rates_analyser

PYTHONPATH:=.:$(PYTHONPATH)
SYSPREFIX=/usr/local
export PYTHONPATH

PYTHONPATH+=:.:$(HOME)/bin

all: fft_raw2asc meteor_detector.py meteor_bb_analyser.py

fft_raw2asc: fft_raw2asc.c

meteor_detector.py: meteor_detector.grc
	rm -f met-bb-fifo*
	touch met-bb-fifo0 met-bb-fifo1 met-bb-fifo2
	grcc -d . meteor_detector.grc

meteor_bb_analyser.py: meteor_bb_analyser.grc
	grcc -d . meteor_bb_analyser.grc
	
install:
	cp $(INSTALL_FILES) $(HOME)/bin

clean:
	rm -f meteor_detector.py
	rm -f meteor_bb_analyser.py
