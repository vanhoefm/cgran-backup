TARFILES=simple_fm_rcv.grc handle_rds.c simple_fm_helper.py simple_fm_rcv Makefile
PYTHONPATH:=$(PYTHONPATH):.:$(HOME)/bin
install: simple_fm_rcv.py
	-mkdir -p $(HOME)/bin
	cp -p simple_fm_rcv.py simple_fm_rcv simple_fm_helper.py $(HOME)/bin

simple_fm_rcv.py:
	grcc -d . simple_fm_rcv.grc
	
tarfile:
	tar czvf simple_fm_rcv.tar.gz $(TARFILES)
