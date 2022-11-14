ACLOCAL_AMFLAGS = -I config

include $(top_srcdir)/Makefile.common

EXTRA_DIST = bootstrap configure config.h.in \
	Makefile.swig Makefile.swig.gen.t

SUBDIRS = config src docs
DIST_SUBDIRS = config src

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA =

docs:
	cd docs && $(MAKE) $(AM_MAKEFLAGS) docs

.PHONY: docs

