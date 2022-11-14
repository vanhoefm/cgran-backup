AM_CPPFLAGS = -I$(top_srcdir)/kernel -I$(top_srcdir)/dft \
-I$(top_srcdir)/simd $(GCELL_CPPFLAGS)

noinst_LTLIBRARIES = libcell.la
EXTRA_DIST=spufftw-embed.S spufftw-embed-gcell.S

if HAVE_CELL
SUBDIRS=spu
SPU_STUB=spufftw-embed.lo
endif
if HAVE_GCELL
SPU_STUB_GCELL=spufftw-embed-gcell.lo
endif

if SINGLE
PLANS=plans-single.c
else
PLANS=plans-double.c
endif


libcell_la_SOURCES = cell.c conf.c copy.c dft-direct-cell.c	\
transpose.c $(PLANS) fftw-cell.h

libcell_la_LIBADD = $(SPU_STUB) $(SPU_STUB_GCELL)

if HAVE_GCELL
libcell_la_SOURCES += gcell.cc gcell.h
libcell_la_LIBADD += $(GCELL_LIBS)
endif


spufftw-embed.lo: spufftw-embed.S spu/spufftw
	$(LTCOMPILE) -c -x assembler-with-cpp -o $@ $<


spufftw-embed-gcell.lo: spufftw-embed-gcell.S spu/spufftw-gcell
	$(LTCOMPILE) -c -x assembler-with-cpp -o $@ $<
