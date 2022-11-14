
include $(top_srcdir)/Makefile.common
AM_CPPFLAGS += $(AVCODEC_CFLAGS) $(AVFORMAT_CFLAGS) 

# C/C++ headers get installed in ${prefix}/include/gnuradio
grinclude_HEADERS =		\
	mediatools_audiosource_s.h	

###################################
# SWIG Python interface and library

TOP_SWIG_IFILES = 		\
	mediatools.i

# Install so that they end up available as:
#   import gnuradio.mediatools
# This ends up at:
#   ${prefix}/lib/python${python_version}/site-packages/gnuradio
mediatools_pythondir_category =	\
	gnuradio

# additional arguments to the SWIG command
#mediatools_la_swig_args =		\
#	-g

# additional sources for the SWIG-generated library
mediatools_la_swig_sources = 	\
	mediatools_audiosource_s.cc	\
	mediatools_audiosource_impl.cc

# additional arguments to the SWIG command
#mediatools_la_swig_args =		\
#	$(SWIG_PYTHON_ARGS)

# additional libraries for linking with the SWIG-generated library
#mediatools_la_swig_libadd =		\
#	libm.dylib

# additional LD flags for linking the SWIG-generated library
mediatools_la_swig_ldflags =	\
	$(AVCODEC_LIBS) \
	$(AVFORMAT_LIBS) 

# additional Python files to be installed along with the SWIG-generated one
#mediatools_python =			\
#	__init__.py

# additional SWIG files to be installed
#mediatools_swiginclude_headers =	\
#	nothing.i

include $(top_srcdir)/Makefile.swig

# add some of the variables generated inside the Makefile.swig.gen
BUILT_SOURCES = $(swig_built_sources)

# Do not distribute the output of SWIG
no_dist_files = $(swig_built_sources)
