## This makefile should be included using 
##     include $(top_srcdir)/Makefile.swig
## in Makefile.am's which require SWIG wrapping / compilation.
## For just installing .i files, this Makefile is not required.

## swig flags
## -w511 turns off keyword argument warning
## "-outdir $(builddir)" writes all generated output files to
##   the local builddir (which should always be '.')
##   In some older autotools, $(builddir) is not defined, so
##   just use '.' instead.

SWIG_PYTHON_FLAGS =	\
	-fvirtual	\
	-python		\
	-threads    \
	-modern		\
	-keyword	\
	-w511		\
	-outdir .

## standard swig flags used by most components

STD_SWIG_PYTHON_ARGS =				\
	$(SWIG_PYTHON_FLAGS)			\
	$(STD_DEFINES_AND_INCLUDES)		\
	$(WITH_SWIG_INCLUDES)			\
	$(WITH_INCLUDES)

## standard SWIG LD flags for library creation

STD_SWIG_LA_LD_FLAGS =		\
	$(PYTHON_LDFLAGS)	\
	-module			\
	-avoid-version		\
	$(NO_UNDEFINED)

## standard SWIG library additions for library creation

STD_SWIG_LA_LIB_ADD =  		\
	-lstdc++

## standard SWIG CXXFLAGS
## This allows for code to be compiled with "-O1" instead of "-g -O2"
## for some systems, avoiding some optimization issues.

STD_SWIG_CXX_FLAGS = @swig_CXXFLAGS@

## SWIG suffix for automake to know about

SUFFIXES = .i

## Create $(srcdir)/Makefile.swig.gen, containing all of the rules
## for running SWIG to generate or re-generate outputs.  SWIG file
## names are to be defined in TOP_SWIG_IFILES, and must include the
## full path to the file and full filename including extension.  This
## Makefile addition will be made only if either it does not exist or
## if the top-level template has been modified.

generate-makefile-swig $(srcdir)/Makefile.swig.gen: $(top_srcdir)/Makefile.swig.gen.t
## recreate $(srcdir)/Makefile.swig.gen only if ...
	@do_recreate=0; \
	if test -f $(srcdir)/Makefile.swig.gen; then \
## the file exists and can be removed; or ...
		if $(RM) $(srcdir)/Makefile.swig.gen 2>/dev/null; then \
			if touch $(srcdir)/Makefile.swig.gen 2>/dev/null; then \
				do_recreate=1; \
			fi; \
		fi; \
	else \
## the file doesn't exist, but can be created (e.g., by touching it).
		if touch $(srcdir)/Makefile.swig.gen 2>/dev/null; then \
			do_recreate=1; \
		fi; \
	fi; \
	if test "$$do_recreate" == "1"; then \
		${top_srcdir}/generate-swig-am "${top_srcdir}" "$(srcdir)"; \
	else \
		echo "Cannot recreate $(srcdir)/Makefile.swig.gen because the directory or file is write-protected."; \
		exit -1; \
	fi;

swig_built_sources =

## include the built Makefile.swig.gen, always the one from the
## srcdir; this must be included as the last item, because it depends
## on variables defined above.

include $(srcdir)/Makefile.swig.gen
