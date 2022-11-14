include $(top_srcdir)/Makefile.common

AM_CPPFLAGS += $(STD_DEFINES_AND_INCLUDES) $(PYTHON_CPPFLAGS) \
              $(QT_INCLUDES) -I. $(WITH_INCLUDES) \
			  -Wno-overloaded-virtual

# Only include these files in the build if sigplot passes configure checks
# This is mostly to help make distcheck pass
QMAKE_SOURCES = 				\
	sigplotdisplayform_moc.cc		\
	sigplotdisplayform_ui.h

EXTRA_DIST = spectrumdisplayform.ui

# These headers get installed in ${prefix}/include/gnuradio
grinclude_HEADERS =			\
	sigplotdisplayform.h		\
	sigplotdisplayform_ui.h  \
	sigplotguiclass.h		\
	sigplot.h \
	sigplot_sink_c.h

TOP_SWIG_IFILES = sigplot.i

sigplot_pythondir_category = openrd
sigplot_la_swig_sources = 	\
	sigplotdisplayform.cc			\
	sigplotguiclass.cc			\
	$(QMAKE_SOURCES)			\
	sigplot_sink_c.cc

sigplot_la_swig_ldflags = 		\
	$(GNURADIO_CORE_LA)	\
	-lgnuradio-qtgui \
	-lstdc++		\
	$(QT_LIBS)

%_moc.cc : %.h
	$(QT_MOC_EXEC) -DQT_SHARED -DQT_NO_DEBUG -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -p $(srcdir) $< -o $@

%_ui.h : %.ui
	$(QT_UIC_EXEC) $< -o $@

# magic flags
#libgnuradio_qtgui_la_LDFLAGS = $(NO_UNDEFINED) $(LTVERSIONFLAGS)

#libgnuradio_qtgui_la_LIBADD = 		\
#	$(GNURADIO_CORE_LA)	\
#	-lstdc++		\
#	$(QT_LIBS)

include $(top_srcdir)/Makefile.swig

BUILT_SOURCES = $(QMAKE_SOURCES)

# add some of the variables generated inside the Makefile.swig.gen
BUILT_SOURCES += 		\
	$(swig_built_sources)

# Do not distribute built sources, they may contain generated paths
# which are invalid on other systems
no_dist_files = $(BUILT_SOURCES)

