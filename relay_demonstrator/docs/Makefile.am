include $(top_srcdir)/Makefile.common

all-local: docs

docs: prep
	doxygen

prep:
	$(MKDIR_P) html

clean-local:
	$(RM) -fr latex html man xml

