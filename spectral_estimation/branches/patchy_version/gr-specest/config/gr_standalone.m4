dnl
dnl Copyright 2008 Free Software Foundation, Inc.
dnl 
dnl This file is part of GNU Radio
dnl 
dnl GNU Radio is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 3, or (at your option)
dnl any later version.
dnl 
dnl GNU Radio is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl 
dnl You should have received a copy of the GNU General Public License along
dnl with this program; if not, write to the Free Software Foundation, Inc.,
dnl 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
dnl

dnl
dnl GR_STANDALONE([package],[version])
dnl
dnl Handles the bulk of the configure.ac work for an out-of-tree build
dnl
dnl N.B., this is an m4_define because if it were an AC_DEFUN it would
dnl get called too late to be useful.

m4_define([GR_STANDALONE],
[
  AC_CONFIG_SRCDIR([config/gr_standalone.m4])
  AM_CONFIG_HEADER(config.h)

  AC_CANONICAL_BUILD
  AC_CANONICAL_HOST
  AC_CANONICAL_TARGET

  AM_INIT_AUTOMAKE

  LF_CONFIGURE_CC
  LF_CONFIGURE_CXX
  GR_LIB64		dnl check for lib64 suffix after choosing compilers

  dnl add ${prefix}/lib${gr_libdir_suffix}/pkgconfig to the head of the PKG_CONFIG_PATH
  if test x${PKG_CONFIG_PATH} = x; then
      PKG_CONFIG_PATH=${prefix}/lib${gr_libdir_suffix}/pkgconfig
  else
      PKG_CONFIG_PATH=${prefix}/lib${gr_libdir_suffix}/pkgconfig:${PKG_CONFIG_PATH}
  fi
  export PKG_CONFIG_PATH

  LF_SET_WARNINGS
  GR_SET_GPROF
  GR_SET_PROF
  AM_PROG_AS
  AC_PROG_LN_S
  AC_PROG_MAKE_SET
  AC_PROG_INSTALL
  AC_PATH_PROG([RM_PROG], [rm])

  AC_LIBTOOL_WIN32_DLL
  AC_ENABLE_SHARED	dnl do build shared libraries
  AC_DISABLE_STATIC	dnl don't build static libraries
  m4_ifdef([LT_INIT],[LT_INIT],[AC_PROG_LIBTOOL])
  dnl GR_FORTRAN

  GR_NO_UNDEFINED	dnl do we need the -no-undefined linker flag
  GR_SCRIPTING		dnl Locate python, SWIG, etc

  dnl Set the c++ compiler that we use for the build system when cross compiling
  if test "x$CXX_FOR_BUILD" = x
  then
    CXX_FOR_BUILD=${CXX}
  fi
  AC_SUBST(CXX_FOR_BUILD)

  dnl Checks for header files.
  AC_HEADER_STDC

  dnl Checks for typedefs, structures, and compiler characteristics.
  AC_C_CONST
  AC_C_INLINE
  AC_TYPE_SIZE_T
  AC_HEADER_TIME
  AC_C_BIGENDIAN

  dnl Check for Mingw support
  GR_PWIN32

  AC_CHECK_PROG([XMLTO],[xmlto],[yes],[])
  AM_CONDITIONAL([HAS_XMLTO], [test x$XMLTO = xyes])

  PKG_CHECK_MODULES(GNURADIO_CORE, gnuradio-core >= 3)
  LIBS="$LIBS $GNURADIO_CORE_LIBS"
])
