dnl
dnl Copyright 2005 Free Software Foundation, Inc.
dnl 
dnl This file is part of GNU Radio
dnl 
dnl GNU Radio is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 2, or (at your option)
dnl any later version.
dnl 
dnl GNU Radio is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl 
dnl You should have received a copy of the GNU General Public License
dnl along with GNU Radio; see the file COPYING.  If not, write to
dnl the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
dnl Boston, MA 02111-1307, USA.
dnl 

# GR_X86_64()
#
# Checks to see if we're on a x86_64 machine, and if so, ensure
# that libdir ends in "64"
#
AC_DEFUN([GR_X86_64],[
  AC_REQUIRE([AC_CANONICAL_HOST])
  AC_REQUIRE([AC_PROG_CXX])

  AC_MSG_CHECKING([gr_libdir_suffix])
  gr_libdir_suffix=""
  AC_SUBST(gr_libdir_suffix)

  case "$host_os" in
    linux*) is_linux=yes ;;
    *)      is_linux=no  ;;
  esac

  if test "$is_linux" = no || test "$host_cpu" != "x86_64" && test "$host_cpu" != "powerpc64"; then
    gr_libdir_suffix=""
  elif test "$cross_compiling" = yes; then
    _GR_LIB64_ASK_COMPILER
  elif test "$host_cpu" = "x86_64"; then
    if test -d /lib64 && test ! -L /lib64; then
      gr_libdir_suffix=64
    fi
  else
    _GR_LIB64_ASK_COMPILER  
  fi
  AC_MSG_RESULT([$gr_libdir_suffix])


  AC_MSG_CHECKING([whether to append 64 to libdir])
  t=${libdir##*/lib}
  if test "$t" != 64 && test "$gr_libdir_suffix" = "64"; then
    libdir=${libdir}64
    AC_MSG_RESULT([yes. Setting libdir to $libdir])
  else
    AC_MSG_RESULT([no])
  fi
])

dnl If we're using g++, extract the first SEARCH_DIR("...") entry from the linker script
dnl and see if it contains a suffix after the final .../lib part of the path.
dnl (This works because the linker script varies depending on whether we're generating
dnl 32-bit or 64-bit executables)
dnl
AC_DEFUN([_GR_LIB64_ASK_COMPILER],[
  if test "$ac_cv_cxx_compiler_gnu" = "yes";
  then
    gr_libdir_suffix=`$CXX -Wl,--verbose 2>/dev/null | sed -n -e '/SEARCH_DIR/{s/;.*$//; s,^.*/,,; s/".*$//; s/^lib//; p}'`
  fi
])
