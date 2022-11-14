dnl Copyright (C) 1988 Eleftherios Gkioulekas <lf@amath.washington.edu>
dnl  
dnl This program is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 2 of the License, or
dnl (at your option) any later version.
dnl 
dnl This program is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl 
dnl You should have received a copy of the GNU General Public License
dnl along with this program; if not, write to the Free Software 
dnl Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
dnl 
dnl As a special exception to the GNU General Public License, if you 
dnl distribute this file as part of a program that contains a configuration 
dnl script generated by Autoconf, you may include it under the same 
dnl distribution terms that you use for the rest of that program.

# --------------------------------------------------------------------------
# Check whether the C++ compiler accepts a certain flag
# If it does it adds the flag to CXXFLAGS
# If it does not then it returns an error to lf_ok
# Usage:
#   LF_CHECK_CXX_FLAG(-flag1 -flag2 -flag3 ...)
# -------------------------------------------------------------------------

AC_DEFUN([LF_CHECK_CXX_FLAG],[
  echo 'void f(){}' > conftest.cc
  for i in $1
  do
    AC_MSG_CHECKING([whether $CXX accepts $i])
    if test -z "`${CXX} $i -c conftest.cc 2>&1`"
    then
      CXXFLAGS="${CXXFLAGS} $i"
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
    fi
  done
  rm -f conftest.cc conftest.o
])

# --------------------------------------------------------------------------
# Check whether the C compiler accepts a certain flag
# If it does it adds the flag to CFLAGS
# If it does not then it returns an error to lf_ok
# Usage:
#  LF_CHECK_CC_FLAG(-flag1 -flag2 -flag3 ...)
# -------------------------------------------------------------------------

AC_DEFUN([LF_CHECK_CC_FLAG],[
  echo 'void f(){}' > conftest.c
  for i in $1
  do
    AC_MSG_CHECKING([whether $CC accepts $i])
    if test -z "`${CC} $i -c conftest.c 2>&1`"
    then
      CFLAGS="${CFLAGS} $i"
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
    fi
  done
  rm -f conftest.c conftest.o
])

# --------------------------------------------------------------------------
# Check whether the Fortran compiler accepts a certain flag
# If it does it adds the flag to FFLAGS
# If it does not then it returns an error to lf_ok
# Usage:
#  LF_CHECK_F77_FLAG(-flag1 -flag2 -flag3 ...)
# -------------------------------------------------------------------------

AC_DEFUN([LF_CHECK_F77_FLAG],[
  cat << EOF > conftest.f
c....:++++++++++++++++++++++++
      PROGRAM MAIN
      PRINT*,'Hello World!'
      END
EOF
  for i in $1
  do
    AC_MSG_CHECKING([whether $F77 accepts $i])
    if test -z "`${F77} $i -c conftest.f 2>&1`"
    then
      FFLAGS="${FFLAGS} $i"
      AC_MSG_RESULT(yes)  
    else
      AC_MSG_RESULT(no)
    fi
  done
  rm -f conftest.f conftest.o
])

# ----------------------------------------------------------------------
# Provide the configure script with an --with-warnings option that
# turns on warnings. Call this command AFTER you have configured ALL your
# compilers. 
# ----------------------------------------------------------------------

AC_DEFUN([LF_SET_WARNINGS],[
  dnl Check for --with-warnings
  AC_MSG_CHECKING([whether user wants warnings])
  AC_ARG_WITH(warnings,
              [  --with-warnings         Turn on warnings],
              [ lf_warnings=yes ], [ lf_warnings=no ])
  lf_warnings=yes # hard code for now -eb
  AC_MSG_RESULT($lf_warnings)
  
  dnl Warnings for the two main compilers
  cc_warning_flags="-Wall"
  cxx_warning_flags="-Wall -Woverloaded-virtual"
  if test $lf_warnings = yes
  then
    if test -n "${CC}"
    then
      LF_CHECK_CC_FLAG($cc_warning_flags)
    fi
    if test -n "${CXX}" 
    then
      LF_CHECK_CXX_FLAG($cxx_warning_flags)
    fi
  fi
])
