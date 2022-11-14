AC_DEFUN([OPENRD_SIGPLOT],[
    GRC_ENABLE(sigplot)

    dnl GRC_CHECK_DEPENDENCY(gr-qtgui, gnuradio-core)

    dnl If execution gets to here, $passed will be:
    dnl   with : if the --with code didn't error out
    dnl   yes  : if the --enable code passed muster and all dependencies are met
    dnl   no   : otherwise

    PYTHON_CHECK_MODULE([PyQt4.QtCore], [PyQt4 for Qt4], \
        [passed=yes], [passed=no], \
        [PyQt4.QtCore.PYQT_VERSION >= 260000])

# Check for: 
#	QtOpenGL
#	QtGui
#	QtCore
#	qwt 
#	qwtplot3d
#	qt4

# qt4-core, qt4-gui, qwt5-qt4, qwt5-qt4-dev, libqwtplot3d-qt4, libqwtplot3d-qt4-dev, qt4-dev-tools

    if test $passed = yes; then
        dnl Check for package qt or qt-mt, set QT_CFLAGS and QT_LIBS
        PKG_CHECK_MODULES(QTCORE, QtCore >= 4.2, [],
            [passed=no; AC_MSG_RESULT([sigplot requires libQtCore >= 4.2.])])
        PKG_CHECK_MODULES(QTGUI, QtGui >= 4.2, [],
            [passed=no; AC_MSG_RESULT([sigplot requires libQtGui >= 4.2.])])
        PKG_CHECK_MODULES(QTOPENGL, QtOpenGL >= 4.2, [],
            [passed=no; AC_MSG_RESULT([sigplot requires libQtOpenGL >- 4.2.])])
        PKG_CHECK_MODULES(GNURADIO_NOUSE, gnuradio-core = 3.3.0, [],
            [passed=no; AC_MSG_RESULT([sigplot requires gnuradio = 3.3.0])])

        dnl Fetch QWT variables
        GR_QWT([], [passed=no])

        dnl Process QWT Plot3D only if QWT passed
        if test "$passed" = "yes"; then
            GR_QWTPLOT3D([], [passed=no])
        fi

        dnl Export the include dirs and libraries (note: QTOPENGL_LIBS includes links
        dnl to QtCore and QtGui libraries)
        QT_INCLUDES="$QWT_CFLAGS $QWTPLOT3D_CFLAGS $QTCORE_CFLAGS $QTGUI_CFLAGS"
        QT_LIBS="$QWT_LIBS $QWTPLOT3D_LIBS $QTOPENGL_LIBS"

        dnl Build an includes variable specifically for running qmake by extracting
        dnl all includes from the QWT and QWTPLOT3D, without the -I;
        dnl qmake appends the -I when processing the project file INCLUDEPATH
        for i in $QWT_CFLAGS $QWTPLOT3D_CFLAGS; do
            QMAKE_INCLUDES="$QMAKE_INCLUDES ${i##-I}"
        done

        QT_MOC_EXEC=`pkg-config --variable=moc_location QtCore`
        QT_UIC_EXEC=`pkg-config --variable=uic_location QtCore`

        AC_SUBST(QMAKE_INCLUDES)
        AC_SUBST(QT_INCLUDES)
        AC_SUBST(QT_LIBS)
        AC_SUBST(QT_MOC_EXEC)
        AC_SUBST(QT_UIC_EXEC)
    fi

    AC_CONFIG_FILES([ \
        src/lib/openrd/sigplotsrc/Makefile \
    ])

    GRC_BUILD_CONDITIONAL(sigplot)
])
