AC_DEFUN([FW_TEMPLATE_ERLANG_EUNIT_CHECK],
[
  FW_TEMPLATE_ERLANG_CHECK_MODULE([eunit],
				  [FW_EUNIT_ERLCFLAGS="-D HAVE_EUNIT=1"],
                                  [FW_EUNIT_ERLCFLAGS=""])

  AC_SUBST([FW_EUNIT_ERLCFLAGS])
])