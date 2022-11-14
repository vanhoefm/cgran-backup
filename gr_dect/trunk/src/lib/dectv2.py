# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.31
#
# Don't modify this file, modify the SWIG interface instead.

import _dectv2
import new
new_instancemethod = new.instancemethod
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


def _swig_setattr_nondynamic_method(set):
    def set_attr(self,name,value):
        if (name == "thisown"): return self.this.own(value)
        if hasattr(self,name) or (name == "this"):
            set(self,name,value)
        else:
            raise AttributeError("You cannot add attributes to %s" % self)
    return set_attr



def crc_r(*args):
  """crc_r(string buf) -> unsigned int"""
  return _dectv2.crc_r(*args)
class dectv1_framer_sink_dect_sptr(object):
    """Proxy of C++ dectv1_framer_sink_dect_sptr class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> dectv1_framer_sink_dect_sptr
        __init__(self,  p) -> dectv1_framer_sink_dect_sptr
        """
        this = _dectv2.new_dectv1_framer_sink_dect_sptr(*args)
        try: self.this.append(this)
        except: self.this = this
    def __deref__(*args):
        """__deref__(self)"""
        return _dectv2.dectv1_framer_sink_dect_sptr___deref__(*args)

    __swig_destroy__ = _dectv2.delete_dectv1_framer_sink_dect_sptr
    __del__ = lambda self : None;
    def history(*args):
        """history(self) -> unsigned int"""
        return _dectv2.dectv1_framer_sink_dect_sptr_history(*args)

    def output_multiple(*args):
        """output_multiple(self) -> int"""
        return _dectv2.dectv1_framer_sink_dect_sptr_output_multiple(*args)

    def relative_rate(*args):
        """relative_rate(self) -> double"""
        return _dectv2.dectv1_framer_sink_dect_sptr_relative_rate(*args)

    def start(*args):
        """start(self) -> bool"""
        return _dectv2.dectv1_framer_sink_dect_sptr_start(*args)

    def stop(*args):
        """stop(self) -> bool"""
        return _dectv2.dectv1_framer_sink_dect_sptr_stop(*args)

    def detail(*args):
        """detail(self) -> gr_block_detail_sptr"""
        return _dectv2.dectv1_framer_sink_dect_sptr_detail(*args)

    def set_detail(*args):
        """set_detail(self, gr_block_detail_sptr detail)"""
        return _dectv2.dectv1_framer_sink_dect_sptr_set_detail(*args)

    def name(*args):
        """name(self) -> string"""
        return _dectv2.dectv1_framer_sink_dect_sptr_name(*args)

    def input_signature(*args):
        """input_signature(self) -> gr_io_signature_sptr"""
        return _dectv2.dectv1_framer_sink_dect_sptr_input_signature(*args)

    def output_signature(*args):
        """output_signature(self) -> gr_io_signature_sptr"""
        return _dectv2.dectv1_framer_sink_dect_sptr_output_signature(*args)

    def unique_id(*args):
        """unique_id(self) -> long"""
        return _dectv2.dectv1_framer_sink_dect_sptr_unique_id(*args)

    def basic_block(*args):
        """basic_block(self) -> gr_basic_block_sptr"""
        return _dectv2.dectv1_framer_sink_dect_sptr_basic_block(*args)

    def check_topology(*args):
        """check_topology(self, int ninputs, int noutputs) -> bool"""
        return _dectv2.dectv1_framer_sink_dect_sptr_check_topology(*args)

dectv1_framer_sink_dect_sptr_swigregister = _dectv2.dectv1_framer_sink_dect_sptr_swigregister
dectv1_framer_sink_dect_sptr_swigregister(dectv1_framer_sink_dect_sptr)


def dectv1_framer_sink_dect_block(*args):
  """dectv1_framer_sink_dect_block(dectv1_framer_sink_dect_sptr r) -> gr_block_sptr"""
  return _dectv2.dectv1_framer_sink_dect_block(*args)
dectv1_framer_sink_dect_sptr.block = lambda self: dectv1_framer_sink_dect_block (self)
dectv1_framer_sink_dect_sptr.__repr__ = lambda self: "<gr_block %s (%d)>" % (self.name(), self.unique_id ())


def framer_sink_dect(*args):
  """framer_sink_dect(gr_msg_queue_sptr target_queue) -> dectv1_framer_sink_dect_sptr"""
  return _dectv2.framer_sink_dect(*args)
class dectv1_correlate_access_code_dect_sptr(object):
    """Proxy of C++ dectv1_correlate_access_code_dect_sptr class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> dectv1_correlate_access_code_dect_sptr
        __init__(self,  p) -> dectv1_correlate_access_code_dect_sptr
        """
        this = _dectv2.new_dectv1_correlate_access_code_dect_sptr(*args)
        try: self.this.append(this)
        except: self.this = this
    def __deref__(*args):
        """__deref__(self)"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr___deref__(*args)

    __swig_destroy__ = _dectv2.delete_dectv1_correlate_access_code_dect_sptr
    __del__ = lambda self : None;
    def set_access_code(*args):
        """set_access_code(self, string access_code) -> bool"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_set_access_code(*args)

    def history(*args):
        """history(self) -> unsigned int"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_history(*args)

    def output_multiple(*args):
        """output_multiple(self) -> int"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_output_multiple(*args)

    def relative_rate(*args):
        """relative_rate(self) -> double"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_relative_rate(*args)

    def start(*args):
        """start(self) -> bool"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_start(*args)

    def stop(*args):
        """stop(self) -> bool"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_stop(*args)

    def detail(*args):
        """detail(self) -> gr_block_detail_sptr"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_detail(*args)

    def set_detail(*args):
        """set_detail(self, gr_block_detail_sptr detail)"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_set_detail(*args)

    def name(*args):
        """name(self) -> string"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_name(*args)

    def input_signature(*args):
        """input_signature(self) -> gr_io_signature_sptr"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_input_signature(*args)

    def output_signature(*args):
        """output_signature(self) -> gr_io_signature_sptr"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_output_signature(*args)

    def unique_id(*args):
        """unique_id(self) -> long"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_unique_id(*args)

    def basic_block(*args):
        """basic_block(self) -> gr_basic_block_sptr"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_basic_block(*args)

    def check_topology(*args):
        """check_topology(self, int ninputs, int noutputs) -> bool"""
        return _dectv2.dectv1_correlate_access_code_dect_sptr_check_topology(*args)

dectv1_correlate_access_code_dect_sptr_swigregister = _dectv2.dectv1_correlate_access_code_dect_sptr_swigregister
dectv1_correlate_access_code_dect_sptr_swigregister(dectv1_correlate_access_code_dect_sptr)


def dectv1_correlate_access_code_dect_block(*args):
  """dectv1_correlate_access_code_dect_block(dectv1_correlate_access_code_dect_sptr r) -> gr_block_sptr"""
  return _dectv2.dectv1_correlate_access_code_dect_block(*args)
dectv1_correlate_access_code_dect_sptr.block = lambda self: dectv1_correlate_access_code_dect_block (self)
dectv1_correlate_access_code_dect_sptr.__repr__ = lambda self: "<gr_block %s (%d)>" % (self.name(), self.unique_id ())


def correlate_access_code_dect(*args):
  """correlate_access_code_dect(string access_code, int threshold) -> dectv1_correlate_access_code_dect_sptr"""
  return _dectv2.correlate_access_code_dect(*args)

