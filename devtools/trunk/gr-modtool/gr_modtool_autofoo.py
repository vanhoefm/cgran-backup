#!/usr/bin/env python
""" A tool for editing GNU Radio modules. """

# Copyright 2010 Communications Engineering Lab, KIT, Germany
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import sys
import os
import re
import glob
from datetime import datetime
from optparse import OptionParser, OptionGroup
from string import Template

### Utility functions ########################################################
def get_class_dict():
    " Return a dictionary in the form command->class "
    classdict = {}
    for g in globals().values():
        try:
            if issubclass(g, ModTool):
                classdict[g.name] = g
                for a in g.aliases:
                    classdict[a] = g
        except (TypeError, AttributeError):
            pass
    return classdict

def get_command_from_argv(possible_cmds):
    """ Read the requested command from argv. This can't be done with optparse,
    since the option parser isn't defined before the command is known, and
    optparse throws an error."""
    command = None
    for arg in sys.argv:
        if arg[0] == "-":
            continue
        else:
            command = arg
        if command in possible_cmds:
            return arg
    #return command # If an invalid command was given, return that 
    return None

def read_file_to_string(filename):
    """ Returns a file as a string. """
    fid = open(filename, 'r')
    string = fid.read()
    fid.close()
    return string

def write_string_to_file(string, filename):
    """ Takes a string and writes it into a file. """
    fid = open(filename, 'w')
    fid.write(string)
    fid.close()

def append_makefile_entry(makefilename, variable, entry):
    " Add entry to the variable in a given automake Makefile. "
    fid = open(makefilename, 'r')
    oldfile = fid.readlines()
    fid.close()
    fid = open(makefilename, 'w')
    var_found = False
    for line in oldfile:
        if not var_found and line.find(variable) == 0:
            var_found = True
        if var_found and line.rstrip()[-1] != "\\":
            fid.write(line.rstrip() + " \\\n\t%s\n" % entry)
            var_found = False
            continue
        fid.write(line)
    fid.close()

def remove_makefile_entry(makefilename, variable, entry):
    """ Removes an entry to a variable in a given automake Makefile.
    It is safe to attempt to remove a non-existing entry. """
    fid = open(makefilename, 'r')
    oldfile = fid.readlines()
    fid.close()
    fid = open(makefilename, 'w')
    line_idx = 0
    while line_idx < len(oldfile) and not oldfile[line_idx].find(variable) == 0:
        fid.write(oldfile[line_idx])
        line_idx += 1
    if line_idx == len(oldfile):
        return
    var_def = ''
    while line_idx < len(oldfile) and not oldfile[line_idx].strip() == '':
        var_def += ' ' + oldfile[line_idx].lstrip().rstrip('\t \\\n')
        line_idx += 1
    var_list = var_def.replace(entry, '').strip().split()
    fid.write('%s = \\\n' % variable)
    first_entry_idx = 0
    for substr in var_list:
        first_entry_idx += 1
        if substr.find('=') != -1:
            break
    var_list = var_list[first_entry_idx:]
    fid.write('\t' + ' \\\n\t'.join(var_list) + '\n')
    for line in oldfile[line_idx:]:
        fid.write(line)
    fid.close()

def append_re_line_sequence(filename, linepattern, newline):
    """ Find a sequence of lines that match linepattern, then insert
    newline after the last line that matches. """
    patt = re.compile(linepattern)
    fid = open(filename, 'r')
    oldfile = fid.readlines()
    fid.close()
    fid = open(filename, 'w')
    seq_found = False
    line_written = False
    for line in oldfile:
        if not line_written and not seq_found and patt.search(line) is not None:
            seq_found = True
        if not line_written and seq_found and patt.search(line) is None:
            fid.write(newline + '\n')
            line_written = True
        fid.write(line)
    # Catches case where the last line of the sequence is the last line in file
    if not line_written and seq_found:
        fid.write(newline + '\n')
    fid.close()

def remove_linematch(filename, linepattern):
    """ Remove all lines from a file that match a given pattern. """
    patt = re.compile(linepattern)
    fid = open(filename, 'r')
    oldfile = fid.readlines()
    fid.close()
    fid = open(filename, 'w')
    for line in oldfile:
        if patt.search(line) is None:
            fid.write(line)
    fid.close()

def str_to_fancyc_comment(text):
    """ Return a string as a C formatted comment. """
    l_lines = text.splitlines()
    outstr = "/* " + l_lines[0] + "\n"
    for line in l_lines[1:]:
        outstr += " * " + line + "\n"
    outstr += " */\n"
    return outstr

def str_to_python_comment(text):
    """ Return a string as a Python formatted comment. """
    l_lines = text.splitlines()
    outstr = ""
    for line in l_lines:
        outstr += "# %s\n" % line
    return outstr

def print_class_descriptions():
    ''' Go through all ModTool* classes and print their name,
        alias and description. '''
    desclist = []
    for gvar in globals().values():
        try:
            if issubclass(gvar, ModTool) and not issubclass(gvar, ModToolHelp):
                desclist.append((gvar.name, ','.join(gvar.aliases), gvar.__doc__))
        except (TypeError, AttributeError):
            pass
    print 'Name      Aliases          Description'
    print '====================================================================='
    for description in desclist:
        print '%-8s  %-12s    %s' % description

def get_modname():
    """ Grep the current module's name from Makefile.common """
    fid = open('Makefile.common', 'r')
    pattern = re.compile("^modname ?= ?(.*)$")
    modname = None
    for line in fid:
        mobj = pattern.match(line)
        if mobj is not None:
            modname = mobj.group(1)
            break
    fid.close()
    return modname


### Templates ################################################################
Templates = {}
# Default licence
Templates['defaultlicense'] = """
Copyright %d <YOU OR YOUR COMPANY>.

This is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this software; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street,
Boston, MA 02110-1301, USA.
""" % datetime.now().year

Templates['work_h'] = """
	int work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items);"""

Templates['generalwork_h'] = """
  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);"""

# Header file of a sync/decimator/interpolator block
Templates['block_h'] = Template("""/* -*- c++ -*- */
$license
#ifndef INCLUDED_${fullblocknameupper}_H
#define INCLUDED_${fullblocknameupper}_H

#include <$grblocktype.h>

class $fullblockname;
typedef boost::shared_ptr<$fullblockname> ${fullblockname}_sptr;

${fullblockname}_sptr ${modname}_make_$blockname ($arglist);

/*!
 * \\brief 
 *
 */
class $fullblockname : public $grblocktype
{
	friend ${fullblockname}_sptr ${modname}_make_$blockname ($argliststripped);

	$fullblockname ($argliststripped);

 public:
	~$fullblockname ();

$workfunc
};

#endif /* INCLUDED_${fullblocknameupper}_H */

""")


# Work functions for C++ GR blocks
Templates['work_cpp'] = """work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items)
{
	const float *in = (const float *) input_items[0];
	float *out = (float *) output_items[0];

	// Do signal processing

	// Tell runtime system how many output items we produced.
	return noutput_items;
}
"""

Templates['generalwork_cpp'] = """general_work (int noutput_items,
			       gr_vector_int &ninput_items,
			       gr_vector_const_void_star &input_items,
			       gr_vector_void_star &output_items)
{
  const float *in = (const float *) input_items[0];
  float *out = (float *) output_items[0];

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each (noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
"""

# C++ file of a GR block
Templates['block_cpp'] = Template("""/* -*- c++ -*- */
$license
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gr_io_signature.h>
#include <$fullblockname.h>


${fullblockname}_sptr
${modname}_make_$blockname ($argliststripped)
{
	return $sptr (new $fullblockname ($arglistnotypes));
}


$fullblockname::$fullblockname ($argliststripped)
	: $grblocktype ("$blockname",
		gr_make_io_signature ($inputsig),
		gr_make_io_signature ($outputsig)$decimation)
{
$constructorcontent}


$fullblockname::~$fullblockname ()
{
}
""")

Templates['block_cpp_workcall'] = Template("""

int
$fullblockname::$workfunc
""")

Templates['block_cpp_hierconstructor'] = """
	connect(self(), 0, d_firstblock, 0);
	// connect other blocks
	connect(d_lastblock, 0, self(), 0);
"""

# Header file for QA
Templates['qa_h'] = Template("""/* -*- c++ -*- */
$license
#ifndef INCLUDED_QA_${fullblocknameupper}_H
#define INCLUDED_QA_${fullblocknameupper}_H

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>
#include <${fullblockname}.h>

class qa_$fullblockname : public CppUnit::TestCase {

	CPPUNIT_TEST_SUITE (qa_$fullblockname);
	CPPUNIT_TEST (t1);
	CPPUNIT_TEST (t2);
	CPPUNIT_TEST_SUITE_END ();

 private:
	void t1 ();
	void t2 ();
};

#endif /* INCLUDED_QA_${fullblocknameupper}_H */

""")

# C++ file for QA
Templates['qa_cpp'] = Template("""/* -*- c++ -*- */
$license
#include <cppunit/TestAssert.h>
#include <qa_$fullblockname.h>

void
qa_$fullblockname::t1()
{
	// Insert CPPUNIT tests/asserts here
}

void
qa_$fullblockname::t2()
{
	// Insert CPPUNIT tests/asserts here
}

""")

# SWIG code
Templates['swig'] = Template("""GR_SWIG_BLOCK_MAGIC(${modname},$blockname);

${modname}_${blockname}_sptr ${modname}_make_$blockname ($arglist);

class ${modname}_$blockname : public $grblocktype
{
 private:
	${modname}_$blockname ($argliststripped);
};
""")

# Python QA code
Templates['qa_python'] = Template("""#!/usr/bin/env python
$license
#

from gnuradio import gr, gr_unittest
import ${modname}$swig

class qa_$blockname (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_t (self):
        # set up fg
        self.tb.run ()
        # check data

        
if __name__ == '__main__':
    gr_unittest.main ()
""")


Templates['hier_python'] = Template('''$license

from gnuradio import gr

class $blockname(gr.hier_block2):
    def __init__(self, $arglist):
    """
    docstring
	"""
        gr.hier_block2.__init__(self, "$blockname",
				gr.io_signature($inputsig),  # Input signature
				gr.io_signature($outputsig)) # Output signature

        # Define blocks
        self.connect()

''')

# Implementation file, C++ header
Templates['impl_h'] = Template('''/* -*- c++ -*- */
$license
#ifndef INCLUDED_QA_${fullblocknameupper}_H
#define INCLUDED_QA_${fullblocknameupper}_H

class $fullblockname
{
 public:
	$fullblockname($arglist);
	~$fullblockname();


 private:

};

#endif /* INCLUDED_${fullblocknameupper}_H */

''')

# Implementation file, C++ source
Templates['impl_cpp'] = Template('''/* -*- c++ -*- */
$license

#include <$fullblockname.h>


$fullblockname::$fullblockname($argliststripped)
{
}


$fullblockname::~$fullblockname()
{
}
''')


Templates['grc_xml'] = Template('''<?xml version="1.0"?>
<block>
  <name>$blockname</name>
  <key>$fullblockname</key>
  <category>$modname</category>
  <import>import $modname</import>
  <make>$modname.$blockname($arglistnotypes)</make>
  <!-- Make one 'param' node for every Parameter you want settable from the GUI.
       Sub-nodes:
       * name
       * key (makes the value accessible as $$keyname, e.g. in the make node)
       * type -->
  <param>
    <name>...</name>
    <key>...</key>
    <type>...</type>
  </param>

  <!-- Make one 'sink' node per input. Sub-nodes:
       * name (an identifier for the GUI)
       * type
       * vlen
       * optional (set to 1 for optional inputs) -->
  <sink>
    <name>in</name>
    <type><!-- e.g. int, real, complex, byte, short, xxx_vector, ...--></type>
  </sink>

  <!-- Make one 'source' node per output. Sub-nodes:
       * name (an identifier for the GUI)
       * type
       * vlen
       * optional (set to 1 for optional inputs) -->
  <source>
    <name>out</name>
    <type><!-- e.g. int, real, complex, byte, short, xxx_vector, ...--></type>
  </source>
</block>
''')

# Usage
Templates['usage'] = """
gr_modtool.py <command> [options] -- Run <command> with the given options.
gr_modtool.py help -- Show a list of commands.
gr_modtool.py help <command> -- Shows the help for a given command. """

### Code generator class #####################################################
class CodeGenerator(object):
    """ Creates the skeleton files. """
    def __init__(self):
        self.defvalpatt = re.compile(" *=[^,)]*")
        self.grtypelist = {
                'sync': 'gr_sync_block',
                'decimator': 'gr_sync_decimator',
                'interpolator': 'gr_sync_interpolator',
                'general': 'gr_block',
                'hiercpp': 'gr_hier_block2',
                'impl': ''}


    def strip_default_values(self, string):
        """ Strip default values from a C++ argument list. """
        return self.defvalpatt.sub("", string)

    def strip_arg_types(self, string):
        """" Strip the argument types from a list of arguments
        Example: "int arg1, double arg2" -> "arg1, arg2" """
        string = self.strip_default_values(string)
        return ", ".join([part.strip().split(' ')[-1] for part in string.split(',')])


    def get_template(self, tpl_id, **kwargs):
        ''' Request a skeleton file from a template.
            First, it prepares a dictionary which the template generator
            can use to fill in the blanks, then it uses Python's
            Template() function to create the file contents. '''
        # Licence
        if tpl_id in ('block_h', 'block_cpp', 'qa_h', 'qa_cpp', 'impl_h', 'impl_cpp'):
            kwargs['license'] = str_to_fancyc_comment(kwargs['license'])
        elif tpl_id in ('qa_python', 'hier_python'):
            kwargs['license'] = str_to_python_comment(kwargs['license'])
        # Standard values for templates
        kwargs['argliststripped'] = self.strip_default_values(kwargs['arglist'])
        kwargs['arglistnotypes'] = self.strip_arg_types(kwargs['arglist'])
        kwargs['fullblocknameupper'] = kwargs['fullblockname'].upper()
        kwargs['grblocktype'] = self.grtypelist[kwargs['blocktype']]
        # Specials for qa_python
        kwargs['swig'] = ''
        if kwargs['blocktype'] != 'hierpython':
            kwargs['swig'] = '_swig'
        # Specials for block_h
        if tpl_id == 'block_h':
            if kwargs['blocktype'] == 'general':
                kwargs['workfunc'] = Templates['generalwork_h']
            elif kwargs['blocktype'] == 'hiercpp':
                kwargs['workfunc'] = ''
            else:
                kwargs['workfunc'] = Templates['work_h']
        # Specials for block_cpp
        if tpl_id == 'block_cpp':
            kwargs['decimation'] = ''
            kwargs['constructorcontent'] = ''
            kwargs['sptr'] = kwargs['fullblockname'] + '_sptr'
            if kwargs['blocktype'] == 'decimator':
                kwargs['decimation'] = ", <+decimation+>"
            elif kwargs['blocktype'] == 'interpolator':
                kwargs['decimation'] = ", <+interpolation+>"
            if kwargs['blocktype'] == 'general':
                kwargs['workfunc'] = Templates['generalwork_cpp']
            elif kwargs['blocktype'] == 'hiercpp':
                kwargs['workfunc'] = ''
                kwargs['constructorcontent'] = Templates['block_cpp_hierconstructor']
                kwargs['sptr'] = 'gnuradio::get_initial_sptr'
                return Templates['block_cpp'].substitute(kwargs)
            else:
                kwargs['workfunc'] = Templates['work_cpp']
            return Templates['block_cpp'].substitute(kwargs) + \
                   Templates['block_cpp_workcall'].substitute(kwargs)
        # All other ones
        return Templates[tpl_id].substitute(kwargs)


### ModTool base class #######################################################
class ModTool(object):
    """ Base class for all modtool command classes. """
    def __init__(self):
        self._subdirs = ['lib', 'python', 'swig', 'grc'] # List subdirs where stuff happens
        self._has_subdirs = {}
        self._skip_subdirs = {}
        self._info = {}
        for subdir in self._subdirs:
            self._has_subdirs[subdir] = False
            self._skip_subdirs[subdir] = False
        self.parser = self.setup_parser()
        self.tpl = CodeGenerator()
        self.args = None
        self.options = None
        self._dir = None


    def setup_parser(self):
        """ Init the option parser. If derived classes need to add options,
        override this and call the parent function. """
        parser = OptionParser(usage=Templates['usage'], add_help_option=False)
        ogroup = OptionGroup(parser, "General options")
        ogroup.add_option("-h", "--help", action="help", help="Displays this help message.")
        ogroup.add_option("-d", "--directory", type="string", default=".",
                help="Base directory of the module.")
        ogroup.add_option("-n", "--module-name", type="string", default=None,
                help="Name of the GNU Radio module. This is normally autodetected from Makefile.common.")
        ogroup.add_option("-N", "--block-name", type="string", default=None,
                help="Name of the block, minus the module name prefix.")
        ogroup.add_option("--skip-lib", action="store_true", default=False,
                help="Don't do anything in the lib/ subdirectory.")
        ogroup.add_option("--skip-swig", action="store_true", default=False,
                help="Don't do anything in the swig/ subdirectory.")
        ogroup.add_option("--skip-python", action="store_true", default=False,
                help="Don't do anything in the python/ subdirectory.")
        ogroup.add_option("--skip-grc", action="store_true", default=True,
                help="Don't do anything in the grc/ subdirectory.")
        parser.add_option_group(ogroup)
        return parser


    def setup(self):
        """ Initialise all internal variables, such as the module name etc. """
        (options, self.args) = self.parser.parse_args()
        self._dir = options.directory
        if not self._check_directory(self._dir):
            print "No GNU Radio module found in the given directory. Quitting."
            sys.exit(1)
        print "Operating in directory " + self._dir

        if options.skip_lib:
            print "Force-skipping 'lib'."
            self._skip_subdirs['lib'] = True
        if options.skip_python:
            print "Force-skipping 'python'."
            self._skip_subdirs['python'] = True
        if options.skip_swig:
            print "Force-skipping 'swig'."
            self._skip_subdirs['swig'] = True

        self._info['modname'] = get_modname()
        print "GNU Radio module name identified: " + self._info['modname']

        self._info['blockname'] = options.block_name
        self.options = options


    def run(self):
        """ Override this. """
        pass


    def _check_directory(self, directory):
        """ Guesses if dir is a valid GNU Radio module directory by looking for
        Makefile.common and at least one of the subdirs lib/, python/ and swig/.
        Changes the directory, if valid. """
        has_makefile = False
        try:
            files = os.listdir(directory)
            os.chdir(directory)
        except OSError:
            print "Can't read or chdir to directory %s." % directory
            return False
        for f in files:
            if os.path.isfile(f) and f == 'Makefile.common':
                has_makefile = True
            elif os.path.isdir(f):
                if (f in self._has_subdirs.keys()):
                    self._has_subdirs[f] = True
                else:
                    self._skip_subdirs[f] = True
        return bool(has_makefile and (self._has_subdirs.values()))


    def _get_mainswigfile(self):
        """ Find out which name the main SWIG file has. In particular, is it
            a MODNAME.i or a MODNAME_swig.i? Returns None if none is found. """
        modname = self._info['modname']
        swig_files = (modname + '.i',
                      modname + '_swig.i')
        for fname in swig_files:
            if os.path.isfile(os.path.join(self._dir, 'swig', fname)):
                return fname
        return None


### ModTool derived classes ##################################################
class ModToolAdd(ModTool):
    """ Add block to the out-of-tree module. """
    name = 'add'
    aliases = ('insert',)
    _block_types = ('sink', 'source', 'sync', 'decimator', 'interpolator',
            'general', 'hiercpp', 'hierpython', 'impl')
    def __init__(self):
        ModTool.__init__(self)
        self._info['inputsig'] = "<+MIN_IN+>, <+MAX_IN+>, sizeof (<+float+>)"
        self._info['outputsig'] = "<+MIN_IN+>, <+MAX_IN+>, sizeof (<+float+>)"
        self._add_cc_qa = False
        self._add_py_qa = False


    def setup_parser(self):
        parser = ModTool.setup_parser(self)
        parser.usage = '%prog add [options]. \n Call %prog without any options to run it interactively.'
        ogroup = OptionGroup(parser, "Add module options")
        ogroup.add_option("-t", "--block-type", type="choice",
                choices=self._block_types, default=None, help="One of %s." % ', '.join(self._block_types))
        ogroup.add_option("--license-file", type="string", default=None,
                help="File containing the license header for every source code file.")
        ogroup.add_option("--argument-list", type="string", default=None,
                help="The argument list for the constructor and make functions.")
        ogroup.add_option("--add-python-qa", action="store_true", default=None,
                help="If given, Python QA code is automatically added if possible.")
        ogroup.add_option("--add-cpp-qa", action="store_true", default=None,
                help="If given, C++ QA code is automatically added if possible.")
        parser.add_option_group(ogroup)
        return parser


    def setup(self):
        ModTool.setup(self)
        options = self.options

        self._info['blocktype'] = options.block_type
        if self._info['blocktype'] is None:
            while self._info['blocktype'] not in self._block_types:
                self._info['blocktype'] = raw_input("Enter code type: ")
                if self._info['blocktype'] not in self._block_types:
                    print 'Must be one of ' + str(self._block_types)
        print "Code is of type: " + self._info['blocktype']

        if (not self._has_subdirs['lib'] and self._info['blocktype'] != 'hierpython') or \
           (not self._has_subdirs['python'] and self._info['blocktype'] == 'hierpython'):
            print "Can't do anything if the relevant subdir is missing. See ya."
            sys.exit(1)

        if self._info['blockname'] is None:
            self._info['blockname'] = raw_input("Enter name of block/code (without module name prefix): ")
        # TODO sanitize
        print "Block/code identifier: " + self._info['blockname']

        self._info['prefix'] = self._info['modname']
        if self._info['blocktype'] == 'impl':
            self._info['prefix'] += 'i'
        self._info['fullblockname'] = self._info['prefix'] + '_' + self._info['blockname']
        print "Full block/code identifier is: " + self._info['fullblockname']

        self._info['license'] = self.setup_choose_license()

        if options.argument_list is not None:
            self._info['arglist'] = options.argument_list
        else:
            self._info['arglist'] = raw_input('Enter valid argument list, including default arguments: ')

        if not (self._info['blocktype'] in ('impl') or self._skip_subdirs['python']):
            self._add_py_qa = options.add_python_qa
            if self._add_py_qa is None:
                self._add_py_qa = (raw_input('Add Python QA code? [Y/n] ').lower() != 'n')
        if not (self._info['blocktype'] in ('hierpython') or self._skip_subdirs['lib']):
            self._add_cc_qa = options.add_cpp_qa
            if self._add_cc_qa is None:
                self._add_cc_qa = (raw_input('Add C++ QA code? [Y/n] ').lower() != 'n')

        if self._info['blocktype'] == 'source':
            self._info['inputsig'] = "0, 0, 0"
            self._info['blocktype'] = "sync"
        if self._info['blocktype'] == 'sink':
            self._info['outputsig'] = "0, 0, 0"
            self._info['blocktype'] = "sync"


    def setup_choose_license(self):
        """ Select a license by the following rules, in this order:
        1) The contents of the file given by --license-file
        2) The contents of the file LICENSE or LICENCE in the modules
           top directory
        3) The default license. """
        if self.options.license_file is not None \
            and os.path.isfile(self.options.license_file):
            return read_file_to_string(self.options.license_file)
        elif os.path.isfile('LICENSE'):
            return read_file_to_string('LICENSE')
        elif os.path.isfile('LICENCE'):
            return read_file_to_string('LICENCE')
        else:
            return Templates['defaultlicense']



    def run(self):
        """ Go, go, go. """
        if self._info['blocktype'] != 'hierpython' and not self._skip_subdirs['lib']:
            self._run_lib()
        has_swig = self._info['blocktype'] in (
                'sink',
                'source',
                'sync',
                'decimator',
                'interpolator',
                'general',
                'hiercpp') and self._has_subdirs['swig'] and not self._skip_subdirs['swig']
        if has_swig:
            self._run_swig()
        if self._add_py_qa:
            self._run_python_qa()
        if self._info['blocktype'] == 'hierpython':
            self._run_python_hierblock()
        if (not self._skip_subdirs['grc'] and self._has_subdirs['grc'] and
            (self._info['blocktype'] == 'hierpython' or has_swig)):
            self._run_grc()


    def _run_lib(self):
        """ Do everything that needs doing in the subdir 'lib'.
        - add .cc and .h files
        - include them into Makefile.am
        - check if C++ QA code is req'd
        - if yes, create qa_*.{cc,h} and add them to Makefile.am and qa_modname.cc
        """
        print "Traversing lib..."
        fname_h = self._info['fullblockname'] + '.h'
        fname_cc = self._info['fullblockname'] + '.cc'
        print "Adding files " + fname_h + " and " + fname_cc + "..."
        if self._info['blocktype'] in ('source', 'sink', 'sync', 'decimator', 'interpolator', 'general', 'hiercpp'):
            write_string_to_file(self.tpl.get_template('block_h', **self._info), os.path.join('lib', fname_h))
            write_string_to_file(self.tpl.get_template('block_cpp', **self._info), os.path.join('lib', fname_cc))
        elif self._info['blocktype'] == 'impl':
            write_string_to_file(self.tpl.get_template('impl_h', **self._info), os.path.join('lib', fname_h))
            write_string_to_file(self.tpl.get_template('impl_cpp', **self._info), os.path.join('lib', fname_cc))
        append_makefile_entry('lib/Makefile.am', 'modinclude_HEADERS', fname_h)
        append_makefile_entry('lib/Makefile.am', 'libgnuradio_%s_la_SOURCES' % self._info['modname'], fname_cc)

        if not self._add_cc_qa:
            return
        fname_qa_h = 'qa_%s' % fname_h
        fname_qa_cc = 'qa_%s' % fname_cc
        print "Adding files " + fname_qa_h + " and " + fname_qa_cc + "..."
        write_string_to_file(self.tpl.get_template('qa_h', **self._info),
                os.path.join('lib', fname_qa_h))
        write_string_to_file(self.tpl.get_template('qa_cpp', **self._info),
                os.path.join('lib', fname_qa_cc))
        append_makefile_entry('lib/Makefile.am', 'noinst_HEADERS', fname_qa_h)
        append_makefile_entry('lib/Makefile.am', 'libgnuradio_%s_qa_la_SOURCES' % self._info['modname'], fname_qa_cc)
        append_re_line_sequence('lib/qa_%s.cc' % self._info['modname'], '^#include',
                 '#include <%s>' % fname_qa_h)
        append_re_line_sequence('lib/qa_%s.cc' % self._info['modname'], 'addTest',
                '\ts->addTest(qa_%s::suite());' % self._info['fullblockname'])


    def _run_swig(self):
        """ Do everything that needs doing in the subdir 'swig'.
        - add .i file
        - include in Makefile.am and modname.i
        """
        print "Traversing swig..."
        fname_i = self._info['fullblockname'] + '.i'
        print "Adding file '" + fname_i + "'..."
        write_string_to_file(self.tpl.get_template('swig', **self._info),
                os.path.join('swig', fname_i))
        print "Editing swig/Makefile.am..."
        append_makefile_entry(
                os.path.join('swig', 'Makefile.am'),
                '%s_swiginclude_headers' % self._info['modname'],
                fname_i)
        fname_mainswig = self._get_mainswigfile()
        if fname_mainswig is None:
            print 'Warning: No main swig file found.'
        else:
            print "Editing %s/%s..." % ('swig', fname_mainswig)
            append_re_line_sequence(os.path.join('swig', fname_mainswig),
                    '^#include', '#include "%s.h"' % self._info['fullblockname'])
            append_re_line_sequence(os.path.join('swig', fname_mainswig),
                    '^%include "(?!gnuradio\.i)', '%%include "%s"' % fname_i)


    def _run_python_qa(self):
        """ Do everything that needs doing in the subdir 'python' to add
        QA code.
        - add .py files
        - include in Makefile.am
        """
        print "Traversing python..."
        fname_py_qa = 'qa_' + self._info['fullblockname'] + '.py'
        print "Adding file '" + fname_py_qa + "'..."
        write_string_to_file(
                self.tpl.get_template('qa_python', **self._info),
                os.path.join('python', fname_py_qa))
        append_makefile_entry(
                os.path.join('python', 'Makefile.am'),
                'noinst_PYTHON',
                fname_py_qa)
        os.chmod(os.path.join('python', fname_py_qa), 0755)


    def _run_python_hierblock(self):
        """ Do everything that needs doing in the subdir 'python' to add
        a Python hier_block.
        - add .py file
        - include in Makefile.am
        """
        print "Traversing python..."
        fname_py = self._info['blockname'] + '.py'
        print "Adding file '" + fname_py + "'..."
        write_string_to_file(
                self.tpl.get_template('hier_python', **self._info),
                os.path.join('python', fname_py))
        append_makefile_entry(
                os.path.join('python', 'Makefile.am'),
                'modpython_PYTHON',
                fname_py)

    def _run_grc(self):
        """ Do everything that needs doing in the subdir 'grc' to add
        a GRC bindings XML file.
        - add .xml file
        - include in Makefile.am
        """
        print "Traversing grc..."
        fname_grc = self._info['fullblockname'] + '.xml'
        print "Adding file '" + fname_grc + "'..."
        write_string_to_file(self.tpl.get_template('grc_xml', **self._info),
                os.path.join('grc', fname_grc))
        print "Editing grc/Makefile.am..."
        append_makefile_entry(os.path.join('grc', 'Makefile.am'),
                                    'dist_grcblocks_DATA', fname_grc)


### Remove module ###########################################################
class ModToolRemove(ModTool):
    """ Remove block (delete files and remove Makefile entries) """
    name = 'remove'
    aliases = ('rm', 'del')
    def __init__(self):
        ModTool.__init__(self)

    def setup_parser(self):
        " Initialise the option parser for 'gr_modtool.py rm' "
        parser = ModTool.setup_parser(self)
        parser.usage = '%prog rm [options]. \n Call %prog without any options to run it interactively.'
        ogroup = OptionGroup(parser, "Remove module options")
        ogroup.add_option("-p", "--pattern", type="string", default=None,
                help="Filter possible choices for blocks to be deleted.")
        ogroup.add_option("-y", "--yes", action="store_true", default=False,
                help="Answer all questions with 'yes'.")
        parser.add_option_group(ogroup)
        return parser

    def setup(self):
        ModTool.setup(self)
        options = self.options

        if options.pattern is not None:
            self._info['pattern'] = options.pattern
        else:
            if options.block_name is not None:
                self._info['pattern'] = options.block_name
            else:
                self._info['pattern'] = raw_input('Which blocks do you want to delete? (Regex): ')
                if len(self._info['pattern']) == 0:
                    self._info['pattern'] = '.'
        self._info['yes'] = options.yes


    def run(self):
        """ Go, go, go! """
        if not self._skip_subdirs['lib']:
            self._run_subdir('lib', ('*.cc', '*.h'), ('noinst_HEADERS',
                            'modinclude_HEADERS',
                            'libgnuradio_%s_la_SOURCES' % self._info['modname'],
                            'libgnuradio_%s_qa_la_SOURCES' % self._info['modname']))
        if not self._skip_subdirs['swig']:
            self._run_subdir('swig', ('*.i',),
                             ('%s_swiginclude_headers' % self._info['modname'],),
                             self._get_mainswigfile(), lambda b: '[^/].*%s' % b[:-2])
        if not self._skip_subdirs['python']:
            self._run_subdir('python', ('*.py',),
                             ('%s_swiginclude_headers' % self._info['modname'],),
                    '__init__.py', lambda b: '.*import.*%s.*' % b[:-3])
        if not self._skip_subdirs['grc']:
            self._run_subdir('grc', ('*.xml',), ('modpython_PYTHON', 'noinst_PYTHON',))


    def _run_subdir(self, path, globs, makefile_vars,
            linematchrm_file=None, linematchrm_pattern=None):
        """ Delete all files that match a certain pattern in path.
            path - The directory in which this will take place
            globs - A tuple of standard UNIX globs of files to delete (e.g. *.xml)
            makefile_vars - A tuple with a list of Makefile.am-variables which
                            may contain references to the globbed files
            linematchrm_file - If given, this file will be opened to delete
                               lines that match a given pattern.
            linematchrm_pattern - This is the pattern that will be deleted from
                                  previously mentioned file.
        """
        # 1. Create a filtered list
        files = []
        for g in globs:
            files = files + glob.glob("%s/%s"% (path, g))
        files_filt = []
        print "Searching for matching files in %s/:" % path
        for f in files:
            if re.search(self._info['pattern'], os.path.basename(f)) is not None:
                files_filt.append(f)
        if len(files_filt) == 0:
            print "None found."
            return
        # 2. Delete files, Makefile entries and other occurences
        yes = self._info['yes']
        for f in files_filt:
            b = os.path.basename(f)
            if not yes:
                ans = raw_input("Really delete %s? [Y/n/a/q]: " % f).lower().strip()
                if ans == 'a':
                    yes = True
                if ans == 'q':
                    sys.exit(0)
                if ans == 'n':
                    continue
            print "Deleting %s." % f
            os.unlink(f)
            print "Deleting occurrences of %s from %s/Makefile.am..." % (b, path)
            for var in makefile_vars:
                remove_makefile_entry('%s/Makefile.am' % path, var, b)
            if not (linematchrm_file is None or linematchrm_pattern is None):
                print "Deleting occurrences of %s from %s/%s..." % (b, path, linematchrm_file)
                remove_linematch('%s/%s' % (path, linematchrm_file), linematchrm_pattern(b))


### Help module ##############################################################
class ModToolHelp(ModTool):
    ''' Show some help. '''
    name = 'help'
    aliases = ('h', '?')
    def __init__(self):
        ModTool.__init__(self)

    def setup(self):
        pass

    def run(self):
        cmd_dict = get_class_dict()
        cmds = cmd_dict.keys()
        cmds.remove(self.name)
        for a in self.aliases:
            cmds.remove(a)
        help_requested_for = get_command_from_argv(cmds)
        if help_requested_for is None:
            print 'Usage:' + Templates['usage']
            print '\nList of possible commands:\n'
            print_class_descriptions()
            return
        cmd_dict[help_requested_for]().setup_parser().print_help()




### Main code ################################################################
def main():
    """ Here we go. Parse command, choose class and run. """
    cmd_dict = get_class_dict()
    command = get_command_from_argv(cmd_dict.keys())
    if command is None:
        print 'Usage:' + Templates['usage']
        sys.exit(2)

    modtool = cmd_dict[command]()
    modtool.setup()
    modtool.run()

if __name__ == '__main__':
    main()

