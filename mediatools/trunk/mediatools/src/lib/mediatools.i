/* -*- c++ -*- */

%include "gnuradio.i"			// the common stuff
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"

namespace std
{
    %template(strvec) vector<string>;
}

%{
#include "mediatools_audiosource_s.h"
#include <vector>
#include <string>
%}

// ----------------------------------------------------------------

GR_SWIG_BLOCK_MAGIC(mediatools,audiosource_s);

mediatools_audiosource_s_sptr mediatools_make_audiosource_s (std::vector<std::string>);

class mediatools_audiosource_s : public gr_sync_block
{
private:
  mediatools_audiosource_s (std::vector<std::string>);
};
