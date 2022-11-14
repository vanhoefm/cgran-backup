#ifndef INCLUDED_MEDIATOOLS_AUDIOSOURCE_C_H
#define INCLUDED_MEDIATOOLS_AUDIOSOURCE_C_H

#include <gr_block.h>
#include "mediatools_audiosource_impl.h"

class mediatools_audiosource_s;

typedef boost::shared_ptr<mediatools_audiosource_s> mediatools_audiosource_s_sptr;

mediatools_audiosource_s_sptr mediatools_make_audiosource_s (std::vector<std::string>);

class mediatools_audiosource_s : public gr_block
{
public:
  friend mediatools_audiosource_s_sptr mediatools_make_audiosource_s (std::vector<std::string>);
  mediatools_audiosource_s (std::vector<std::string>);  	// private constructor
  ~mediatools_audiosource_s ();	// public destructor
  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);

  mediatools_audiosource_impl *d_impl;
  std::vector<std::string> d_list;
  std::vector<int16_t> d_data;
};

#endif 
