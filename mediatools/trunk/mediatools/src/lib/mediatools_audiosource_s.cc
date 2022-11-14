#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <mediatools_audiosource_s.h>
#include <gr_io_signature.h>

mediatools_audiosource_s_sptr 
mediatools_make_audiosource_s (std::vector<std::string> list)
{
  return mediatools_audiosource_s_sptr (new mediatools_audiosource_s (list));
}

static const int MIN_IN = 0;	// mininum number of input streams
static const int MAX_IN = 0;	// maximum number of input streams
static const int MIN_OUT = 1;	// minimum number of output streams
static const int MAX_OUT = 1;	// maximum number of output streams

mediatools_audiosource_s::mediatools_audiosource_s (std::vector<std::string> list)
  : gr_block ("audiosource_s",
	      gr_make_io_signature (MIN_IN, MAX_IN, 0),
	      gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof(short)))
{
    d_list = list;
    d_impl = new mediatools_audiosource_impl();
}

mediatools_audiosource_s::~mediatools_audiosource_s ()
{
}

int 
mediatools_audiosource_s::general_work (int noutput_items,
			       gr_vector_int &ninput_items,
			       gr_vector_const_void_star &input_items,
			       gr_vector_void_star &output_items)
{
  short *out = (short *) output_items[0];

  while(d_data.size() < noutput_items){
    // make sure we have a file open
    while(d_impl->d_ready == false){
        if(d_list.size()<1){
            printf("reached end of pl\n");
            // possibly branch to an idle state here to dump 0's instead of exiting?
            return 0;
        }
        d_impl->open(d_list[0]);
        d_list.erase(d_list.begin(), d_list.begin()+1);
    }

    // grab new data and output
    d_impl->readData(d_data);
  }

  std::copy(d_data.begin(), d_data.begin()+noutput_items, out);
  d_data.erase(d_data.begin(), d_data.begin()+noutput_items);

  return noutput_items;
}
