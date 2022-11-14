#include "dabp_d2f.h"
#include <gr_io_signature.h>

dabp_d2f_sptr dabp_make_d2f()
{
    return dabp_d2f_sptr(new dabp_d2f());
}

dabp_d2f::dabp_d2f() 
    : gr_sync_block("d2f",
        gr_make_io_signature(1,1,sizeof(double)),
        gr_make_io_signature(1,1,sizeof(float)))
{
}

dabp_d2f::~dabp_d2f()
{
}

int dabp_d2f::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
{
    const double *in=(const double*)input_items[0];
    float *out=(float*)output_items[0];
    for(int i=0;i<noutput_items;i++) {
        out[i]=in[i];
    }
    return noutput_items;
}

