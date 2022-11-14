#include "dabp_test_grbuf.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>
#include <cstring>

dabp_test_grbuf_sptr dabp_make_test_grbuf(int len)
{
    return dabp_test_grbuf_sptr(new dabp_test_grbuf(len));
}

dabp_test_grbuf::dabp_test_grbuf(int len)
    : gr_block("test_grbuf", 
        gr_make_io_signature(1,1,sizeof(unsigned short)), 
        gr_make_io_signature(1,1,sizeof(unsigned short))), 
    d_len(len), d_stage(0)
{
    set_relative_rate(1);
    set_output_multiple(len);
}

dabp_test_grbuf::~dabp_test_grbuf()
{
}

void dabp_test_grbuf::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    int input_required=noutput_items;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_test_grbuf::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    int nblocks=noutput_items/d_len;
    const unsigned short *in=(const unsigned short*)input_items[0];
    unsigned short *out=(unsigned short *)output_items[0];
    int i;
    
    switch(d_stage) {
        case 0:
        consume_each(d_len);
        d_stage++;
        return 0;
        
        case 1:
        consume_each(d_len);
        d_stage++;
        return d_len;
        
        case 2:
        consume_each(noutput_items);
        d_stage=0;
        return noutput_items;
    }
    return 0;
}

