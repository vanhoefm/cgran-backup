#ifndef INCLUDED_DABP_TEST_GRBUF_H
#define INCLUDED_DABP_TEST_GRBUF_H
#include <gr_block.h>

class dabp_test_grbuf;

typedef boost::shared_ptr<dabp_test_grbuf> dabp_test_grbuf_sptr;

dabp_test_grbuf_sptr dabp_make_test_grbuf(int len);

class dabp_test_grbuf : public gr_block
{
    private:
    friend dabp_test_grbuf_sptr dabp_make_test_grbuf(int len);
    dabp_test_grbuf(int len);
    int d_len;
    int d_stage;
    
    public:
    ~dabp_test_grbuf();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};

#endif // INCLUDED_DABP_TEST_GRBUF_H

