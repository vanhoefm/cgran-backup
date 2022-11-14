#ifndef INCLUDED_DABP_D2F_H
#define INCLUDED_DABP_D2F_H
#include <gr_sync_block.h>

class dabp_d2f;

typedef boost::shared_ptr<dabp_d2f> dabp_d2f_sptr;

dabp_d2f_sptr dabp_make_d2f();

class dabp_d2f : public gr_sync_block
{
    private:
    friend dabp_d2f_sptr dabp_make_d2f();
    dabp_d2f();
    public:
    ~dabp_d2f();
    int work (int noutput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};
#endif // INCLUDED_DABP_D2F_H

