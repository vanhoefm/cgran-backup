/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_d2h_c_cuda from python as grgpu.d2h_c_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,d2h_c_cuda);

grgpu_d2h_c_cuda_sptr grgpu_make_d2h_c_cuda ();

class grgpu_d2h_c_cuda : public gr_block
{
 public:
  void set_verbose (int verbose);
private:
  grgpu_d2h_c_cuda ();
};
