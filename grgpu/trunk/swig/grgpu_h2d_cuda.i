/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_h2d_cuda from python as grgpu.h2d_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,h2d_cuda);

grgpu_h2d_cuda_sptr grgpu_make_h2d_cuda ();

class grgpu_h2d_cuda : public gr_block
{
 public:
  void set_verbose (int verbose);
private:
  grgpu_h2d_cuda ();
};
