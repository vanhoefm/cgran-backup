/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_mul_const_ff_cuda from python as grgpu.mul_const_ff_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,mul_const_ff_cuda);

grgpu_mul_const_ff_cuda_sptr grgpu_make_mul_const_ff_cuda ();

class grgpu_mul_const_ff_cuda : public gr_block
{
private:
  grgpu_mul_const_ff_cuda ();
};
