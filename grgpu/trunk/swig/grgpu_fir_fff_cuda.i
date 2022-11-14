/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_fir_fff_cuda from python as grgpu.fir_fff_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,fir_fff_cuda);

grgpu_fir_fff_cuda_sptr grgpu_make_fir_fff_cuda ();

class grgpu_fir_fff_cuda : public gr_block
{
private:
  grgpu_fir_fff_cuda ();
};
