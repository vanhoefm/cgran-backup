/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_fft_vfc_cuda from python as grgpu.fft_vfc_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,fft_vfc_cuda);

grgpu_fft_vfc_cuda_sptr grgpu_make_fft_vfc_cuda ();

class grgpu_fft_vfc_cuda : public gr_block
{
private:
  grgpu_fft_vfc_cuda ();
};
