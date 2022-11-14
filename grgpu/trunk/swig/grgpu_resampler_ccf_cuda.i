/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_resampler_ccf_cuda from python as grgpu.resampler_ccf_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,resampler_ccf_cuda);

grgpu_resampler_ccf_cuda_sptr grgpu_make_resampler_ccf_cuda ();

class grgpu_resampler_ccf_cuda : public gr_block
{
private:
  grgpu_resampler_ccf_cuda ();
};
