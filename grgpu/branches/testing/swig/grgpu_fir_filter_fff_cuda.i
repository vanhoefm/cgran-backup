/*
 * First arg is the package prefix.
 * Second arg is the name of the class minus the prefix.
 *
 * This does some behind-the-scenes magic so we can
 * access grgpu_fir_filter_fff_cuda from python as grgpu.fir_filter_fff_cuda
 */
GR_SWIG_BLOCK_MAGIC(grgpu,fir_filter_fff_cuda);

grgpu_fir_filter_fff_cuda_sptr grgpu_make_fir_filter_fff_cuda (const std::vector<float> &taps);

class grgpu_fir_filter_fff_cuda : public gr_block
{
 public:
  void set_verbose (int verbose);
private:
  grgpu_fir_filter_fff_cuda (const std::vector<float> &taps);
};
