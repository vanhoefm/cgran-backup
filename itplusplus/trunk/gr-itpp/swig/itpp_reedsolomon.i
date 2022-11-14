GR_SWIG_BLOCK_MAGIC(itpp,reedsolomon_encoder_vbb);

itpp_reedsolomon_encoder_vbb_sptr itpp_make_reedsolomon_encoder_vbb (int m, int t, bool systematic = false)
	throw(std::invalid_argument);

class itpp_reedsolomon_encoder_vbb : public gr_sync_block
{
  private:
	itpp_reedsolomon_encoder_vbb (int m, int t, bool systematic);

  public:
	double get_rate() const;
	int get_n() const;
	int get_k() const;
};

GR_SWIG_BLOCK_MAGIC(itpp,reedsolomon_decoder_vbb);

itpp_reedsolomon_decoder_vbb_sptr itpp_make_reedsolomon_decoder_vbb (int m, int t, bool systematic = false)
	throw(std::invalid_argument);

class itpp_reedsolomon_decoder_vbb : public gr_sync_block
{
  private:
	itpp_reedsolomon_decoder_vbb (int m, int t, bool systematic);

  public:
	double get_rate() const;
	int get_n() const;
	int get_k() const;
};
