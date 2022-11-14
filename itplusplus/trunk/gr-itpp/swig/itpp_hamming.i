GR_SWIG_BLOCK_MAGIC(itpp,hamming_encoder_vbb);

itpp_hamming_encoder_vbb_sptr itpp_make_hamming_encoder_vbb (short m)
	throw(std::invalid_argument);

class itpp_hamming_encoder_vbb : public gr_sync_block
{
	private:
		itpp_hamming_encoder_vbb (short m);

	public:
		double get_rate();
		short get_n();
		short get_k();
};

GR_SWIG_BLOCK_MAGIC(itpp,hamming_decoder_vbb);

itpp_hamming_decoder_vbb_sptr itpp_make_hamming_decoder_vbb (short m)
	throw(std::invalid_argument);

class itpp_hamming_decoder_vbb : public gr_sync_block
{
  private:
	itpp_hamming_decoder_vbb (short m);

  public:
	double get_rate();
	short get_n();
	short get_k();
};

