GR_SWIG_BLOCK_MAGIC(itpp,egolay_encoder_vbb);

itpp_egolay_encoder_vbb_sptr itpp_make_egolay_encoder_vbb ();

class itpp_egolay_encoder_vbb : public gr_sync_block
{
  private:
	itpp_egolay_encoder_vbb ();

  public:
	double get_rate() const;
};

GR_SWIG_BLOCK_MAGIC(itpp,egolay_decoder_vbb);

itpp_egolay_decoder_vbb_sptr itpp_make_egolay_decoder_vbb ();

class itpp_egolay_decoder_vbb : public gr_sync_block
{
  private:
	itpp_egolay_decoder_vbb ();

  public:
	double get_rate() const;
};

