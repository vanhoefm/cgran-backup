GR_SWIG_BLOCK_MAGIC(itpp,besselj_ff);

itpp_besselj_ff_sptr itpp_make_besselj_ff (int order = 0)
	throw(std::invalid_argument);

class itpp_besselj_ff : public gr_sync_block
{
  private:
	itpp_besselj_ff (int order);

 public:
	bool set_order(int order);
	int order();
};


GR_SWIG_BLOCK_MAGIC(itpp,besseli_ff);

itpp_besseli_ff_sptr itpp_make_besseli_ff (int order = 0)
	throw(std::invalid_argument);

class itpp_besseli_ff : public gr_sync_block
{
  private:
	itpp_besseli_ff (int order);

 public:
	bool set_order(int order);
	int order();
};

