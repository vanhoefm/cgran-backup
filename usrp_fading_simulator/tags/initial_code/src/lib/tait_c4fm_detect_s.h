#ifndef INCLUDED_TAIT_C4FM_DETECT_H
#define INCLUDED_TAIT_C4FM_DETECT_H

#include <gr_sync_block.h>
#include <stdio.h>

extern "C" {
	void spapmd_ProcessC4FMDetect(int16_t *input, int16_t *output, uint16_t NumOfSamples);
	void spapmd_C4FMDetectReset(void);
	void spapmd_C4FMDetectConfigure(uint16_t aveCount);
	void callDSP_func(int16_t *in, int16_t *out, uint16_t NumOfSamples,
			  void (*DSP_func)(int16_t*, int16_t*, uint16_t),
			  int req_num_inputs);
}

class tait_c4fm_detect_s;

/*
 * We use boost::shared_ptr's instead of raw pointers for all access
 * to gr_blocks (and many other data structures).  The shared_ptr gets
 * us transparent reference counting, which greatly simplifies storage
 * management issues.  This is especially helpful in our hybrid
 * C++ / Python system.
 *
 * See http://www.boost.org/libs/smart_ptr/smart_ptr.htm
 *
 * As a convention, the _sptr suffix indicates a boost::shared_ptr
 */
typedef boost::shared_ptr<tait_c4fm_detect_s> tait_c4fm_detect_s_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of tait_c4fm_detect_s.
 *
 * To avoid accidental use of raw pointers, tait_c4fm_detect_s's
 * constructor is private.  tait_make_c4fm_detect_s is the public
 * interface for creating new instances.
 */
tait_c4fm_detect_s_sptr tait_make_c4fm_detect_s (/*uint16_t averageCount*/);

/*!
 * \brief C4FM detection block
 * \ingroup block
 *
 * This uses the preferred technique: subclassing gr_sync_block.
 */
class tait_c4fm_detect_s : public gr_sync_block
{
	private:
	// The friend declaration allows tait_make_c4fm_detect_s to
	// access the private constructor.
	
		friend tait_c4fm_detect_s_sptr tait_make_c4fm_detect_s (/*uint16_t averageCount*/);
	
		tait_c4fm_detect_s (/*uint16_t averageCount*/);	// private constructor
		uint16_t conf;

	public:
		~tait_c4fm_detect_s ();	// public destructor

  // Where all the action really happens

		int work (int noutput_items,
			  gr_vector_const_void_star &input_items,
			  gr_vector_void_star &output_items);
};

#endif


