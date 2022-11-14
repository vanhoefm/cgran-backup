#ifndef INCLUDED_TAIT_flat_rayleigh_channel_cc_H
#define INCLUDED_TAIT_flat_rayleigh_channel_cc_H

#include <gr_sync_block.h>
#include <flat_rayleigh.h>

/*

This class implements a flat Rayleigh Fading Simulator based on Christos Komninakis work referred to in his paper "A Fast and Accurate Rayleigh Fading Simulator" (http://www.ee.ucla.edu/~chkomn/). Some analysis has been carried out by myself confirming that his algorithm is similar to the Rayleigh model available within Matlab "rayleighchan()".


Inputs

- seed: the random channel seed (e.g. -115).
- fdT: Discrete Doppler Rate (small positive fraction, e.g. 0.001 - 0.2), where fd is the Doppler fade frequency and 1/T is the sample rate of the channel. Note that the current implementation only supports rational fractions of 0.2. To accommodate other values the IIR filter needs to re-designed using Matlab (please reefer to the paper mentioned above for details).
- pwr: the square root power of the resulting fading waveform (e.g. pwr = 5 would produce a fading waveform with an output power of 25).
- flag_indep: Determines whether individual blocks processed by the channel should be treated as independent blocks or as one continuous block. 1 if blocks are independent, 0 if continuous across blocks.

	
Jonas Hodel
01/02/07
	
*/	
class tait_flat_rayleigh_channel_cc;

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
typedef boost::shared_ptr<tait_flat_rayleigh_channel_cc> tait_flat_rayleigh_channel_cc_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of tait_flat_rayleigh_channel_cc.
 *
 * To avoid accidental use of raw pointers, tait_flat_rayleigh_channel_cc's
 * constructor is private.  tait_make_flat_rayleigh_channel_cc is the public
 * interface for creating new instances.
 */
tait_flat_rayleigh_channel_cc_sptr tait_make_flat_rayleigh_channel_cc 
(int seeed, float fD, float pwr, bool flag_indep);

/*!
 * \brief Applies flat Rayleigh channel fading on a stream of complex samples.
 * \ingroup tait
 *
 */
class tait_flat_rayleigh_channel_cc : public gr_sync_block
{
private:
	// The friend declaration allows tait_make_flat_rayleigh_channel_cc to
	// access the private constructor.
	
	friend tait_flat_rayleigh_channel_cc_sptr tait_make_flat_rayleigh_channel_cc 
	(int seeed, float fD, float pwr, bool flag_indep);
	
	/*!
	* \brief Constructor
	*
	* \p seed: 		The random channel seed (e.g. -115).
	* \p fdT: 		Discrete Doppler Rate (small positive fraction, e.g. 0.001 - 0.2), 
	*			where fd is the Doppler fade frequency and 1/T is the sample rate of 
	*			the channel. Note that the current implementation only supports 
	*			rational fractions of 0.2. To accommodate other values the IIR 
	*			filter needs to re-designed using Matlab (please reefer to 
	*			the paper mentioned above for details).
	* \p pwr: 		The square root power of the resulting fading waveform 
	*			(e.g. pwr = 5 would produce a fading waveform with an 
	*			output power of 25).
	* \p flag_indep: 	Determines whether individual blocks processed by the channel should 
	*			be treated as independent blocks or as one continuous block.
	*			1 if blocks are independent, 0 if continuous across blocks
	*/
	tait_flat_rayleigh_channel_cc (int seeed, float fD, float pwr, bool flag_indep);  	// private constructor
	
	// The flat Rayleigh Fading Chanel.
	flat_rayleigh *mychan;
	
	public:
	~tait_flat_rayleigh_channel_cc ();	// public destructor
	
	// Where all the action really happens
	
	int work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items);
};

#endif /* INCLUDED_TAIT_flat_rayleigh_channel_cc_H */
