#ifndef INCLUDED_TAIT_EXAMPLE_FF_H
#define INCLUDED_TAIT_EXAMPLE_FF_H

#include <gr_sync_block.h>

// This is how your C functions should be declared.
extern "C" 
{
	void my_c_function();
}

class tait_example_ff;

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
typedef boost::shared_ptr<tait_example_ff> tait_example_ff_sptr;

/*!
* \brief Return a shared_ptr to a new instance of tait_example_ff.
*
* To avoid accidental use of raw pointers, tait_example_ff's
* constructor is private.  tait_make_example_ff is the public
* interface for creating new instances.
*/
tait_example_ff_sptr tait_make_example_ff ();

/*!
* \brief square a stream of floats.
* \ingroup block
*
* This uses the preferred technique: subclassing gr_sync_block.
*/
class tait_example_ff : public gr_sync_block
{
private:
// The friend declaration allows tait_make_example_ff to
// access the private constructor.

friend tait_example_ff_sptr tait_make_example_ff ();

	tait_example_ff ();  	// private constructor

public:
	~tait_example_ff ();	// public destructor

// Where all the action really happens

int work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items);
};

#endif /* INCLUDED_TAIT_EXAMPLE_FF_H */
