#ifndef INCLUDED_TAIT_socket_encode_schar_H
#define INCLUDED_TAIT_socket_encode_schar_H

#include <gr_sync_interpolator.h>

using namespace std;

class tait_socket_encode_schar;

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
typedef boost::shared_ptr<tait_socket_encode_schar> tait_socket_encode_schar_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of tait_socket_encode_schar.
 *
 * To avoid accidental use of raw pointers, tait_socket_encode_schar's
 * constructor is private.  tait_make_socket_encode_schar is the public
 * interface for creating new instances.
 */
tait_socket_encode_schar_sptr tait_make_socket_encode_schar ();

/*!
 * \brief turns a stream of short ints into a stream of chars (1 int becomes 2 chars).
 * \ingroup block
 *
 * Subclassing gr_sync_interpolator as a 1:4 input to output relationship exists.
 */
class tait_socket_encode_schar : public gr_sync_interpolator
{
private:
  // The friend declaration allows tait_make_socket_encode_schar to
  // access the private constructor.

  friend tait_socket_encode_schar_sptr tait_make_socket_encode_schar ();

  tait_socket_encode_schar ();  	// private constructor

 public:
  ~tait_socket_encode_schar ();	// public destructor

  // Where all the action really happens

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_TAIT_socket_encode_schar_H */
