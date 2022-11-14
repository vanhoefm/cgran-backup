/*
 * Copyright (c) 2007 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <ifftw.h>

#ifdef FFTW_SINGLE 
# ifdef HAVE_GCELL	/* FIXME until we sort out spe memory mgmt */
#  define MAX_N 4096
# else
#  define MAX_N 8192
# endif
# define REQUIRE_N_MULTIPLE_OF 4
# define VL 2
#else
# ifdef HAVE_GCELL
#  define MAX_N 2048
# else
#  define MAX_N 4096
# endif
# define REQUIRE_N_MULTIPLE_OF 1
# define VL 1
#endif

#define MAX_PLAN_LEN 5
#define MAX_NSPE 16

struct spu_radices {
     signed char r[MAX_PLAN_LEN];
};

struct cell_iodim {
     int n0;  /* lower bound, n0 <= n */
     int n1;  /* upper bound, n < n1 */

     /* strides expressed in bytes  */
     int is_bytes;
     int os_bytes;

     /* derivative of M w.r.t. this dimension, for twiddle problems */
     int dm;
};

struct dft_context {
     struct spu_radices r;
     /* strides expressed in bytes  */
     int n, is_bytes, os_bytes;
     struct cell_iodim v[2];
     int sign;
     int Wsz_bytes;

     /* pointers, converted to ulonglong */
     unsigned long long xi, xo;
     unsigned long long W;

     /* optional twiddles for dftw */
     unsigned long long Ww;
};

struct transpose_context {
     unsigned long long A;
     int n;
     int s0_bytes;
     int s1_bytes;
     int nspe;
     int my_id;
};

struct copy_context {
     unsigned long long I;
     unsigned long long O;
     int n, v;
     int is_bytes, os_bytes;
     int ivs_bytes, ovs_bytes;
     int nspe;
     int my_id;
};

/* operations that the SPE's can execute */

enum spu_op {
     FFTW_SPE_DFT,
     FFTW_SPE_TRANSPOSE,
     FFTW_SPE_COPY,
     FFTW_SPE_EXIT
};

struct spu_context {
     union spu_context_u {
	  struct dft_context dft;
	  struct transpose_context transpose;
	  struct copy_context copy;
	  /* possibly others */
     } u;

     char pad[15 -
	      (((sizeof(union spu_context_u) + 
		 sizeof(enum spu_op)
		 + sizeof(int)) - 1) 
	       & 15)];
     volatile int done;
     enum spu_op op;
};

extern const struct spu_radices 
   X(spu_radices)[(MAX_N/REQUIRE_N_MULTIPLE_OF) + 1];

void X(dft_direct_cell_register)(planner *p);
void X(ct_cell_direct_register)(planner *p);

void *X(cell_aligned_malloc)(size_t n);

void X(cell_activate_spes)(void);
void X(cell_deactivate_spes)(void);

int X(cell_nspe)(void);
void X(cell_set_nspe)(int n);

#ifdef HAVE_GCELL
#define _GC_ALIGN(ptr, size) ((((INT) ptr) + (size)-1) & ~((size)-1))
#define LOCAL_SPU_CONTEXT \
  char _buf[128 + sizeof(struct spu_context[MAX_NSPE])]; \
  struct spu_context *_ctx = (struct spu_context *) _GC_ALIGN(_buf, 128)
#define	CELL_GET_CTX(i) &_ctx[i]
#define CELL_SPE_RUN_WAIT(nspes) X(cell_spe_run_wait((nspes), _ctx))
#else
#define LOCAL_SPU_CONTEXT
#define CELL_GET_CTX(i) X(cell_get_ctx(i))
#define CELL_SPE_RUN_WAIT(nspes) X(cell_spe_run_wait((nspes), 0))
struct spu_context *X(cell_get_ctx)(int spe);
#endif

void X(cell_spe_run_wait)(int nspes, struct spu_context *ctx);

#define FITS_IN_INT(x) ((x) == (int)(x))
