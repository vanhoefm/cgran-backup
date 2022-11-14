/* -*- c++ -*- */
/*
 * Copyright 2009 Free Software Foundation, Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gcell.h"
#include <algorithm>
#include <stdio.h>


static int s_refcnt = 0;
static int s_user_max_spes = -1;	// user specified max # of spes to use (-1 to use them all)
static int s_max_spes = -1;		// actual max # of usable spes (-1 answer currently unknown)
static gc_job_manager_sptr s_mgr;	// shared pointer to gcell gc_job_manager
static bool s_we_created_mgr = false;

extern spe_program_handle_t X(spu_fftw_gcell);
static gc_proc_id_t s_fftw_proc_id = GCP_UNKNOWN_PROC;

#if defined(FFTW_SINGLE)      
# define FFTW_PROC_NAME "fftwf"
#elif defined(FFTW_LDOUBLE)
# define FFTW_PROC_NAME "fftwl"
#else
# define FFTW_PROC_NAME "fftw"
#endif


extern "C" void *X(cell_aligned_malloc)(size_t n)
{
  void *p;
  posix_memalign(&p, 128, n);
  return p;
}

extern "C" int X(cell_nspe)(void)
{
  return s_max_spes;
}

extern "C" void X(cell_set_nspe)(int n)
{
  s_user_max_spes = n;
}

extern "C" void X(cell_activate_spes)(void)
{
  if (s_refcnt++ == 0){		// FIXME needs mutex
    gc_job_manager_sptr mgr;

    try {
      mgr = gc_job_manager::singleton();
      s_we_created_mgr = false;
    }
    catch (...){
      // No job manager currently registered.  Create one.
      gc_jm_options opts;
      opts.program_handle = gc_program_handle_from_address(&X(spu_fftw_gcell));
      if (s_user_max_spes >= 0)
	opts.nspes = s_user_max_spes;
      mgr = gc_make_job_manager(&opts);
      s_we_created_mgr = true;
    }

    if ((s_fftw_proc_id = mgr->lookup_proc(FFTW_PROC_NAME)) == GCP_UNKNOWN_PROC){
      // Our code isn't loaded into the SPEs...
      s_max_spes = 0;
    }
    else if (s_user_max_spes >= 0)
      s_max_spes = std::min(s_user_max_spes, mgr->nspes());
    else
      s_max_spes = mgr->nspes();

    s_mgr = mgr;
  }
}

extern "C" void X(cell_deactivate_spes)(void)
{
  if (--s_refcnt == 0){		// FIXME needs mutex
    if (s_we_created_mgr){
      s_we_created_mgr = false;
      s_mgr->shutdown();
    }
    s_mgr.reset();	// If nobody else is using it, the job manager will be destroyed.
    s_max_spes = -1;
  }
}

static gc_job_desc *
gcp_fftw(gc_job_manager_sptr mgr, struct spu_context *ctx)
{
  gc_job_desc *jd = mgr->alloc_job_desc();
  jd->proc_id = s_fftw_proc_id;
  jd->input.nargs = 0;
  jd->output.nargs = 0;
  jd->eaa.nargs = 1;

  jd->eaa.arg[0].ea_addr = ptr_to_ea(ctx);
  jd->eaa.arg[0].direction = GCJD_DMA_GET;
  jd->eaa.arg[0].get_size = sizeof(*ctx);

  if (!mgr->submit_job(jd)){
    gc_job_status_t s = jd->status;
    mgr->free_job_desc(jd);
    throw gc_bad_submit("fftw", s);
  }
  return jd;
}


/*
 * alloc, build, submit, wait
 */
extern "C" void X(cell_spe_run_wait)(int njobs, struct spu_context *ctx)
{
  gc_job_desc *jobs[njobs];
  bool done[njobs];
  int  n;

  try {
    for (n = 0; n < njobs; n++)
      jobs[n] = gcp_fftw(s_mgr, &ctx[n]);

    int ndone = s_mgr->wait_jobs(njobs, jobs, done, GC_WAIT_ALL);
    if (ndone != njobs){
      fprintf(stderr, "fftw_cell_spu_run_wait: wait_jobs: expected %d, got %d\n", njobs, ndone);
    }
  }
  catch(...){
    fprintf(stderr, "fftw_cell_spu_run_wait: exception\n");
  }

  for (int j = 0; j < n; j++)
    s_mgr->free_job_desc(jobs[j]);
}
