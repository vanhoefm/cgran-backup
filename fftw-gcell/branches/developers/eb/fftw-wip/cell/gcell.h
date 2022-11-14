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

#ifndef INCLUDED_FFTW_GCELL_H
#define INCLUDED_FFTW_GCELL_H

#include <gcell/gc_job_manager.h>
#include <gcell/gc_job_desc.h>

extern "C" {
#include "ifftw.h"
#include "fftw-cell.h"
}

// gc_job_manager_sptr X(gcell_job_manager)();
// gc_proc_id_t X(gcell_proc_id)();

#endif /* INCLUDED_FFTW_GCELL_H */
