/* -*- c++ -*- */
/*
 * Copyright 2005 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

class bbn_tap;
typedef boost::shared_ptr<bbn_tap> bbn_tap_sptr;
%template(bbn_tap_sptr) boost::shared_ptr<bbn_tap>;

bbn_tap_sptr bbn_make_tap(std::string dev_name, int freq=2437);

/*!
 * \brief thread-safe message queue
 */
%ignore bbn_tap;
class bbn_tap  {

public:
  bbn_tap(std::string dev_name, int freq);
  ~bbn_tap();
  int fd();
  std::string name();
  int tap_write(const std::string buf);
  int tap_read_fd();
  std::string tap_process_tx(std::string buf);
};
