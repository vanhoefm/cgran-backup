/** atomic.h: A user-space atomic variable using pthread_mutexes
 * 
 * @author Brian Shaw
 * 
 */
/* 
 * This file is part of WiFi Localization
 * 
 * This program is free software; you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation; either version 2 of the License, or 
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
 * for more details.
 * 
 * You should have received a copy of the GNU General Public License along 
 * with this program; if not, write to the Free Software Foundation, Inc., 
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef NRL_ATOMIC_H
#define NRL_ATOMIC_H

#include <pthread.h>

using namespace std;

//Mutex-based implementation of an atomic integer.
class Atomic{
  pthread_mutex_t lock;
  int myval;
  
public:
  Atomic(int);
  ~Atomic();
  void setval(int);
  int getval(void);
};

#endif