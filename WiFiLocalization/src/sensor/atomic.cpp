/** atomic.cpp: A user-space atomic variable using pthread_mutexes
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

#include "atomic.h"

using namespace std;

/*----------Atomic class implementation---------*/

Atomic::Atomic(int val){
  pthread_mutex_init(&lock, NULL);
  myval = val;
}

Atomic::~Atomic(){
  pthread_mutex_destroy(&lock);
}

void Atomic::setval(int val){
  pthread_mutex_lock(&lock);
  myval = val;
  pthread_mutex_unlock(&lock);
  return;
}

int Atomic::getval(void){
  pthread_mutex_lock(&lock);
  int val = myval;
  pthread_mutex_unlock(&lock);
  return val;
}