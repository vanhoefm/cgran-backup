/** sensor_threads.h: Sender and Logger thread functionality for sensor
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

#ifndef NRL_SENSOR_THREADS_H
#define NRL_SENSOR_THREADS_H

#include "atomic.h"
#include "pktbuffer.h"
#include "sensorsocket.h"

#include <pthread.h>

using namespace std;


// A structure to be passed to the solver when it is initialized
struct solverparams{
  Pktbuffer* packetbuf;
  TCPClientConnection* link;
};

void* pktsender_thread(void* paramstruct);
void* logger_thread(void* arg);

extern Atomic* istimetodie;

#endif