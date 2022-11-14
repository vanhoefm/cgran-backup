/** pktbuffer.cpp: Buffer for logging function (no WxWidgets)
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

using namespace std;

#include "pktbuffer.h"
#include "sensor_globals.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/*-----------------Implementation of Packet Buffer--------------------*/

Pktbuffer::Pktbuffer(int cpy){
  //Capacity is required, 0 indicates "use default"
  if (cpy > 1000 || cpy <= 0)
    capacity = BBNSENDER_PKTBUF_SIZE;
  else
    capacity = cpy;
  
  //initialize mutex
  pthread_mutex_init(&lock, NULL);
}

Pktbuffer::~Pktbuffer(void){
  //Destructor will remove all packets from the buffer!
  //Warning: This does not check to see if any other thread is using the buffer!
  //It is wise to kill the threads that need this buffer first (especially the consumer thread)
  
  unsigned int a;
  for (a = 0; a < pktarray.size(); a++){
    if (pktarray[a] != NULL) {
      printf("There was a non-NULL packet in the buffer when flushed!\n");
      delete pktarray[a];
    }
  }
  //printf("Made it to destroy lock\n");
  pthread_mutex_destroy(&lock);
  //printf("Made it past destroy lock\n");
} 

//Grabs the buffer's mutex
int Pktbuffer::getMutex(bool blocking){
  //remember to let go of the mutex!
  int err;
  if (blocking){
    err = pthread_mutex_lock(&lock);
    if (err) {
      perror("Error obtaining mutex (blocking):");
      exit(1);
    }
  } else {
    err = pthread_mutex_trylock(&lock);
    if (err){
      if (err == EAGAIN){
	printf("Buffer in use, exiting\n");
	return 1;
      } else {
	perror("Error obtaining mutex (nonblocking):");
	exit(1);
      }
    }
  }
  return 0; //success
}
  
//Note: In this current implementation, the mutex is acquired twice
//if that becomes a problem, come up with a private version of is_full
//that acquires the mutex but does not release it unless there is an error
int Pktbuffer::add_packet(Packet* pkt, bool blocking){
  //printf("Adding packet! or trying...?\n");
  
  if (pkt == NULL){
    printf("Warning! Trying to add null packet\n");
    return 1;
  }
  
  if (getMutex(blocking))
    return 1;
  // CRITICAL SECTION!

  if (pktarray.size() >= capacity){
    pthread_mutex_unlock(&lock);
    printf("Warning! Packet buffer is full. Dropping packet.\n");
    delete pkt;
    return 2;
  }
  
  pktarray.push_back(pkt);
  pthread_mutex_unlock(&lock);
  // ~CRITICAL SECTION!
  return 0;
}


//Retrieve and remove a packet from the buffer
//Called by the consumer thread
Packet* Pktbuffer::get_packet(bool blocking){
  
  //printf("Calling get_packet\n");
  
  if (getMutex(blocking))
    return NULL;
  // CRITICAL SECTION!
    
  if (pktarray.size() == 0){
    //No packets to be taken
    pthread_mutex_unlock(&lock);
    return NULL;
  }
  //Grab packet
  Packet* tmppkt = pktarray[0]->copy();
  pktarray.pop_front();
  pthread_mutex_unlock(&lock);
  // ~CRITICAL SECTION!
  
  //printf("About to return PKT# %ld\n", (long int) tmppkt);
  return tmppkt;
}

