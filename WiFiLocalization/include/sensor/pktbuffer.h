/** pktbuffer.h: Buffer for logging function (no WxWidgets)
 * 
 * It's a FIFO queue protected by a mutex
 * with the ability to detect when it's overwhelmed
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

#ifndef NRL_PKTBUFFER_H
#define NRL_PKTBUFFER_H

#include <pthread.h>
#include <deque>

#include "packet.h" //can easily be eliminated

using namespace std;

class Pktbuffer{
  unsigned int capacity;
  pthread_mutex_t lock;
  //FIFO queue of packets
  deque<Packet*> pktarray;
  int getMutex(bool blocking);
  
public:
  Pktbuffer(int cpy); //Capacity is required, 0 indicates "use default"
  ~Pktbuffer(void); //Destructor will remove all packets from the buffer!
  
  int add_packet(Packet* pkt, bool blocking);
  Packet* get_packet(bool blocking);
};

#endif