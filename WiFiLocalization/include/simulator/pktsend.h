/** pktsend.h: Asynchronous packet sending (using a buffer) for the simulator
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
#ifndef NRL_PKTSEND_H
#define NRL_PKTSEND_H

#include <wx/socket.h>
#include <wx/thread.h>

#include "buffer.h"
#include "packet.h"

using namespace std;

//How long logger should wait between checks for packets
#define SENDER_QUEUE_DELAY	200
#define SENDER_BUFFER_CAPACITY	20

// A struct for storing information to be sent
class SenderPayload{
public:
  Packet* pkt;
  wxSocketBase* sock;
  
  SenderPayload(Packet* packet, wxSocketBase* socket);
};

///Note: The sender thread will call the socket without grabbing a mutex for it. 
///This is OK since the main thread knows when the sender is active and has a means to "blacklist" broken connection pointers.
///BUG: If a new connection is generated with a pointer that is on the blacklist, sending won't work. This is expected to be rare.
///BUG: There is no way to remove items from the blacklist without restarting the program. This results in worse performance as more connections are broken and re-established.
class PktSenderThread : public wxThread{
private:
  bool done;
  CondBuffer* buffer; //Note: Don't delete this, the main thread owns it!
  wxMutex blmutex;
  std::vector<wxSocketBase*> blacklist; //Previously existant connections that should be ignored because they were broken
  
  void sendViaSocket(SenderPayload* item);
  
public:
  PktSenderThread(CondBuffer* logbuf);
  
  void cancel(wxSocketBase* socket); //Gets called by main thread before socket destruction, hence the mutex for the blacklist.
  virtual void *Entry(); // thread execution
  //Function for waking up this thread when it's time to kill it.
  void signalDone(void);
};


///main thread wrapper for the sender thread and related functions
///NOTE: These functions eat (take ownership of) the data that they are given!
class PktSender{
private:
  PktSenderThread* thread;
  CondBuffer* buf;
  
public:
  PktSender(); ///Creates buffer and initializes sending thread
  ~PktSender(); /// Destroys buffer and sending thread
  
  ///Place packet into buffer for sending via socket
  int send(Packet* packet, wxSocketBase* socket);
  int sendGroup(std::vector<Packet*> packets, wxSocketBase* socket);
  
  ///Cancels all pending writes using this socket (for when the connection breaks and so on)
  void cancel(wxSocketBase* socket);
  ///Tells thread to quit
  void signalDone(void);
};


#endif
