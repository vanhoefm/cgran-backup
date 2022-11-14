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
#include "pktsend.h"

using namespace std;

SenderPayload::SenderPayload(Packet* packet, wxSocketBase* socket){
  pkt = packet;
  sock = socket;
}

/*-------PktSenderThread implementation---------*/

PktSenderThread::PktSenderThread(CondBuffer* buf) : wxThread(wxTHREAD_JOINABLE){
  buffer = buf;
  done = false;
}


void* PktSenderThread::Entry(void){
  ///Actual thread execution
  
  cout << "INFO: sender thread running" << endl;
  
  int a;
  std::deque<void*>* items;
  
  while (!done){
    
    //cout << "Sender Attempting remove" << endl;
    //Grab everything in the buffer
    items = buffer->remove(SENDER_QUEUE_DELAY, This(), false); //Blocks when buffer is empty
    
    //cout << "Sender Remove succeeded" << endl;
    
    try {
      //cout << "Sender remove: items = " << items << endl;
      if (items != NULL){ //NULL gets returned when it's time to exit
	//Log the items
	//cout << "Items->size() is " << items->size() << endl;
	for (a = 0; a < items->size(); a++){
	  SenderPayload* item = (SenderPayload*) (*items)[a];
	  //cout << "Sending packet " << item->pkt << " via socket " << item->sock << endl;
	  if (item->pkt == NULL)
	    throw "Packet sender got NULL item pkt";
	  //else
	  //  item->pkt->print(&cout);
	  if (item->sock == NULL)
	    throw "Packet sender got NULL socket";
	  sendViaSocket(item);
	  delete item->pkt;
	  delete item; //Deletes the struct but not the socket
	}
	delete items;
      }
    } catch (const char* err){
      cout << "Sending thread error: " << err << endl;
    } catch (...){
      cout << "Other error in sending thread\n";
    }
    
    done = done || TestDestroy(); //REQUIRED: Check to see if the thread needs to exit   
  }
  
  cout << "Exiting sender thread" << endl;
  //Exit and join with main thread
  return NULL;
}


///Note: Errors are thrown as wxSocketError
void PktSenderThread::sendViaSocket(SenderPayload* item){
  wxMutexLocker lock(blmutex);
  int b;
  bool isblacklisted = false;
  for (b = 0; b < blacklist.size(); b++)
    isblacklisted = isblacklisted || (blacklist[b] == item->sock);
  
  if (isblacklisted){
    cout << "Item was on the blacklist!" << endl;
    return;
  }
  
  wxSocketBase* sock = item->sock;
  char buf[256];
  //cout << "SendViaSocket about to serialize" << endl;
  int len = item->pkt->serialize(buf, 256);
  
  if (sock == NULL)
    throw "Cannot send via a NULL socket!";
  
  //cout << "SendViaSocket about to write!" << endl;
  
  sock->SetFlags(wxSOCKET_WAITALL);
  sock->Write(buf, len);
  
  //cout << "Successfully wrote to socket" << endl; //Not technically true, may have had error
  
  //Sent. Check for errors.
  if (!sock->Error())
    return;
  //Note: This will occur when sockets get invalidated. Handle transparently.
  cout << "Unable to write socket (was a connection broken?): ";
  wxSocketError lasterr = sock->LastError();
  cout << lasterr << endl;
  return;
}
  

void PktSenderThread::cancel(wxSocketBase* socket){
  wxMutexLocker lock(blmutex);
  blacklist.push_back(socket);
}

void PktSenderThread::signalDone(void){
  done = true;
  buffer->signalDone();
}


/*-------PktSender implementation-----------*/


PktSender::PktSender(void){
  ///Creates buffer and initializes sending thread
  buf = new CondBuffer(SENDER_BUFFER_CAPACITY);
  thread = new PktSenderThread(buf);
  wxThreadError err = thread->Create();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to create sender thread\n";
  err = thread->Run();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to run sender thread\n";
}


PktSender::~PktSender(void){
  /// Destroys buffer and sending thread
  thread->signalDone();
  thread->Wait();
  delete thread;
  delete buf;
}


int PktSender::send(Packet* packet, wxSocketBase* socket){
  //NOTE: This no longer takes ownership of the original
  SenderPayload* payload = new SenderPayload(packet->copy(), socket);
  return buf->addone((void*) payload);
}


int PktSender::sendGroup(std::vector<Packet*> packets, wxSocketBase* socket){
  std::vector<void*> vec;
  vec.reserve(packets.size());
  SenderPayload* payload;
  int a;
  for (a = 0; a < packets.size(); a++){
    if (packets[a] != NULL){ //when simulator simulates a packet drop, NULL appears
      payload = new SenderPayload(packets[a]->copy(), socket);
      //cout << "Placing outbound packet in the buffer" << endl;
      vec.push_back((void*) payload);
    }
  }
  return buf->add(vec);
}

void PktSender::signalDone(void){
  thread->signalDone();
}

void PktSender::cancel(wxSocketBase* sock){
  thread->cancel(sock);
}

