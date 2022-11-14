/** sensorsocket.h: Definitions for the socket code
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

#ifndef SOLVER_SOCKET_H
#define SOLVER_SOCKET_H

//Designated local path and port number to use
//NOTE: Do not use the local socket, it works but wxWidgets won't support it.
#define LOCAL_SERVER_PATH "/tmp/nrlmqp_localization_solver_server"
#define IS_SOLVER_LOCAL false

//A class representing a connection to the solver
//(designed to be really simple to create and destroy)
//Constructor initializes the connection
//Destructor closes the connection
class TCPClientConnection{
  int sockfd, portno;
  bool valid, is_local;
  char servname[41]; //Takes up to 40 characters of server name
 
  bool makeconnection(void);
  
public:
  //constructors
  TCPClientConnection(char* target, int port, bool islocal, bool allow_failure);
  TCPClientConnection(bool islocal, bool allow_failure);
  //destructor
  ~TCPClientConnection(void);
  
  bool isConnected(void);
  void start(bool allow_failure);
  void stop(void);
  
  int sendItem(void* item, size_t size);
};

/*------------Prototypes-----------------*/

void send_over_link(char* item, long size, TCPClientConnection* link);

#endif
