/** sensorsocket.cpp: Socket class implementations for localization solver
 * 
 * Everything here is just a C++ wrapper for basic socket code
 * (implementing the functionality needed by the localization system)
 * 
 * @author Brian Shaw
 * 
 * Limitations:
 * 
 * Local path always uses hardcoded default right now. 
 * Note: rm /tmp/nrlmqp_localization_solver_server will resolve the issue of that path being unavailable
 * 
 * Other notes:
 * 
 * Why port 5749? That's WI in ASCII, in hex.
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

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "sensorsocket.h"
#include "sensor_globals.h"

using namespace std;

//Note: Sockets block when sending

///Send a serialized item over a TCPClientConnection
///ensures the entire item gets sent
void send_over_link(char* item, long size, TCPClientConnection* link){
  
  char buf[256];
  bzero(buf, 256 * sizeof(char));
  memcpy(buf, item, size);
  
  int len = size;
  while (len > 0)
    len = link->sendItem(buf, len);
  
  if (len < 0)
    perror("Warning: Problem sending packet! This may corrupt data at the solver.");
}


/******Client Connection Implementation*********/

//Check if a port number is valid for use with the solver
// (port is between 5000 and 65535)
bool port_is_invalid(int port){
  if (port < 2000)
    return true;
  if (port > 65535)
    return true;
  return false;
}


//Create a TCP Connection
//target is the server being connected to
//islocal is a boolean for whether or not the server is on the same computer
TCPClientConnection::TCPClientConnection(char* target, int port, bool islocal, bool allow_failure){
  //Sanity checking arguments
  if (!islocal && port_is_invalid(port)){
    printf("ERROR: Port %d is not valid! Cannot make connection\n", port);
    exit(1);
  }
  
  //Save connection information in case we try to reconnect later
  strncpy(servname, target, 40);
  servname[40] = '\0';
  if (islocal)
    portno = 0;
  else {
    portno = port;
    if (portno == 0)
      portno = DEFAULT_PORT;
  }
  is_local = islocal;
  valid = false;
  
  //Attempt to establish TCP connection
  start(allow_failure);
}


//Constructor ONLY for local connections
//islocal should be true if you want a local connection
TCPClientConnection::TCPClientConnection(bool islocal, bool allow_failure){
  //This is only included as an idiot-resistance feature
  if (!islocal){
    printf("ERROR: Can't make Internet socket without host name and port number!\n");
    exit(1);
  }
  
  //Save connection information in case we try to reconnect later
  is_local = islocal;
  valid = false;
  
  //Attempt to establish TCP connection
  start(allow_failure);
}  


//Open a TCPConnection, externally callable and used by constructor
void TCPClientConnection::start(bool allow_failure){
  //Attempt to establish TCP connection
  bool failure = !makeconnection();
  if (failure){
    if (!allow_failure){
      printf("ERROR: Connection failed. Exiting...\n");
      exit(1);
    }
    valid = false;
  } else {
    //Valid TCP connection
    valid = true;
  }
}


//Helper function to make a connection using info stored in the class variables
bool TCPClientConnection::makeconnection(){
  int domain, error;
  if (is_local)
    domain = AF_LOCAL;
  else
    domain = AF_INET;
  
  sockfd = socket(domain, SOCK_STREAM, 0);
  if (sockfd < 0){
    perror("ERROR: Unable to create socket");
    return false;
  }
  
  if (is_local){
    //Things to do for Local socket
    struct sockaddr_un servaddr1;
    servaddr1.sun_family = AF_UNIX;
    strcpy(servaddr1.sun_path, LOCAL_SERVER_PATH);
    
    error = connect(sockfd, (struct sockaddr*) &servaddr1, sizeof(servaddr1));
    
  } else {
    //Things to do for Internet socket
    struct sockaddr_in servaddr2;
    struct hostent* server = gethostbyname(servname);
    if (server == NULL){
      printf("Error: No such host\n");
      close(sockfd);
      return false;
    }
    
    bzero((char*) &servaddr2, sizeof(servaddr2));
    servaddr2.sin_family = AF_INET;
    bcopy((char*) server->h_addr, (char*) &servaddr2.sin_addr.s_addr, server->h_length);
    servaddr2.sin_port = htons(portno);
    
    error = connect(sockfd, (struct sockaddr*) &servaddr2, sizeof(servaddr2));
  }
  
  //Common error handling for both local and internet sockets
  if (error < 0){
    perror("Error: Unable to connect socket");
    close(sockfd);
    return false;
  }
  //It worked! We have a socket
  return true;
}


//Destructor closes connection
TCPClientConnection::~TCPClientConnection(void){
  if (valid)
    close(sockfd);
}


//close connection
void TCPClientConnection::stop(void){
  if (valid)
    close(sockfd);
}


//Wrapper for socket write() function
int TCPClientConnection::sendItem(void* item, size_t size){
  return (int) write(sockfd, item, size);
}


bool TCPClientConnection::isConnected(void){
  return valid;
}
