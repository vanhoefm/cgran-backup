#!/usr/bin/env python
from socket import *
from USRPF_client_manager import *

"""
 Listens out for clients. Once a client connects uses a client manager
 to service client requests. The manager is the interface between 
 the client and the USRP Fading Simulator. Only one client is handled at a time.
 This will spin forever.

"""

HOST = ''
PORT = 8881
BUFSIZ = 1024
ADDR = (HOST, PORT)

serversock = socket(AF_INET, SOCK_STREAM)
# Ensure you can restart server quickly when it terminates
serversock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
serversock.bind(ADDR)
serversock.listen(1)

while 1:
	print 'waiting for connection...'
	clientsock, addr = serversock.accept()
	print '...connected from:', addr

	# Create a client manager.
	manager = USRPF_client_manager()

	while 1:
		# Read what the client has to say.
		data = clientsock.recv(BUFSIZ)
		if not data: 
			# Connection has closed.
			break
		# Pass the client instruction onto the manager.
		reply = manager.instruct(data)
		# Send the managers response to the client.
		clientsock.send(reply)
	
	# Ensures the manager is happy when everything needs to end.
	manager.end()
	clientsock.close()

