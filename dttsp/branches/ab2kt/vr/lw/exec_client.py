#!/usr/bin/env python
"""
Simple remote execution client.
Establishes single connection.
Terminates on input EOF.
Each message must be a syntactically complete statement.
Maximum message [statement] length is 8192.
Where necessary, send substrings and concatenate in server.
"""

# Copyright FSF
# License GPLv3
# Author Frank Brickle <brickle@pobox.com

import sys
import socket

HOST = 'localhost'
PORT = 18617

def loop(host = HOST,
         port = PORT,
         input_func = sys.stdin.readline,
         exception = KeyboardInterrupt,
         msgfile = sys.stderr,
         verbose = False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    while True:
        try:
            data = input_func()
        except exception:
            break
        if not data:
            if verbose:
                print >>msgfile, 'End of input; quitting.'
            break
        if verbose:
            print >>msgfile, 'Got', repr(data)
        s.send(data)
        data = s.recv(8192)
        if verbose:
            print >>msgfile, 'Got back', repr(data)
        if not data:
            break
        if data[0:2] <> 'ok':
            print >>msgfile, repr(data)
    if verbose:
        print >>msgfile, 'Done.'
    s.close()


if __name__ == '__main__':
    loop(verbose = True)

