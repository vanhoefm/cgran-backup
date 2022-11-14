#!/usr/bin/env python
"""
Simple remote execution server.
Handles single connection in child.
Each message must be a syntactically complete statement.
Maximum message [statement] length is 8192.
Where necessary, send substrings and concatenate in server.
At present, only returns 'ok' or ('error'+exception).
"""

# Copyright FSF
# License GPLv3
# Author Frank Brickle <brickle@pobox.com>

import os
import sys
import socket

def loop(host, port, verbose = False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))

    if verbose:
        print >>sys.stderr, 'Ready on host', host, 'port', port

    # wait for request for exec handling
    while True:
        try:
            s.listen(1)
            conn, addr = s.accept()
        except KeyboardInterrupt:
            break
        if verbose:
            print >>sys.stderr, 'Connected by', addr

        # handle in a child
        pid = os.fork()

        if not pid:
            # in child
            s.close()
            mypid = os.getpid()

            while True:
                data = conn.recv(8192)

                # do we have a reason to live?
                if not data:
                    if verbose:
                        print >>sys.stderr, '%d: got empty data; quitting.' % (mypid)
                    break
                if verbose:
                    print >>sys.stderr, '%d: got %s' % (mypid, repr(data))

                # can we do as asked?
                try:
                    # still don't believe python lets you do this in full env
                    exec(data)
                except Exception, e:
                    if verbose:
                        print >>sys.stderr, '%d: oops! Badly formed (%r).' % (mypid, e)
                    # sorry
                    conn.send('error: ' + repr(e))
                    continue
                # no problem
                conn.send('ok')
              
            # ciao
            if verbose:
                print >>sys.stderr, '%d: server subprocess out of action now.' % (mypid)
            conn.close()
            sys.exit(0)

        # in parent
        if verbose:
            print >>sys.stderr, 'Being handled by', pid
            
        conn.close()

        pid = os.wait()[0]
        if verbose:
            print >>sys.stderr, "Server %d subprocess completed" % (pid)

    sys.exit(0)


HOST = 'localhost'
PORT = 18617

if __name__ == '__main__':
    loop(HOST, PORT, verbose = True)
