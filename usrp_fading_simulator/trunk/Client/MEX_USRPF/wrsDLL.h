#ifndef	__INCLUDED_WRS_RADIO_SERVICE__
#define __INCLUDED_WRS_RADIO_SERVICE__

/*--------------------------------------------------------------------------- 
   If, in your application an #include line is needed for the Windows.h 
   header file, this should be preceded with the 
   #define WIN32_LEAN_AND_MEAN macro. 

   "For historical reasons, the Windows.h header defaults to including 
   the Winsock.h header file for Windows Sockets 1.1. The declarations 
   in the Winsock.h header file will conflict with the declarations 
   in the Winsock2.h header file required by Windows Sockets 2.0. 
   The WIN32_LEAN_AND_MEAN macro prevents the Winsock.h from being 
   included by the Windows.h header [1]". 

   [1] http://msdn2.microsoft.com/en-us/library/ms737629.aspx

*--------------------------------------------------------------------------*/
#include <winsock2.h>

#define WRS_ERROR -1
#define WRS_NO_ERROR -2
#define WRS_CONNECTION_CLOSED -3
#define kBufferSize 4096


#ifdef __cplusplus
extern "C" {
#endif 


/*---------------------------------------------------------------------------
* wrs_open_server()
*
* Purpose:	Opens a sever by listening for clients. Once a client 
*			connects a socket is returned.
* 
* Input:	pcAddress - The IP address of the computer acting as the server
*			(use ipconfig if your not sure what your IP address is).
*			nPort - The port to listen on (should be an unused port eg 8881).
*
* Returns: 	On success a socket is returned. If an error occurs 
*			INVALID_SOCKET is returned.
*
* Author: 	Jonas Hodel
* Date: 	24/01/2007
*
*--------------------------------------------------------------------------*/
SOCKET wrs_open_server(const char *pcAddress, int nPort);



/*---------------------------------------------------------------------------
* wrs_open_client()
*
* Purpose:	Connects to a server (counterpart to wrs_open_server()).
*			Once successfully connected, a socket is returned.
*
* Input:	pcHost - The IP address of the host computer.
*			nPort - The port the the server is listening on.			
*
* Returns: 	On success a socket is returned, otherwise INVALID_SOCKET.
*						
* Author: 	Jonas Hodel
* Date: 	24/01/2007
*
*--------------------------------------------------------------------------*/
SOCKET wrs_open_client(const char *pcHost, int nPort);



/*---------------------------------------------------------------------------
* wrs_read_short_int()
*
* Purpose:	Reads short integers from an opened socket. wrs_open_server()
*			or wrs_open_client() should be called before attempting a read.
* 
* Input:	sd - The socket to read from.
*			buffer - Where the values are read to.
*			max_buffer_len - The length of buffer. This specifies the maximum 
*			number of short integers that can be read at once.
*
* Returns: 	On success the number of short integers read is returned.
*			On failure WRS_ERROR is returned. WRS_CONNECTION_CLOSED is 
			returned if the connection has been closed.
*
* Author: 	Jonas Hodel
* Date: 	24/01/2007
*
*--------------------------------------------------------------------------*/
int wrs_read_short_int(SOCKET sd, short int *buffer, int max_buffer_len);



/*---------------------------------------------------------------------------
* wrs_write_short_int()
*
* Purpose:	Writes short integers to an opened socket. wrs_open_server()
*			or wrs_open_client() should be called before attempting a write.
* 
* Input:	sd - The socket to write to.
*			buffer - Where the values to be written are located.
*			buffer_len - The length of buffer (the number of short integers
*			to write).
*
* Returns: 	On success the number of short integers written is returned.
*			On failure WRS_ERROR is returned.
*
* Author: 	Jonas Hodel
* Date: 	24/01/2007
*
*--------------------------------------------------------------------------*/
int wrs_write_short_int(SOCKET sd, short int *buffer, int buffer_len);



/*---------------------------------------------------------------------------
* wrs_read_string()
*
* Purpose:	Reads strings from an opened socket. wrs_open_server()
*			or wrs_open_client() should be called before attempting a read.
* 
* Input:	sd - The socket to read from.
*			buffer - Where the values are read to.
*			max_buffer_len - The length of buffer. This specifies the maximum 
*			number of characters that can be read at once.
*
* Returns: 	On success the number of read characters is returned.
*			On failure WRS_ERROR is returned. WRS_CONNECTION_CLOSED is 
			returned if the connection has been closed.
*
* Author: 	Jonas Hodel
* Date: 	24/01/2007
*
*--------------------------------------------------------------------------*/
int wrs_read_string(SOCKET sd, char *buffer, int buffer_len);



/*---------------------------------------------------------------------------
* wrs_write_string()
*
* Purpose:	Writes strings to an opened socket. wrs_open_server()
*			or wrs_open_client() should be called before attempting a write.
* 
* Input:	sd - The socket to write to.
*			buffer - Where the values to be written are located.
*			buffer_len - The length of buffer (the number of chars
*			to write).
*
* Returns: 	On success the number of chars written is returned.
*			On failure WRS_ERROR is returned.
*
* Author: 	Jonas Hodel
* Date: 	20/04/2007
*
*/
int wrs_write_string(SOCKET sd, char *buffer, int buffer_len);



/*---------------------------------------------------------------------------
* wrs_close()
*
* Purpose:	Closes a socket which was opened using either wrs_open_server()
*			or wrs_open_client().
*						
* Input:	sd - The socket to be closed.		
*
* Returns: 	On success WRS_NO_ERROR is returned, otherwise WRS_ERROR.
*						
*
* Author: 	Jonas Hodel
* Date: 	24/01/2007
*
*--------------------------------------------------------------------------*/
int wrs_close(SOCKET sd);



/*---------------------------------------------------------------------------
* wrs_get_last_error_string()
*
* Purpose:	Returns a string related to the last encountered error code.
*			A pointer to a staic string is returned so this is not thread safe.
*			This string is meant to server as a guide only as some errors may be cryptic
*			(thank winsock).
*						
* Author: 	Jonas Hodel
* Date: 	23/04/2007
*
*--------------------------------------------------------------------------*/
const char* wrs_get_last_error_string(void);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __INCLUDED_WRS_RADIO_SERVICE__
