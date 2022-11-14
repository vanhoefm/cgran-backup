/*-------------------------------------------------------------------------
* MEX funciton that opens a TCP/IP connection to a server computer
* running the USRP Fading Simulator Server Software.  
*
* Inputs:
*
* - ip_address: The IP address of the server computer. If you are unsure of
* the IP address, on the server, use ifconfig (in Windows the equivalent is
* ipconfig)
*
* Outputs:
*
* If successful an open connection is returned. This can then be used by
* other MEX_USRPF_* functions. If a connection can not be opened, -1 
* is returned. 
*
* Example:
*
* connection = USRPF_open_connection('172.25.114.66')
*
* Author: Jonas Hodel
* Date: 23/04/07
*------------------------------------------------------------------------*/
#include "mex.h"
#include "wrsDLL.h"

// 8881 is the default port that the server is listening on.
#define PORT 8881

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
const mxArray *prhs[])
{
    char *ip_address;
    char *input_buf, *output_buf;
    int buflen, status;
    SOCKET socket;
    double *connection;
    
  /* Check for proper number of arguments. */
    if (nrhs != 1)
        mexErrMsgTxt("One input required, in the form of an IP adress. e.g. 172.25.113.64");
    else if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    
  /* Input must be a string. */
    if (mxIsChar(prhs[0]) != 1)
        mexErrMsgTxt("Input must be a string.");
    
  /* Input must be a row vector. */
    if (mxGetM(prhs[0]) != 1)
        mexErrMsgTxt("Input must be a row vector.");
    
  /* Get the length of the input string. */
    buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0])) + 1;
    
  /* Allocate memory for the input string. */
    ip_address = mxCalloc(buflen, sizeof(char));
    
  /* Copy the string data from prhs[0] into a C string
   * input_buf. */
    status = mxGetString(prhs[0], ip_address, buflen);
    if (status != 0)
        mexWarnMsgTxt("Not enough space. String is truncated.");
    
    /* Create matrix for the return argument. */
    plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL);
    connection = mxGetPr(plhs[0]);
    
  /* Starts a client which will open a conneciton to a server. */
    socket = wrs_open_client(ip_address, PORT);
    
    if (socket == INVALID_SOCKET)
    {
        mexWarnMsgTxt("Unable to open a connection to the server");
        // Returns the last error encountered.
        mexWarnMsgTxt(wrs_get_last_error_string());
        *connection = -1;
    }
    else
        *connection = (double) socket;
}

