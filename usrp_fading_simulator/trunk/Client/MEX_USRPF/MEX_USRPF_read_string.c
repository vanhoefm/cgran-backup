/*-------------------------------------------------------------------------
* MEX function that reads a string from an open socket.
*
* Inputs:
*
* - connection: An open connection to the server.
*
* Outputs:
*
* If successful the read string is returned, otherwise an empty string 
* is returned.
*
* Example:
*
* string = USRPF_read_string(connection)
*
* Author: Jonas Hodel
* Date: 24/04/07
*------------------------------------------------------------------------*/
#include "mex.h"
#include "wrsDLL.h"
#include <string.h>
 
#define BUFFER_SIZE 256
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
const mxArray *prhs[])
{
    char *string;
    int buflen, mrows, ncols;
    SOCKET socket;
    
  /* Check for proper number of arguments. */
    if (nrhs != 1)
        mexErrMsgTxt("One input required, an open connection");
    else if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    
      /* The input must be a noncomplex scalar double.*/
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
    !(mrows == 1 && ncols == 1)) {
        mexErrMsgTxt("First input must be an open connection.");
    }
    
    /* Allocate memory for the output string. */
    string = mxCalloc(BUFFER_SIZE, sizeof(char));   
    
    socket = (SOCKET) *mxGetPr(prhs[0]);
    
    if (wrs_read_string(socket, string, BUFFER_SIZE) == WRS_ERROR){
        mexWarnMsgTxt("Unable to read string");
        mexWarnMsgTxt(wrs_get_last_error_string());
        plhs[0] = NULL;
        return;
    }
    else{
        /* Set C-style string 'string' to MATLAB mexFunction output*/
        plhs[0] = mxCreateString(string);
        return;
    }
}
