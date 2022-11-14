/*-------------------------------------------------------------------------
* MEX funciton that sends a string to the USRP server PC.
*
* Inputs:
*
* - connection: An open connection to the server.
* - string: The string to send to the server computer.
*
* Outputs:
*
* If successful 1 is returned, otherwise 0 is returned.
*
* Example:
*
* USRPF_send_string(connection, 'hello, how are you')
*
* Author: Jonas Hodel
* Date: 23/04/07
*------------------------------------------------------------------------*/
#include "mex.h"
#include "wrsDLL.h"
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
const mxArray *prhs[])
{
    char *string;
    int buflen, status, mrows, ncols;
    SOCKET socket;
    double *success;
    
  /* Check for proper number of arguments. */
    if (nrhs != 2)
        mexErrMsgTxt("Two inputs required, an open connection followed by the string to send");
    else if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    
      /* The input must be a noncomplex scalar double.*/
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
    !(mrows == 1 && ncols == 1)) {
        mexErrMsgTxt("First input must be an open connection.");
    }
    
    /* Second input must be a string. */
    if (mxIsChar(prhs[1]) != 1)
        mexErrMsgTxt("Second input must be a string.");
    
  /* Input must be a row vector. */
    if (mxGetM(prhs[1]) != 1)
        mexErrMsgTxt("Second input must be a row vector, in the form of a string.");
    
  /* Get the length of the input string. */
    buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1])) + 1;
    
  /* Allocate memory for input string. */
    string = mxCalloc(buflen, sizeof(char));
    
  /* Copy the string data from prhs[1] into a C string
   * "string". */
    status = mxGetString(prhs[1], string, buflen);
    if (status != 0)
        mexWarnMsgTxt("Not enough space. String is truncated.");
    
  /* Create matrix for the return argument. */
    plhs[0] = mxCreateDoubleMatrix(mrows,ncols, mxREAL);
    success = mxGetPr(plhs[0]);
    
    socket = (SOCKET) *mxGetPr(prhs[0]);
    
    if (wrs_write_string(socket, string, strlen(string)) == WRS_ERROR){
        mexWarnMsgTxt("Unable to send string");
        mexWarnMsgTxt(wrs_get_last_error_string());
        *success = 0; 
        return;
    }
    else{
       *success = 1; 
       return;
    }
}

