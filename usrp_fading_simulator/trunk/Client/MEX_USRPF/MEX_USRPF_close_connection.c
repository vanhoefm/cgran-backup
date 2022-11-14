/*-------------------------------------------------------------------------
* MEX funciton that closes an open TCP/IP connection to a server computer.
*
* Inputs:
*
* - connection: The open connection to close 
*               (see MEX_USRPF_open_connection()).
*
* Outputs:
*
* If successful 1 is returned, otherwise 0 is returned.
*
* Example:
*
* Author: Jonas Hodel
* Date: 23/04/07
*------------------------------------------------------------------------*/
#include "mex.h"
#include "wrsDLL.h"
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
const mxArray *prhs[])
{
    double *connection, *success;
    int mrows, ncols;
    int temp_success;
    
  /* Check for proper number of arguments. */
    if (nrhs != 1) {
        mexErrMsgTxt("One input required, in the form of an open connection");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
  /* The input must be a noncomplex scalar double.*/
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
    !(mrows == 1 && ncols == 1)) {
        mexErrMsgTxt("Input must be an open connection");
    }
    
  /* Create matrix for the return argument. */
    plhs[0] = mxCreateDoubleMatrix(mrows,ncols, mxREAL);
    
  /* Assign pointers to each input and output. */
    connection = mxGetPr(prhs[0]);
    success = mxGetPr(plhs[0]);
    
  /* Call the wrs function that opens a connection */
    temp_success = wrs_close((SOCKET) *connection);
    
    if (temp_success == WRS_ERROR)
    {
        mexWarnMsgTxt("Unable to close the connection to the server");
        // Returns the last error encountered.
        mexWarnMsgTxt(wrs_get_last_error_string());
        *success = 0;
    }
    else
        *success = 1;
}