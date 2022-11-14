%**************************************************************************
% USRPF_close_connection(): Closes an open TCP/IP connection to the USRP
% server PC.
%
% Inputs:
%
% - connection: an open connection.
%
% Outputs:
%
% An error if the connection cannot be closed.
%
% Example:
%
% connection = USRPF_open_connection('172.25.114.66');
% USRPF_close_connection(connection);
%
% Author: Jonas Hodel
% Date: 23/04/07
%**************************************************************************
function USRPF_close_connection(connection)
    success = MEX_USRPF_close_connection(connection);
    if success == 0
        error('Unable to close connection')
    end
end