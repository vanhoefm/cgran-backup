%**************************************************************************
% USRPF_open_connection(): Opens a TCP/IP connection to a server computer
% running the USRP Fading Simulator Server Software.  
%
% Inputs:
%
% - ip_address: The IP address of the server computer. If you are unsure of
% the IP address, on the server, use ifconfig (in Windows the equivalent is
% ipconfig)
%
% Outputs:
%
% If successful an open connection is returned. This can then be used by
% other USRPF_* functions. If a connection can not be opened, an error is
% returned. 
%
% Example:
%
% connection = USRPF_open_connection('172.25.114.66')
%
% Author: Jonas Hodel
% Date: 20/04/07
%**************************************************************************
function connection = USRPF_open_connection(ip_address)
    connection = MEX_USRPF_open_connection(ip_address);
    if connection == -1
        error('Unable to open connection')
    end
end