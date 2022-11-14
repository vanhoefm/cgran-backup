%**************************************************************************
% USRPF_send_file(): Sends a file to the USRPF server computer using
% Matlab's built in FTP. The server computer has been configured to to
% listen for FTP connections on port 1980.
%
% Inputs:
%
% - ip_address: The IP address of the USRPF Server computer. If you are
% unsure of the address then use ifconfig in a command prompt on the Linux
% computer.
% 
% Outputs:
%
% Example:
%
% USRPF_send_file('172.25.114.1', 'waveform.dat')
%
% Author: Jonas Hodel
% Date: 01/05/07
%**************************************************************************
function USRPF_send_file(ip_address, file_name)
    % Constants (as configured on the server)
    port = 1980;
    user = 'hodeljo';
    psswrd = 'usrpf';
    directory = 'upload';
    
    % Crate string 'ip_address:port'.
    server_str = [ip_address, ':', num2str(port)];
    % Open connection
    open_server = ftp(server_str, user, psswrd);
    % Change to default directory on server.
    cd(open_server, directory);
    % Upload file to server.
    mput(open_server, file_name);
    % Close the connection.
    close(open_server);
end