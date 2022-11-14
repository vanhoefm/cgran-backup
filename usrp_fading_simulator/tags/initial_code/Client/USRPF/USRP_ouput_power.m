%**************************************************************************
% Quick test to understand the relationship between software gain
% (multiplier) and the USRP output power. This was tested by applying
% various software gains to a transmitted test tone and measuring the
% resulting RSSI on a TP9100.
%
% Based on these results I have concluded that the relationship is:
%
% "output_power = 18*log10(multiplier) - 73" (dB)
%
% This relationship is only reliable from 0 to -40 dB.
%
% Author: Jonas Hodel
% Date: 07/05/07
%**************************************************************************
 
close all
 
% RSSI was measured at these multiplier values.
multiplier = ones(1, 15);
j = 1;
for k = multiplier
    multiplier(j) = 2^(j-1);
    j = j + 1;
end
 
% These are the results of the RSSI measurements.
RSSI = [-84 -58 -59 -48 -41 -35 -43 -37 -31 -24 -16 -10 -6 -2 -1];
 
% Plot the results
plot(multiplier, RSSI, 'b.-')
title('Output power vs Software multiplier')
xlabel('Software multiplier (scalar)')
ylabel('Resulting output power (dBm)')
grid on
figure
 
% Plot the results on a compared to 20*log10().
plot(20*log10(multiplier), RSSI, 'b.-')
grid on
 
% Fit line to gain an understanding of the relationship.
x = 20*log10(multiplier);
y = (45/50)*x - 73;
hold on
plot(x, y, 'r')
title('Output power vs Software multiplier')
xlabel('Software multiplier, 20*log10(scalar))')
ylabel('Resulting output power (dBm)')
legend('Measured results', 'Aprox. line of best fit')