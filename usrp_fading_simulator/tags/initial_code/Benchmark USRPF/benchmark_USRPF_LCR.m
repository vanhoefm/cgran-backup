%--------------------------------------------------------------------------
% Script that makes a comparison between the theoretical LCR and the LCR
% of a waveform passed through the USRPF Fading Simulator.
%
% Author: Jonas Hodel 
% Date: 08/053/07
%--------------------------------------------------------------------------

fs = 192e3;             % Sample rate (Samples/sec)
rf_freq = 440.1e6;      % Transmitter frequency (Hz)
c = 2.997925e8;         % speed of light (m/s)
speed = 100;            % Speed for fading profile (km/h)
 
fd = ((speed/3.6)*rf_freq)/c;
 
% Matlab LCR
%--------------------------------------------------------------------------
try
    if isempty(faded_sigTx)
        faded_sigTx = read_complex_binary('faded_c4fm1011test.dat');
    end
catch
    faded_sigTx = [];
    benchmark_USRPF_LCR
    return
end

P = abs(faded_sigTx);
Prms = sqrt(sum(abs(faded_sigTx).^2)/length(faded_sigTx));

P_min = min(P);
P_max = max(P);

%This LCR function appears to be from the Matlab File Exchange
[x, t] = lcr(P, linspace(P_min,P_max, 150));

% Normalise Results to RMS probability
[normalise_x, normalise_t] = lcr(P, Prms);
x = x/normalise_x;

figure
semilogy(20*log10(t/Prms),x, 'b')
%--------------------------------------------------------------------------
 

% Theory
%--------------------------------------------------------------------------
% The following is reproduced from "TIA Standard: Digital C4FM/CQPSK
% Transceiver Measurement Methods", section "1.5.33 Faded Channel
% Simulator". 
%
%     "The Level Crossing Rate (LCR) shall be compared against a calculated
%     LCR. The calculated Rayleigh level crossing rate, L(P), is as
%     follows: For P<0: L(P) = 0; for P>=0: L(P) = ((2*PI*P/Pave)^0.5)fd
%     exp(-P/Pave), where P is the signal power level, Pave is the mean
%     power level, and fd is the Doppler frequency offset associated with
%     the simulated vehicle speed. The Doppler frequency is given by the
%     following: fd = (v/c)fc where v is the simulated vehicle speed, c is
%     the speed of light in a vacuum (2.99792e8 m/s), and fc is the
%     assigned channel frequency.
% 
%     The measured LCR curve shall not deviate from the calculated LCR
%     curve by more than ±10% of the simulated vehicle speed. This shall
%     hold for 3 dB above the mean power level to 30 dB below the mean
%     power level." 

R = linspace(P_min, P_max, 1000);
Rrms = Prms;
 
Rho = R./Rrms;
%

% Normalise Results to RMS probability
LCR_normalise = sqrt(2*pi)*fd.*exp(-1.^2);

% The upper TIA specified theoretical LCR bound.
LCR_upper = sqrt(2*pi)*fd*1.1.*Rho.*exp(-Rho.^2)/LCR_normalise;
% The lower  TIA specified theoretical LCR bound.
LCR_lower = sqrt(2*pi)*fd*0.9.*Rho.*exp(-Rho.^2)/LCR_normalise;
%--------------------------------------------------------------------------


% Plot mean power
%--------------------------------------------------------------------------
hold on
mean_power = 20*log10(mean(P));
mean_line = [mean_power mean_power];
plot(mean_line, [10^-4 10^1], 'b--')
%--------------------------------------------------------------------------


% Plot theoritical limits
%-------------------------------------------------------------------------- 
semilogy(20*log10(Rho), LCR_upper, 'r')
semilogy(20*log10(Rho), LCR_lower, 'r')
title('LCR: USRPF Fading performance compared to Theory');
xlabel('Normalized Threshold (dB)')
ylabel('Normalized LCR ')
grid on
%--------------------------------------------------------------------------


% The measured LCR curve shall not deviate from the calculated LCR curve by
% more than ±10% of the simulated vehicle speed. This shall hold for 3 dB
% above the mean power level to 30 dB below the mean power level."   
%--------------------------------------------------------------------------
plot(mean_line + 3, [10^-4 10^1], 'r--')
plot(mean_line - 30, [10^-4 10^1], 'r--')
%--------------------------------------------------------------------------


legend('USRPF Fading simulator', 'Mean measured power', 'TIA specified theoretical limits')

