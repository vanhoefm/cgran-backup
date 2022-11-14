%--------------------------------------------------------------------------
% Script that makes a comparison between the theoretical CPDF and the CPDF
% of a waveform passed through the USRPF Fading Simulator.
%
% Author: Jonas Hodel 
% Date: 08/05/07
%--------------------------------------------------------------------------
 
% Matlab CPDF
%--------------------------------------------------------------------------
try
    if isempty(faded_sigTx)
        faded_sigTx = read_complex_binary('faded_c4fm1011test.dat');
    end
catch
    faded_sigTx = [];
    benchmark_USRPF_CPDF
    return
end
P = abs(faded_sigTx);
[n x] = hist(P,10000);
% Probability
n = n/sum(n);
% Cumulative probability
n = cumsum(n);
%--------------------------------------------------------------------------
 
% Theory
%--------------------------------------------------------------------------
% The following is reproduced from "TIA Standard: Digital C4FM/CQPSK
% Transceiver Measurement Methods", section "1.5.33 Faded Channel
% Simulator". 
%
%     "The measured Rayleigh Cumulative Probability Distribution Function
%     (CPDF) shall be compared against a calculated CPDF. The calculated
%     Rayleigh CPDF, F(P), is as follows: for P<0: F(P) = 0; for P>=0: F(P)
%     = 1- exp(- P/Pave), where P is the signal power level and Pave is the
%     mean power level. Measured CPDF of power shall be within ±1 dB of the
%     calculated CPDF of power for 10 dB above the mean power level to 20
%     dB below the mean power level. Measured CPDF of power shall be within
%     ±5 dB of the calculated CPDF of power for 20 dB below the mean power
%     level to 30 dB below the mean power level."
P_min = min(P);
P_max = max(P);
R = linspace(P_min, P_max, 1000);
Rrms = sqrt(sum(abs(faded_sigTx).^2)/length(faded_sigTx));
Rho = R./Rrms;
CPDF = 1 - exp(-Rho.^2);
%--------------------------------------------------------------------------
 
 
% Plot restuls
%--------------------------------------------------------------------------
hold on
plot(20*log10(x), n)
% plot(20*log10(Rho), CPDF, 'r')
mean_power = 20*log10(mean(P));
mean_line = [mean_power mean_power];
plot(mean_line, [0 1], 'b--')
title('CPDF: USRPF Fading performance compared to Theory');
xlabel('dB')
ylabel('CPDF')
grid on
%--------------------------------------------------------------------------
 
 
% Measured CPDF of power shall be within ±1 dB of the calculated CPDF of
% power for 10 dB above the mean power level to 20 dB below the mean power
% level.  
%--------------------------------------------------------------------------
upper_bound = max(find(20*log10(Rho) <= mean_power + 10 - 1));
lower_bound = min(find(20*log10(Rho) >= mean_power - 20 - 1));
range = lower_bound:upper_bound;
plot(20*log10(Rho(range)) + 1, CPDF(range), 'r')
 
upper_bound = max(find(20*log10(Rho) <= mean_power + 10 + 1));
lower_bound = min(find(20*log10(Rho) >= mean_power - 20 + 1));
range = lower_bound:upper_bound;
plot(20*log10(Rho(range)) - 1, CPDF(range), 'r')
%--------------------------------------------------------------------------
 
 
% Measured CPDF of power shall be within ±5 dB of the calculated CPDF of
% power for 20 dB below the mean power level to 30 dB below the mean power
% level.  
%--------------------------------------------------------------------------
upper_bound = max(find(20*log10(Rho) <= mean_power - 20 - 5));
lower_bound = min(find(20*log10(Rho) >= mean_power - 30 - 5));
range = lower_bound:upper_bound;
plot(20*log10(Rho(range)) + 5, CPDF(range), 'r')
 
upper_bound = max(find(20*log10(Rho) <= mean_power - 20 + 5));
lower_bound = min(find(20*log10(Rho) >= mean_power - 30 + 5));
range = lower_bound:upper_bound;
plot(20*log10(Rho(range)) - 5, CPDF(range), 'r')
%--------------------------------------------------------------------------


legend('USRPF Fading simulator', 'Mean measured power', 'TIA theoretical limits')


% Plot mean power + 10, - 20 and -30 dB
%--------------------------------------------------------------------------
plot(mean_line + 10, [0 1], 'r--')
plot(mean_line - 20, [0 1], 'r--')
plot(mean_line - 30, [0 1], 'r--')
%--------------------------------------------------------------------------