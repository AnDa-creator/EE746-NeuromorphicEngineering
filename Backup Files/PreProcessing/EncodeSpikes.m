clear;
close all;

% Using Ben Spiker Algorithm (BSA) MATLAB Function from Auckland University of Technology
%% Reading Data from array.csv file (from Lyon model processes in Python)

T = readtable('array.csv');
T_array = table2array(T);
T_norm = (T_array - min(T_array))./(max(T_array) - min(T_array));
% T_norm = normalize(T_array);

N = size(T_norm,2);

encodedDataArray = [];

for i = 1:N
    
    normData = T_norm(:,i); % Single Channel
    
    % normData is your data after normalisation
    order=23;
    passband=0.2;
    threshold=0.89;
    
    % Creation of a passband filter 
    %filter=fir1(order, passband)*2;
    filter=fir1(order, passband)*max(normData)*2; % Suggestion 
    filterSize=length(filter);

    tempData=cat(1,(ones(filterSize,1)*normData(1)),normData,(ones(filterSize,1)*normData(end))); % add two vectors in the begining and in the end with n-order elements with the last value of the signal
    % Encoding the data
    encodedData = Bsa(tempData, filter, threshold); 
    encodedDataArray = [encodedDataArray encodedData];
    
end

writematrix(encodedDataArray,'PreProcessedData.csv')

figure
plot(normData);
title('Scaled data');

figure
%stem(encodedData)
stem(encodedData,'r')
title('Spike trains');

%% Reconstruction of last channel/ sample in the loop above

% % Signal reconstruction by convolution of the data and the filter. The
% % decodedData length dl = el+fl-1 where el is the encodedData length and fl
% % the filter length.

decodedData=conv(encodedData,filter);
decodedData=decodedData(filterSize+1:end-((2*filterSize)-1),:);
mse=mean((normData(:,:) - decodedData(:,:)).^2); % mean square error
disp(mse);



figure
hold on
plot(normData,'b');
plot(decodedData,'r')
title('Reconstructed data');
hold off 

%% Test on sine wave

% 
% fs = 300;                    % Sampling frequency (samples per second)
% dt = 1/fs;                   % seconds per sample
% StopTime = 2;             % seconds
% t = (0:dt:StopTime-dt)';     % seconds
% F = 1;                      % Sine wave frequency (hertz)
% normData = 0.5*(sin(2*pi*F*t)+1);
% 
% % normData is your data after normalisation
% order=23;
% passband=0.05;
% threshold=0.89;
% % Creation of a passband filter 
% %filter=fir1(order, passband)*2;
% filter=fir1(order, passband)*max(normData)*2; % Suggestion 
% filterSize=length(filter);
% 
% tempData=cat(1,(ones(filterSize,1)*normData(1)),normData,(ones(filterSize,1)*normData(end))); % add two vectors in the begining and in the end with n-order elements with the last value of the signal
% % Encoding the data
% encodedData = Bsa(tempData, filter, threshold); 
% 
% % Signla reconstruction by convolution of the data and the filter. The
% % decodedData length dl = el+fl-1 where el is the encodedData length and fl
% % the filter length.
% decodedData=conv(encodedData,filter);
% decodedData=decodedData(filterSize+1:end-((2*filterSize)-1),:);
% mse=mean((normData(:,:) - decodedData(:,:)).^2); % mean square error
% disp(mse);
% 
% figure
% plot(normData);
% title('Scaled data');
% 
% figure
% %stem(encodedData)
% stem(encodedData,'r')
% title('Spike trains');
% 
% figure
% hold on
% plot(normData,'b');
% plot(decodedData,'r')
% title('Reconstructed data');
% hold off 