function EncodeSpikesFx(inpath, outpath)

T = readtable(inpath);
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

writematrix(encodedDataArray,outpath)

end