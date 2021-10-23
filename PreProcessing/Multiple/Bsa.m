% This function is a translation from Spanish to English from BSA algorithm
% (BSA, a Fast and Accurate Spike Train Encoding Scheme) implemented by Imanol
% Bilbao (2018) and translate by Dr. Israel Espinosa
% (https://kedri.aut.ac.nz/staff/staff-profiles/israel-espinosa-ramos)
% Auckland University of Technology, Auckland, New Zealand
% Knowledge Engineering an Discovery Research Institute
% https://kedri.aut.ac.nz/
function encodedSignal = Bsa(inputSignal, filter, threshold)
% The signal must be a vector of n elements
n_filter = length(filter);
n_input = length(inputSignal);

for i = 1:n_input
    error1 = 0;
    error2 = 0;
    
    for j = 1:n_filter
        if i+j-1 <= n_input
            error1 = error1 + abs(inputSignal(i+j-1) - filter(j));
            error2 = error2 + abs(inputSignal(i+j-1));
        end
    end
    
    if error1 <= (error2 - threshold)
    %if error1 <= (error2 * threshold) % multiplicative
        encodedSignal(i)=1;
        for j = 1:n_filter
            if i+j-1 <= n_input
                inputSignal(i+j-1) = inputSignal(i+j-1)-filter(j);
            end
        end
    else
        encodedSignal(i)=0;
    end
end
encodedSignal=encodedSignal';
end