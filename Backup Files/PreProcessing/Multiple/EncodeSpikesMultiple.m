clear;
close all;

% Using Ben Spiker Algorithm (BSA) MATLAB Function from Auckland University of Technology
%% Reading Data from csv files (from Lyon model processed in Python)

G = ["f", "m"];

for i = 1:2
  for j = 1:3
    outdir = "trainBSA/" + G(i) + num2str(j);
    mkdir(outdir);
    disp(G(i) + num2str(j));
    for k = 0:9
      for l = 0:9
        inpath = "traincsv/" + G(i) + num2str(j) + "/0" + num2str(k) + G(i) + num2str(j) + "set" + num2str(l) + ".csv";
        outpath = "trainBSA/" + G(i) + num2str(j) + "/0" + num2str(k) + G(i) + num2str(j) + "set" + num2str(l) + ".csv";
        EncodeSpikesFx(inpath,outpath);
      end
    end
  end
end


