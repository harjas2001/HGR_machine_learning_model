clear, clc

%% Load Training Data
training_data = readmatrix("emg_gestures-27-sequential-2018-04-25-13-52-24-430.csv");

%% Load Demo Data
demo_data = readmatrix("emg_gestures-27-repeats_short-2018-04-25-13-56-56-520.csv");

%% Time axis
t = (0:size(training_data, 1)-1)/5120;
t = t';

t2 = (0:size(demo_data, 1)-1)/5120;
t2=t2';

%% Trajectories
trajectories = demo_data(:, 29);

%% Training Trajectories
training_trajectories = training_data(:, 29);


%% Trimming Trajectories
for i = 1:length(trajectories)-1
    if (trajectories(i)-trajectories(i+1)~=0)
        trajectories(i) = -1;
    end
end

rows_m1 = find(demo_data(:, 29) == -1);

%% Trimming Training Trajectories
for i = 1:length(training_trajectories)-1
    if(training_trajectories(i) - training_trajectories(i+1) ~= 0)
        training_trajectories(i) = -1;
    end
end

rows_training = find(training_data(:, 29) == -1);

%% Trimming rows from demo data
demo_data(rows_m1, :) = [];
demo_data(1, :) = [];

%% Removing the Unwanted Values
gar = trajectories==-1;
trajectories(gar) = [];

%% Removing Unwanted Training Trajectories
gar2 = training_trajectories==-1;
training_trajectories(gar2) = [];

%% Converting ADC values to voltage (mV)
electrodes = demo_data(:, 1:8);
voltage_electrodes = electrodes;
for i = 1:numel(voltage_electrodes)
    voltage_electrodes(i) = ((voltage_electrodes(i)*5)/2^12)*(1000/200);
end

%% Converting ADC values to voltage (mV): Training Data
training_electrodes = training_data(:, 1:8);
trn_vol = training_electrodes;

for i = 1:numel(trn_vol)
    trn_vol(i) = ((trn_vol(i)*5)/2^12)*(1000/200);
end

%% Filtering (Bandpass (cf = 20 700) & Bandstop (cf = 60 150))
fs = 5120;

cutoff = [20 700];
filtered = bandpass(voltage_electrodes, cutoff, fs);
BS_filtered = bandstop(filtered, [60 150], fs);
%% Concatenate the labels with the electrodes
BS_filtered = [BS_filtered, trajectories];

%% Feature Extraction
window_size = 3000;

MAV = calculateMeanAbsValue(BS_filtered(:, 1:8), window_size);
WLarray = calculateWaveformLength(BS_filtered(:, 1:8), window_size);

%% Generating the Labels

% Using the window of size 3000, segment the trajectories set and obtain
% the last value of each segment and append that to a new matrix 
% 'demo_labels'
step_size = window_size/2;
num_windows = floor((length(trajectories) - window_size)/step_size) + 1;
demo_labels = zeros(num_windows, 1);

for i = 1:num_windows
    start = (i - 1) * step_size + 1;
    end_idx = start + window_size - 1;

    window_data = trajectories(start:end_idx);
    demo_labels(i) = window_data(end);
end

%% Creating the demo table
demonstration = table(MAV, WLarray, 'VariableNames', ["MeanAbsVal", "WaveformLength"]);

%% Classifying and Obtaining Prediction: LDA Model
[yfit, scores] = trainedModel.predictFcn(demonstration);

accuracy = (sum(yfit == demo_labels)/numel(demo_labels))*100;

%% Confusion Matrix
cm_demo = confusionchart(demo_labels, yfit);
cm_demo.Title = "Classification of MAV/WL Using LDA";
cm_demo.RowSummary = 'row-normalized';
cm_demo.ColumnSummary = 'column-normalized';

%% ROC Curve

%% Difference in Trials Plot
figure(1)
plot(training_trajectories(1:2.5e5));
hold on
plot(trajectories(1:2.5e5));
hold off
legend("Sequential Gestures", "Short Gestures")
title("Sequential vs Short Trial: 25 Apr 2023 Subject 27")
xlabel("Time (ms)")
ylabel("Hand Gesture")
%% Sequential EMG Signal (Unfiltered)
legendText = cell(1, 8);
for i = 1:8
    strNum = ['Electrode ', num2str(i)];

    legendText{i} = strNum;
end

figure(2)
plot(t, trn_vol)
xlim([1 50])
title("Raw Sequential EMG Signal")
legend(legendText, 'Location', 'best')
xlabel("Samples")
ylabel("Voltage (mV)")

%% Short EMG Signal (Frequency Filtered)
legendText = cell(1, 8);
for i = 1:8
    strNum = ['Electrode ', num2str(i)];

    legendText{i} = strNum;
end

figure(3)
plot(t2, voltage_electrodes)
xlim([1 10])
title("Raw Short EMG Signal")
legend(legendText, 'Location', 'best')
xlabel("Samples")
ylabel("Voltage (mV)")

%% Plotting Filtered vs Unfiltered Demo Data
figure(4)
subplot(2,1,1)
plot(t2, voltage_electrodes(:, 2))
xlim([0 25])
title("Raw Short EMG Signal")
xlabel("Time (s)")
ylabel("Voltage (mV)")

subplot(2, 1, 2)
plot(t2, BS_filtered(:, 2))
xlim([0 25])
title("Filtered Short EMG Signal")
xlabel("Time (s)")
ylabel("Voltage (mV)")

%% Sound Generation
figure(5)
plot(demo_labels);
xlabel("Time (ms)")
ylabel("Hand Gesture")

soundGenerator(demo_labels);
%% Prediction Sound
figure(6)
plot(yfit)
xlabel("Time (ms)")
ylabel("Hand Gesture")

soundGenerator(yfit);