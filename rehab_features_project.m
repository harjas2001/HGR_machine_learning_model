clear, clc

%% Load Data
data = readmatrix('emg_gestures-27-sequential-2018-04-25-13-52-24-430.csv');

%% Trajectories
trajectories = data(:, 29);

%% Trimming Trajectories
for i = 1:length(trajectories)-1
    if (trajectories(i)-trajectories(i+1)~=0)
        trajectories(i) = -1;
    end
end

gar = trajectories==-1;
rows_m1 = find(data(:, 29) == -1);

trajectories(gar) = [];
%% Converting raw ADC values to voltage
data(rows_m1, :) = [];
data(1, :) = [];

electrodes = data(:, 1:8);
voltage_electrodes = electrodes;
for i = 1:numel(voltage_electrodes)
    voltage_electrodes(i) = ((voltage_electrodes(i)*5)/2^12)*(1000/200);
end

%% Plotting
% figure(1)
% plot(voltage_electrodes(:, 1));
% 
% figure(2)
% plot(abs(fft(voltage_electrodes(:, 1))))
%% Filtering Prior to Feature Calculation
fs = 5120;

cutoff = [20 700];
voltage_electrodes = bandpass(voltage_electrodes, cutoff, fs);

voltage_electrodes = bandstop(voltage_electrodes, [60 150], fs);

%% Plotting
% figure(3)
% plot(voltage_electrodes(:, 1));

%% Voltage_electrodes + Labels
voltage_electrodes = [voltage_electrodes, trajectories];

%% Splitting the data
rng('default')
cv = cvpartition(size(voltage_electrodes, 1), 'HoldOut', 0.2);
idx = cv.test;

dataTrain = voltage_electrodes(~idx, :);
dataTest = voltage_electrodes(idx, :);

Train_Electrodes = dataTrain(:, 1:8);
Train_Labels = voltage_electrodes(:, 9);
%%
window_size = 3000;

MAV = calculateMeanAbsValue(voltage_electrodes(:, 1:8), window_size);
WLarray = calculateWaveformLength(voltage_electrodes(:, 1:8), window_size);
% num_windows = floor((size(Train_Electrodes, 1)-window_size)/step_size) + 1;
% MAV = zeros(num_windows, size(Train_Electrodes, 2));
% WLarray = zeros(num_windows, size(Train_Electrodes, 2));
% WL = 0;
% SSC = zeros(num_windows, size(Train_Electrodes, 2));
% SSCi = 0;
% 
% for i = 2:num_windows
%     current_window = Train_Electrodes((i-1)*step_size+1:(i-1)*step_size+window_size,:);
% 
%     MAV(i, :) = mean(abs(current_window), 1);
% end
% 
% for k = 2:num_windows
%     current_window = Train_Electrodes((k-1)*step_size+1:(k-1)*step_size+window_size,:);
% 
%     for electrode = 1:size(Train_Electrodes, 2)
%         window_data = current_window(:, electrode);
%         WL = sum(abs(diff(window_data)));
%         WLarray(k, electrode) = WL;
%     end
% end

% %% zero crossing 
% threshold = 10/5000; % threshold of 10 mV
% ZC_counts = zeros(num_windows, size(Train_Electrodes, 2));% initialize zero crossing count matrix
% % window_size = 5000;
% % step_size = 2500;
% 
% % loop through the signal in overlapping windows
% for i = 1:size(Train_Electrodes, 2)
%     for j = 1:num_windows
%         start_idx = (j-1)*step_size + 1;
%         end_idx = start_idx + window_size - 1;
%         current_window = Train_Electrodes(start_idx:end_idx,i);
% 
%         % initialize zero crossing count
% 
%         % loop through the current window and count zero crossings
%         ZC = 0;
%         for k = 2:length(current_window)
% 
%             % check for zero crossing
%             if (current_window(k-1) > threshold && current_window(k) < -threshold) || (current_window(k-1) < -threshold && current_window(k) > threshold)
% 
%                 % check for dead zone
%                 if abs(current_window(k) - current_window(k-1)) > 0.01
% 
%                     % increment zero crossing count
%                     ZC = ZC + 1;
%                 end
%             end
%         end
% 
%         ZC_counts(j,i) = ZC; % store zero crossing count in matrix
% 
%     end
% end

%% Slope Sign Change (?)

% for k = 2:num_windows
%      current_window = voltage_electrodes((k-1)*step_size+1:(k-1)*step_size+window_size,:);
%      if current_window(k) > current_window(k-1) && current_window(k) > current_window(k+1)
%             SSCi = SSCi + 1;
%             SSC(k, electrode) = SSCi;
% 
%      elseif current_window(k) < current_window(k-1) && current_window(k) < current_window(k+1)
%             SSCi = SSCi + 1;
%             SSC(k) = SSCi;
%      end
% end

%% Labelling the Training Data
window_size = 3000;
step_size = window_size/2;
num_windows = floor((length(Train_Labels) - window_size)/step_size) + 1;
labels = zeros(num_windows, 1);

for i = 1:num_windows
    start_index = (i - 1) * step_size + 1;
    end_index = start_index + window_size - 1;
    window_data = Train_Labels(start_index:end_index);
    labels(i) = window_data(end);
end


training = table(MAV, WLarray, labels, 'VariableNames', ["MeanAbsVal", "WaveformLength", "HandGesture"]);

%% Training the model
features = [MAV, WLarray];
labels = training.HandGesture;

lda_model = fitcdiscr(features, labels);

%% Test data Features

testing_Data = dataTest(:, 1:8);
MAV_test = calculateMeanAbsValue(testing_Data, window_size);
WL_test = calculateWaveformLength(testing_Data, window_size);

testing = [MAV_test, WL_test];

predicted_labels = predict(lda_model, testing);
