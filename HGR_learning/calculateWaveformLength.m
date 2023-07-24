function wl_matrix = calculateWaveformLength(matrix, window_size)
    [num_rows, num_cols] = size(matrix);
    num_segments = floor(num_rows / (window_size / 2)) - 1;
    
    % Initialize wl_matrix
    wl_matrix = zeros(num_segments, num_cols);
    
    % Apply sliding window and calculate waveform length
    for i = 1:num_segments
        start_index = (i-1) * (window_size / 2) + 1;
        end_index = start_index + window_size - 1;
        segment = matrix(start_index:end_index, :);
        
        % Calculate the absolute differences between consecutive elements
        diff_segment = diff(segment);
        
        % Calculate the cumulative sum of absolute differences
        cum_sum = cumsum(abs(diff_segment));
        
        % Calculate the waveform length
        wl = sum(cum_sum);
        
        wl_matrix(i, :) = wl;
    end
end





