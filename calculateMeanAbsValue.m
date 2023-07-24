function mav_matrix = calculateMeanAbsValue(matrix, window_size)
    [num_rows, num_cols] = size(matrix);
    num_segments = floor(num_rows / (window_size / 2)) - 1;
    
    % Initialize mav_matrix
    mav_matrix = zeros(num_segments, num_cols);
    
    % Apply sliding window and calculate mean absolute value
    for i = 1:num_segments
        start_index = (i-1) * (window_size / 2) + 1;
        end_index = start_index + window_size - 1;
        segment = matrix(start_index:end_index, :);
        mav_matrix(i, :) = mean(abs(segment));
    end
end