function soundGenerator(labels)
    
    frequencies = [261.63, 293.66, 329.63, 349.23, 392, 440, 493.88, 523.25, 587.33, 659.25];

    for i = 1:numel(labels)
        value = labels(i);

        % Check if the value is within the valid range
        if value >= 0 && value <= 9
            frequency = frequencies(value + 1); % Retrieve the corresponding frequency
            duration = 0.5; % Duration of the sound in seconds

            % Generate a sound tone
            t = 0:1/22050:duration; % Sampling frequency of 44100 Hz
            soundData = sin(2*pi*frequency*t);

            % Play the sound
            sound(soundData, 22050);
            pause(duration + 0.1); % Pause for the duration of the sound plus a small buffer
        end
    end
end