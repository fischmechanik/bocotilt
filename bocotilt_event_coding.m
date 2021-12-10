function[EEG] = bocotilt_event_coding(EEG, RESPS, positions, trial_log)

    % Struct for new events
    nec = 0;
    new_events = struct();

    % Some response channel preprocessing
    resps_left = rescale(RESPS.data(1, :));
    resps_right = rescale(RESPS.data(2, :));
    critval_responses = 0.5;
    maxrt = 1500;

    % Iterate events recoding rt and accuracy info
    trial_nr = 0;
    for e = 1 : length(EEG.event)

        % Check for block start markers
        if (strcmpi(EEG.event(e).type(1), {'S'}) & ismember(str2num(EEG.event(e).type(2 : 4)), [120 : 160]))

            % Get block number
            block_nr = str2num(EEG.event(e).type(2 : 4)) - 120;

            % Rest sequential marker
            previous_task = -1;

        end

        % If trial
        if (strcmpi(EEG.event(e).type(1), {'S'}) & ismember(str2num(EEG.event(e).type(2 : 4)) - 40, [0 : 32]))
    
            trial_nr = trial_nr + 1;

            % Get event code
            ecode = str2num(EEG.event(e).type(2 : 4)) - 40;

            % Decode bonustrial
            if ecode <= 16
                bonustrial = 1;
            else
                bonustrial = 0;
            end

            % Decode task
            if ismember(ecode, [1 : 8, 17 : 24])
                tilt_task = 1;
            else
                tilt_task = 0;
            end

            % Decode cue
            if ismember(mod(ecode, 8), [1, 2, 3, 4]) 
                cue_ax = 1;
            else
                cue_ax = 0;
            end

            % Decode target
            if ismember(mod(ecode, 4), [1, 2]) 
                target_red_left = 1;
            else
                target_red_left = 0;
            end

            % Decode distractor
            if mod(ecode, 2) == 1
                distractor_red_left = 1;
            else
                distractor_red_left = 0;
            end

            % Check response interference
            if target_red_left == distractor_red_left
                response_interference = 1;
            else
                response_interference = 0;
            end

            % Code task sequence
            current_task = tilt_task;
            if previous_task == -1
                task_switch = -1;
            elseif current_task == previous_task
                task_switch = 0;
            else
                task_switch = 1;
            end
            previous_task = current_task;

            % Decode correct response side
            if tilt_task == 1 & target_red_left == 1
                correct_response = 0; % left
            elseif tilt_task == 0 & target_red_left == 0
                correct_response = 0; % left
            else
                correct_response = 1; % right
            end

            % Lookup response
            left_rt = min(find(resps_left(EEG.event(e).latency + 800 : EEG.event(e).latency + 800 + (maxrt / (1000 / EEG.srate))) >= critval_responses)) * (1000 / EEG.srate);
            right_rt = min(find(resps_right(EEG.event(e).latency + 800 : EEG.event(e).latency + 800 + (maxrt / (1000 / EEG.srate))) >= critval_responses)) * (1000 / EEG.srate);
            if isempty(left_rt) & isempty(right_rt)
                rt = NaN;
                response_side = 2;
            elseif isempty(left_rt)
                rt = right_rt;
                response_side = 1;
            elseif isempty(right_rt)
                rt = left_rt;
                response_side = 0;
            else
                [rt, resp_idx] = min([left_rt, right_rt]);
                if resp_idx == 1
                    response_side = 0;
                else
                    response_side = 1;
                end
            end

            % Accuracy
            if response_side == 2
                acc = 2;
            elseif response_side == correct_response
                acc = 1;
            else
                acc = 0;
            end

            % Create event
            nec = nec + 1;
            new_events(nec).latency = EEG.event(e).latency;
            new_events(nec).duration = 1;
            new_events(nec).type = "trial";
            new_events(nec).code = "trial";
            new_events(nec).urevent = EEG.event(e).urevent;
            new_events(nec).block_nr = block_nr;
            new_events(nec).trial_nr = trial_nr;
            new_events(nec).bonustrial = bonustrial;
            new_events(nec).tilt_task = tilt_task;
            new_events(nec).cue_ax = cue_ax;
            new_events(nec).target_red_left = target_red_left;
            new_events(nec).distractor_red_left = distractor_red_left;
            new_events(nec).response_interference = response_interference;
            new_events(nec).task_switch = task_switch;
            new_events(nec).correct_response = correct_response;
            new_events(nec).response_side = response_side;
            new_events(nec).rt = rt;
            new_events(nec).accuracy = acc;
            new_events(nec).position_color = positions(trial_nr, 1);
            new_events(nec).position_tilt = positions(trial_nr, 2);

            % Code log response side
            if trial_log(trial_nr, 1) == 0
                log_response_side = 2;
            elseif trial_log(trial_nr, 1) == 1
                log_response_side = 0;
            elseif trial_log(trial_nr, 1) == 2
                log_response_side = 1;
            end
            new_events(nec).log_response_side = log_response_side;

            % Code log rt
            log_rt = trial_log(trial_nr, 3);
            if log_rt == -1
                log_rt = NaN;
            end
            if log_rt > maxrt
                log_rt = NaN;
            end
            new_events(nec).log_rt = log_rt;

            % Code log accuracy
            log_accuracy = trial_log(trial_nr, 2);
            if log_accuracy == 3
                log_accuracy = 2;
            end
            if isnan(log_rt)
                log_accuracy = 2;
            end
            new_events(nec).log_accuracy = log_accuracy;
            
            if tilt_task == 0
                new_events(nec).position_target = new_events(nec).position_color;
                new_events(nec).position_distractor = new_events(nec).position_tilt;
            else
                new_events(nec).position_target = new_events(nec).position_tilt;
                new_events(nec).position_distractor = new_events(nec).position_color; 
            end
            
            % Mark bugged trials
            if ecode == 0
                new_events(nec).type = "zerocoded";
                new_events(nec).code = "zerocoded";
            end
        
        end

        % If event is boundary event...
        if strcmpi(EEG.event(e).type, 'boundary')
            stimulus_type = 'boundary';
        end

    end

    % Get indices of trials
    trial_idx = [];
    for e = 1 : length(new_events)
        if strcmpi(new_events(nec).type, "trial")
            trial_idx(end + 1) = e;
        end
    end

    % Code sequences
    old_block = 0;
    n_nontrial = 0;
    for tidx = 1 : length(trial_idx)

        % Get trial index in event structure
        e = trial_idx(tidx);

        % If block changes
        current_block = new_events(e).block_nr;
        if current_block ~= old_block

            % If not first block, fill previous sequence with values
            if old_block ~= 0
                for f = sequence_start : e - 1
                    new_events(f).sequence_length = sequence_length;
                end
            end

            old_block = current_block;
            sequence_length = 1;
            sequence_position = 1;
            sequence_start = e;
            previous_type = new_events(e).bonustrial;
            current_type = new_events(e).bonustrial;

        % If block does not change
        else

            % If sequence continues
            current_type = new_events(e).bonustrial;
            if current_type == previous_type
                sequence_length = sequence_length + 1;
                sequence_position = sequence_position + 1;
                previous_type = new_events(e).bonustrial;

            % If sequence does not continue
            else

                % Fill previous sequence with values
                for f = sequence_start : e - 1
                    new_events(f).sequence_length = sequence_length;
                end

                % Reset sequence parameters
                sequence_length = 1;
                sequence_position = 1;
                sequence_start = e;
                previous_type = new_events(e).bonustrial;
            end
            
        end

        new_events(e).sequence_position = sequence_position;

        % Write length of last sequence
        if tidx == trial_idx(end)
            for f = sequence_start : e
                new_events(f).sequence_length = sequence_length;
            end
        end

    end

    % Check if number of trials match
    if trial_nr ~= size(positions, 1)
        error('number of trials in logfile and number of trials in event markers do not match...')
    end

    % Replace events
    EEG.event = new_events;
    EEG = eeg_checkset(EEG, 'eventconsistency');

end
