function[EEG] = bocotilt_event_coding(EEG, RESPS)

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
        if (strcmpi(EEG.event(e).type(1), {'S'}) & ismember(str2num(EEG.event(e).type(2 : 4)), [1 : 32]))
    
            trial_nr = trial_nr + 1;

            % Get event code
            ecode = str2num(EEG.event(e).type(2 : 4));

            % Decode bonustrial
            if ecode <= 16
                bonustrial = 1;
            else
                bonustrial = 0;
            end

            % Dedode task
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
            left_rt = min(find(resps_left(EEG.event(e).latency + 1200 : EEG.event(e).latency + 1200 + (maxrt / (1000 / EEG.srate))) >= critval_responses)) * (1000 / EEG.srate);
            right_rt = min(find(resps_right(EEG.event(e).latency + 1200 : EEG.event(e).latency + 1200 + (maxrt / (1000 / EEG.srate))) >= critval_responses)) * (1000 / EEG.srate);
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
        
        end

        % If event is boundary event...
        if strcmpi(EEG.event(e).type, 'boundary')
            stimulus_type = 'boundary';
        end

    end

    % Replace events
    EEG.event = new_events;
    EEG = eeg_checkset(EEG, 'eventconsistency');

end