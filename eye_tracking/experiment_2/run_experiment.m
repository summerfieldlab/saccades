
Screen('CloseAll');
Screen('Preference', 'SkipSyncTests', 1);
close all;
clearvars;
addpath(genpath('CogToolbox'));
KbName('UnifyKeyNames');

practice = 1;  
n_trial_total = 288;
eyetracking = 1;
debug_fixation = 0;
abort_trigger = 0;
cont_var = 0;
recal_eye = 0;
n_practice_trials = 32;

global el;
global fixation_condition;

%%
PsychDefaultSetup(2); % Some default settings

data = cell(n_trial_total,15); % Trial numer * number of variables to collect

numColumns = 18;
numRows = 0;
emptyData = cell(numRows, numColumns);
variableNames = {'Subject_Number','Trial_Num','Response','image_Name','Object_Type','Accuracy','Response_Time_1','Response_Time_2','Response_Time_3','block','image_folder', 'target_n', 'dist_n', 'target_coords', 'dist_coords','View', 'Task', 'Aborted'};
Data = cell2table(emptyData, 'VariableNames', variableNames);

prompt = ('Subject Number:');%Collect information for the experiment
title = 'Experiment information';
definput = {''};

resu = inputdlg(prompt, title, [1 20], definput);
subject_n_str = resu{1};
subject_n = str2double(resu{1});

data(1:n_trial_total) = repmat(inputdlg(prompt,title,[1,20],definput),n_trial_total,1);

screens = Screen('Screens');

screenNumber = max(screens);

gray = GrayIndex(screenNumber);

% Open an on screen window using PsychImaging and color it white
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, gray);
% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
% Get the centre coordinate of the window
[xCentre, yCentre] = RectCenter(windowRect);

% Set keycodes SmCircle_sqused
keyCodes = [12:17];

% Set the square which is image is in
stimuli_pixels = screenYpixels - 280;
ImSq = [0 0 stimuli_pixels stimuli_pixels];
[ImSq, xOffsetsigB, yOffsetsigB] = CenterRect(ImSq, windowRect);
InsSq = [0 0 screenXpixels screenYpixels];
[InsSq, xOffsetsigB, yOffsetsigB] = CenterRect(InsSq, windowRect);

% Set text format
Screen('TextSize', window ,24);

% Hide cursor
HideCursor(window);

%%

addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1}))]);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/instructions']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_count-all_A_main']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_count-all_A_practice']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_count-all_B_main']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_count-all_B_practice']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_ignore-distractors_A_main']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_ignore-distractors_A_practice']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_ignore-distractors_B_main']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/participant_',num2str(char(data{1,1})),'/num3-6_ESUZFCKJsame_ignore-distractors_B_practice']);
addpath(['/home/neuronoodle/Documents/MIRA_JULIAN/noise']);

% Load Images
folder_1 = {};

folder_1{1} = strcat('num3-6_ESUZFCKJsame_ignore-distractors_A_main');
folder_1{2} = strcat('num3-6_ESUZFCKJsame_count-all_A_main');
folder_1{3} = strcat('num3-6_ESUZFCKJsame_ignore-distractors_B_main');
folder_1{4} = strcat('num3-6_ESUZFCKJsame_count-all_B_main');

folder = folder_1  ;

condition = {'IGNORE As (Eyes freely moving)','COUNT ALL (Eyes freely moving)','IGNORE As (Eyes fixed)','COUNT ALL (Eyes fixed)'};

trial_list = cell(4, 2);  % Now a cell array to hold two halves of each dataset
condition_list = zeros(4,72);

% Create a shuffled trial order list for each condition
for j = 1:4
    trial_order = randperm(72,72);
    half_index = round(length(trial_order) / 2);
    trial_list{j, 1} = trial_order(1:half_index);  % First half
    trial_list{j, 2} = trial_order(half_index+1:end);  % Second half
    condition_list(j,:) = j*ones(1,72);  % assign condition j to all its trials
end

% Randomizing the order of conditions for the first and second halves
first_half_order = randperm(4);
second_half_order = randperm(4);

% Initializing the final order lists
final_condition_order = zeros(1, 8*36);
final_trial_order_list = cell(1, 8);  % Now a cell array to hold two halves of each dataset

% Assign the first half trials to the first 4 blocks
for i = 1:4
    final_condition_order((i-1)*36+1:i*36) = condition_list(first_half_order(i),1:36);
    final_trial_order_list{i} = trial_list{first_half_order(i), 1};
end

% Assign the second half trials to the next 4 blocks
for i = 1:4
    final_condition_order((i+3)*36+1:(i+4)*36) = condition_list(second_half_order(i),37:72);
    final_trial_order_list{i+4} = trial_list{second_half_order(i), 2};
end

factor_map_view = [0, 0, 1, 1];
factor_map_task = [0, 1, 0, 1];
factor_task_list = {'ignore_distractors', 'count_all'};
factor_view_list = {'free', 'fixed'};
factor_task = factor_map_task(final_condition_order(:));
factor_view = factor_map_view(final_condition_order(:));


% Practice trial and condition allocation

practice_trial_list = zeros(1,32);
for i = 1:4
    temp_list = randperm(72,8);
    practice_trial_list(1, ((i-1)*8 + 1):(i*8)) = temp_list;
end

practice_condition_order = [repmat(2, 1, 8), repmat(1, 1, 8), repmat(4, 1, 8), repmat(3, 1, 8)]  ;

practice_factor_task = factor_map_task(practice_condition_order(:));
practice_factor_view = factor_map_view(practice_condition_order(:));

% Practice folder allocation

practice_folder_1{1} = strcat('num3-6_ESUZFCKJsame_ignore-distractors_A_practice');
practice_folder_1{2} = strcat('num3-6_ESUZFCKJsame_count-all_A_practice');
practice_folder_1{3} = strcat('num3-6_ESUZFCKJsame_ignore-distractors_B_practice');
practice_folder_1{4} = strcat('num3-6_ESUZFCKJsame_count-all_B_practice');

practice_folder = practice_folder_1;

%% Eyetracker intialization

if eyetracking
    Eyelink Initialize;
    
    el = EyelinkInitDefaults(window);
    
    if ~EyelinkInit(0, 1)
        fprintf('Eyelink Init aborted.\n');
        cleanup;  % cleanup function
        return;
    end
    
    Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');
    Eyelink('Command', 'calibration_area_proportion 0.7 0.7');
    Eyelink('Command', 'validation_area_proportion 0.7 0.7');
    Eyelink('Command', 'calibration_corner_scaling 0.6 0.6');
    Eyelink('Command', 'validation_corner_scaling 0.6 0.6');
end

%% Start Instructions

% Display Instruction Start
Screen('TextSize', window, 45); % Set the text size
Screen('TextFont', window, 'Arial'); % Set the font
instruction_start_text = 'In this experiment, you will be presented\nwith images containing letters, and your\ntask is to count the number of specified\nitems within the image.\n\nThis is an eye-tracking experiment, please\ntry to keep your head still during the\nexperiment. You may ask for rests or quit\nthis study at any time during the\nexperiment.\n\nPress any key to continue.'; % Define your instruction text
DrawFormattedText(window, instruction_start_text, 'center', 'center', [255, 255, 255]); % Draw the text in white
Screen('Flip', window); % Flip the buffer to display the text
KbStrokeWait; % Wait for a key press

% Display Instruction Start 2
instruction_start2_text = 'What to expect for the task (1)\n\nAt the beginning of each block, you will be instructed\nto either count all letters in the image (COUNT ALL)\nor to count all letters except the letter A (IGNORE As).\n\nOn half the blocks, you will be instructed to keep\nyour gaze fixated on a red cross in the centre\nof the screen while you perform the counting task\n(Eyes fixed). This means you will have to use your\nperipheral vision to perform the task.\n\nIf your gaze deviates too far from the fixation cross,\nyou will receive feedback and that trial\nwill be redone at the end of the block.\n\nThe rest of the time, you can move your eyes freely\nwhile performing the counting task (Eyes freely moving).';
DrawFormattedText(window, instruction_start2_text, 'center', 'center', [255, 255, 255]); % Draw the text in white
Screen('Flip', window); % Flip the buffer to display the text
KbStrokeWait; % Wait for a key press

% Display Instruction Start 3
instruction_start3_text = 'What to expect for the task (2)\n\nPlease respond as soon as\nyou know the correct answer\nby pressing the spacebar.\nThis will bring you to a response screen.\n\nIf you do not press the spacebar \nwithin 2 seconds, you will\nautomatically move to the response screen.\nPlease only press the spacebar\nonce you know your response.';
DrawFormattedText(window, instruction_start3_text, 'center', 'center', [255, 255, 255]); % Draw the text in white
Screen('Flip', window); % Flip the buffer to display the text
KbStrokeWait; % Wait for a key press

% Display Instruction Start 4
instruction_start4_text = 'First, you will go through practice trials.\nThese consist of 4 blocks of 8 trials.\n\nYou will be instructed what to do in\nthe beginning of each block.\n\nPlease press any key to continue\nonce you understand the instructions.\n\nPress any key to continue.'; % Define your instruction text
DrawFormattedText(window, instruction_start4_text, 'center', 'center', [255, 255, 255]); % Draw the text in white
Screen('Flip', window); % Flip the buffer to display the text
KbStrokeWait; % Wait for a key press


%% Practice trials

if practice

    Practice_Data = cell2table(emptyData, 'VariableNames', variableNames);
    
    % Start eyetracking 
    block = 0 % corresponds to practice trials

    if eyetracking
        % Set-up eye tracker file
        recal = 0;
        el_filename = [char(data{1,1}) '_pr' '.edf'];
        Eyelink('OpenFile', el_filename);
        EyelinkDoTrackerSetup(el);
        EyelinkDoDriftCorrection(el);
        error_start = Eyelink('StartRecording');

        while error_start~=0
            error_start = Eyelink('StartRecording');
            WaitSecs(1);
            if error_start == 0
                disp('Problem solved');
            end
        end
    end
    
    fixation = {};
        
    for trial = 1:n_practice_trials % Depends on actual number of trials (here: 32)
        
        look_away(trial) = 0;
        
        % Send trigger to eyetracker
        if eyetracking
            Eyelink('message',['practice trial start ' num2str(trial)]);
        end      
        
        if trial <= 8
            condition_prac = 2;
        elseif trial > 8 && trial <= 16
            condition_prac = 1;
        elseif trial > 16 && trial <= 24
            condition_prac = 4;
        else
            condition_prac = 3;
        end
        
        baseDir = '/home/neuronoodle/Documents/MIRA_JULIAN/participant_';
            
        participantNum = num2str(char(data{1,1}));
        fullDir = [baseDir, participantNum];
        filePattern = fullfile(fullDir, char(practice_folder(condition_prac)), filesep, '*.png');
        myFiles = dir(filePattern); 
        myFilestemp = struct2table(myFiles);

        % Extract the trial numbers from the 'name' field
        trialNumbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), {myFiles.name});

        % Create a new array sorted by trial numbers
        [~, order] = sort(trialNumbers);
        myFiles = myFilestemp(order,:);
        myFiles = table2struct(myFiles);

        % Load meta data
        f = char(practice_folder(condition_prac));
        meta_data_name = strcat('image_metadata_',f,'.mat');
        meta_data = load(meta_data_name);
     
        % Block start message
        if ismember (trial, [1,9,17,25]) || recal_eye
            recal_eye = 0
            condition_opt = condition(condition_prac);
            Screen(window, 'TextSize',60);
            fulltext = ['Practice Block:' '\n\n' char(condition_opt) '\n\n(Press any key to continue.)'];
            DrawFormattedText(window,fulltext,'center','center',[255 255 255],[],[],[],2);
            Screen('Flip', window);
            KbStrokeWait;
        end
        
        % Converging circle
        for j = 0:51
            circle_size = screenYpixels - j*10 - 500;
            Circle_sq = [0 0 circle_size circle_size];
            SmCircle_sq = [0 0 (circle_size-10) (circle_size-10)];
            [Circle_sq, xOffsetsigB, yOffsetsigB] = CenterRect(Circle_sq, windowRect);
            [SmCircle_sq, xOffsetsigB, yOffsetsigB] = CenterRect(SmCircle_sq, windowRect);
            Screen('FillOval',window, [255 255 255], Circle_sq);
            Screen('FillOval',window, gray, SmCircle_sq);
            Screen('Flip', window);
            WaitSecs(0.01);
        end
           
        % Display stimuli
        trial_num = practice_trial_list(trial);
        trial_index = str2double(extractBefore(extractAfter(myFiles(trial_num).name,'_'),'.png'))+1;
      
        stimuli_image = fullfile(char(folder(practice_condition_order(trial))), myFiles(trial_num).name);
        
        stimuli_image = strrep(stimuli_image,'._','');
        disp(['Attempting to load image: ', stimuli_image]); % Add this line
        image_load = imread(stimuli_image);
        
        stimuli_image = Screen('MakeTexture',window,image_load);
        
        Screen('Preference', 'SkipSyncTests', 1);
        Screen('DrawTexture', window, stimuli_image, [],ImSq);
        
        % Check if the condition is 3 or 4 (eyes fixed)
        if ismember(condition_prac, [3,4]) || debug_fixation  
            crossLength = 15;
            crossCoords = [-crossLength, crossLength, 0, 0; 0, 0, -crossLength, crossLength];
            crossColor = [255, 0, 0];
            Screen('DrawLines', window, crossCoords, 3, crossColor, [xCentre, yCentre]);  
        end
        
        Screen('Flip', window);
        presentation = GetSecs;
        if eyetracking
            Eyelink('message',['stimulus onset ' num2str(trial)]);
        end
        
        % Check for fixation_condition
        if ismember(condition_prac, [3,4]) || debug_fixation
            fixation_condition = 1;
        else
            fixation_condition = 0;
        end
        
        
        
        % Collect stimulus offset
        
        t0 = GetSecs;
        
        % Collect response
        response_Confirmed = false; %false;
        first_time = true;
        t1 = 99999;
        fixation{trial} = 0;
        
        while response_Confirmed == false
            [key_pressed,t1,t2, fixation{trial},first_time,first_time_poss_wrong] = response(window,t0,t1,eyetracking,trial, el, first_time, ImSq, fixation_condition, block, fixation{trial});
            disp(sum(~fixation{trial}));
            
            if fixation_condition
                look_away(trial) =  sum(~fixation{trial}); % Saved per trial; we are summing across all samples recorded during that trial
                
                if ~eyetracking
                    look_away(trial) = false ;
                end       
                
                if abort_trigger && trial > 10           % Also for debugging (to test aborting of only the first 2 trials)
                    abort_trigger = 0 ;
                    look_away(trial) = false ;
                    cont_var = 0 ;
                end
                
                if abort_trigger == true
                    look_away(trial) = true ; % For debugging
                end
                
                if look_away(trial)
                                    
                    cont_var = true
                    
                    disp(['Participant looked away more than twice during trial ', num2str(trial), ',aborting trial.']);
                    
                end
            end
            
            
            [secs, keyCode, deltaSecs] = KbWait([], 3);
            keycode = keyCodes(key_pressed-1);
            if keyCode(keycode)
                key_press = key_pressed;
                imagename = char(myFiles(trial_num).name);
                response_Confirmed = true;
                t3 = GetSecs - t0;
                disp(['response_Confirmed: ', num2str(key_pressed)]);
            end
        end

        if eyetracking
            Eyelink('message',['confirmation response registered ' num2str(trial)]);
        end
        
        if cont_var == true        
            Screen('Preference', 'SkipSyncTests', 1);
            Screen(window,'TextFont','Arial');
            Screen(window,'TextSize',60);
            DrawFormattedText(window, 'Participant looked away from center.\nTrial would have been aborted.\n\n Press any key to continue.','center', 'center', [255 0 0],[],[],[],2);
            Screen('Flip', window);
            KbStrokeWait;
            
            newRow = cell2table({subject_n, trial, key_press, imagename, char(condition_opt), -999, -999, -999, -999, block, char(practice_folder(condition_prac)), meta_data.numerosity_target(trial_index), meta_data.numerosity_dist(trial_index), meta_data.target_coords_scaled(trial_index), meta_data.distract_coords_scaled(trial_index), practice_factor_view(trial), practice_factor_task(trial), 1}, 'VariableNames', variableNames);
            Practice_Data = [Practice_Data; newRow];
            
            if eyetracking
                Eyelink('message',['trial aborted ' num2str(trial)]);
            end
            
            cont_var = false;
            
            if eyetracking && fixation_condition
                if trial > 7
                    consec_look_away = look_away(trial)>0 & look_away(trial-1)>0 & look_away(trial-2)>0  & look_away(trial-3)>0 & look_away(trial-4)>0 & look_away(trial-5)>0 & look_away(trial-6)>0;
                else
                    consec_look_away = 0;
                end
                
                recalibrate_eyetracker; % Closes the current file, transfers it, and opens the calibration sequence

            end

            continue
        end
                    
        % Display feedback
        Screen(window,'TextFont','Arial');
        Screen(window,'TextSize',80);
        
        if ismember (condition_prac, [1,3])
            if key_pressed == meta_data.numerosity_target(trial_index)
                DrawFormattedText(window, 'correct','center', 'center', [0 255 0],[],[],[],2);
                answer = 1;
            else
                DrawFormattedText(window, 'incorrect','center', 'center', [255 0 0],[],[],[],2);
                answer = 0;
                
            end
        else
            if key_pressed == meta_data.numerosity_target(trial_index) + meta_data.numerosity_dist(trial_index)
                DrawFormattedText(window, 'correct','center', 'center', [0 255 0],[],[],[],2);
                answer = 1;
                
            else
                DrawFormattedText(window, 'incorrect','center', 'center', [255 0 0],[],[],[],2);
                answer = 0;
                
            end
        end
        
        
        newRow = cell2table({subject_n, trial, key_press, imagename, char(condition_opt), answer, t1, t2, t3, block, char(practice_folder(condition_prac)), meta_data.numerosity_target(trial_index), meta_data.numerosity_dist(trial_index), meta_data.target_coords_scaled(trial_index), meta_data.distract_coords_scaled(trial_index), practice_factor_view(trial), practice_factor_task(trial), 0}, 'VariableNames', variableNames);
        Practice_Data = [Practice_Data; newRow];
        
        
        Screen('Flip', window);
        if eyetracking
            Eyelink('message',['feedback onset ' num2str(trial)]);
        end
        WaitSecs(1);
        
        % Gap Between Intervals
        
        Screen('Flip', window);
        if eyetracking
            Eyelink('message',['feedback offset ' num2str(trial)]);
        end
        WaitSecs(1);
             
    end
    
    %%
    
    % Save the .csv data
    
    csvname = strcat('practice_data/','practice_subject_',char(data{1,1}),'.csv');
    writetable(Practice_Data, csvname);
    
    
    % Stop eyetracking
    
	if eyetracking
        Eyelink('StopRecording');
        Eyelink('CloseFile');

        try
            fprintf('Receiving data file ''%s''\n', el_filename );
            status=Eyelink('ReceiveFile');
            if status > 0
                fprintf('ReceiveFile status %d\n', status);
            end
            if 2==exist(el_filename, 'file')
                fprintf('Data file ''%s'' can be found in ''%s''\n', el_filename, pwd );
            end
        catch rdf
            fprintf('Problem receiving data file ''%s''\n', el_filename );
            rdf;

        end
	end
        
end

%% End of practice, start actual trials
Screen('Preference', 'SkipSyncTests', 1);

% Display Instruction Start Experiment
instruction_startexp_text = 'This is the end of practice.\n\nPress any keys to start the first \n block of the experiment.'; % Define your instruction text
DrawFormattedText(window, instruction_startexp_text, 'center', 'center', [255, 255, 255]); % Draw the text in white

Screen('Flip', window); % Flip the buffer to display the text
WaitSecs(1);
KbStrokeWait;

%%
% Run Trial

Screen('Preference', 'SkipSyncTests', 1);

num_trial = 36; %10;
num_num_trial = num_trial;

num_block = 8;
for block = 1:num_block
    
    num_trial = num_num_trial; % reset num_trial for the new block
    
    % Start eyetraking
    
    if mod(block, 2) == 1
        if eyetracking
            % Set-up eye tracker file
            recal = 0;
            el_filename = [char(data{1,1}) '_' num2str(block) '.edf'];
            Eyelink('OpenFile', el_filename);
            EyelinkDoTrackerSetup(el);
            EyelinkDoDriftCorrection(el);
            error_start = Eyelink('StartRecording');
            
            while error_start~=0
                error_start = Eyelink('StartRecording');
                WaitSecs(1);
                if error_start == 0
                    disp('Problem solved');
                end
            end
        end
    end
    
    Screen('Preference', 'SkipSyncTests', 1);
    Screen(window,'TextFont','Arial');
    Screen(window,'TextSize',50);
    
    if ismember (block, [2,4,6])
        DrawFormattedText(window, 'This is the end of the block.\n\nPress any key to start the next one.', 'center', 'center', [255, 255, 255]);
    else
        DrawFormattedText(window, 'Press anything to start the block.', 'center', 'center', [255, 255, 255]); % text in white
    end
    
    Screen('Flip', window);
    KbStrokeWait;
    
    % Load images for the block
    if block > 2
        b = block - 2;
    else
        b = block;
    end
    
    
    baseDir = '/home/neuronoodle/Documents/MIRA_JULIAN/participant_';
    participantNum = num2str(char(data{1,1}));
    fullDir = [baseDir, participantNum];
    filePattern = fullfile(fullDir, char(folder(final_condition_order(block*36))), filesep, '*.png');
    myFiles = dir(filePattern); % the folder in which images exist
    myFilestemp = struct2table(myFiles);
    
    
    
    % Assuming that myFilestemp is your 72x1 struct array
    % Extract the trial numbers from the 'name' field
    trialNumbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), {myFiles.name});

    % Create a new array sorted by trial numbers
    [~, order] = sort(trialNumbers);
    myFiles = myFilestemp(order,:);
    myFiles = table2struct(myFiles);
    

    start_ = (block-1)*36+1;
    end_ = block*36;
    condition_block = final_condition_order(start_:end_);
    
    % Load meta data
    f = char(folder(final_condition_order(block*36)));
    meta_data_name = strcat('image_metadata_',f,'.mat');
    meta_data = load(meta_data_name);
    
    % Display Condition of the Block
    condition_opt = condition(final_condition_order(block*36));
    Screen('Preference', 'SkipSyncTests', 1);
    Screen(window, 'TextSize',60);
    fulltext = ['Block ' num2str(block) ':\n\n' char(condition_opt) '\n\n(Press any key to continue.)'];
    DrawFormattedText(window,fulltext,'center','center',[255 255 255],[],[],[],2);
    Screen('Flip', window);
    KbStrokeWait;
    
    block_trials = final_trial_order_list{block};
    notdone = 1;
    trial = 1;
    fixation = {};
    
    while notdone && trial < num_trial+1
        
        
        if recal_eye
            recal_eye = 0
            Screen('Preference', 'SkipSyncTests', 1);
            Screen(window, 'TextSize',60);
            fulltext = ['Block ' num2str(block) ':\n\n' char(condition_opt) '\n\n(Press any key to continue.)'];
            DrawFormattedText(window,fulltext,'center','center',[255 255 255],[],[],[],2);
            %    data{pos_data, 5} = char(condition_opt);
            Screen('Flip', window);
            KbStrokeWait;
        end
            
        look_away(trial) = 0;
        
        pos_data = (block-1)*36 + trial;
	
        % Send trigger to eyetracker
        if eyetracking
            Eyelink('message',['trial start ' num2str(trial) ' Block ' num2str(block)]);
        end
        
        data{pos_data, 2} = trial; % Record trial number
        data{pos_data, 5} = char(condition_opt);
        
        % Access the trial number for the current block and trial
        trial_num = block_trials(trial); % This will be a number from 1-36
        trial_index = str2double(extractBefore(extractAfter(myFiles(trial_num).name,'_'),'.png'))+1;

        
        % Converging circle
        if eyetracking
            Eyelink('message',['start fixation ' num2str(trial) ' Block ' num2str(block)]);
        end
        for j = 0:51
            circle_size = screenYpixels - j*10 - 500;
            Circle_sq = [0 0 circle_size circle_size];
            SmCircle_sq = [0 0 (circle_size-10) (circle_size-10)];
            [Circle_sq, xOffsetsigB, yOffsetsigB] = CenterRect(Circle_sq, windowRect);
            [SmCircle_sq, xOffsetsigB, yOffsetsigB] = CenterRect(SmCircle_sq, windowRect);
            Screen('FillOval',window, [255 255 255], Circle_sq);
            Screen('FillOval',window, gray, SmCircle_sq);
            Screen('Flip', window);
            WaitSecs(0.01);
        end
        if eyetracking
            Eyelink('message',['end fixation ' num2str(trial) ' Block ' num2str(block)]);
        end
        
        
        % Display Stimuli
        
        stimuli_image = fullfile(char(folder(final_condition_order(block*36))), myFiles(trial_num).name);

        stimuli_image = strrep(stimuli_image,'._','');
        disp(['Attempting to load image: ', stimuli_image]);
        image_load = imread(stimuli_image);
        
        stimuli_image = Screen('MakeTexture',window,image_load);
        
        Screen('Preference', 'SkipSyncTests', 1);
        Screen('DrawTexture', window, stimuli_image, [],ImSq);
        
        % Check if the condition is 3 or 4 (eyes fixed)
        if ismember(condition_block(trial), [3,4]) || debug_fixation        
            crossLength = 15;
            crossCoords = [-crossLength, crossLength, 0, 0; 0, 0, -crossLength, crossLength];
            crossColor = [255, 0, 0];
            Screen('DrawLines', window, crossCoords, 3, crossColor, [xCentre, yCentre]);
        end
        
        Screen('Flip', window);
        presentation = GetSecs;
        if eyetracking
            Eyelink('message',['stimulus onset ' num2str(trial) ' Block ' num2str(block)]);
        end
        
        % Check for fixation_condition
        if ismember(condition_block(trial), [3,4]) || debug_fixation
            fixation_condition = 1;
        else
            fixation_condition = 0;
        end
        
        
        
        %%
        
        % Collect stimulus offset
        t0 = GetSecs;
        
        % "Aborted" value is set to 0.
        data{pos_data, 16} = 0;
        
        % Collect Response
        response_Confirmed = false;
        first_time = true;
        t1 = 99999;
	    fixation{trial} = 0;
        
        while response_Confirmed == false
            [key_pressed,t1,t2, fixation{trial},first_time,first_time_poss_wrong] = response(window,t0,t1,eyetracking,trial, el, first_time, ImSq, fixation_condition, block, fixation{trial});
            disp(sum(~fixation{trial}));
            % %

            if fixation_condition
                look_away(trial) =  sum(~fixation{trial}); % Saved per trial; we are summing across all samples recorded during that trial

                if ~eyetracking
                    look_away(trial) = false ;
                end

                if abort_trigger && trial > 10           % Also for debugging
                        abort_trigger = 0 ;
                        look_away(trial) = false ;
                        cont_var = 0 ;
                end

                if abort_trigger == true
                    look_away(trial) = true ; % For debugging
                end

                if look_away(trial)
                    
                    if first_time_poss_wrong
                        block_trials(end+1) = block_trials(trial); % Append the trial at the end of the block
                        condition_block(end+1) = condition_block(trial); % Append condition block at the end of the block
                        num_trial = num_trial + 1;

                        disp(['Participant looked away more than twice during trial ', num2str(trial), ',aborting trial.']);
                        % Save "Trial-Aborted" variable as 1
                        data{pos_data, 16} = 1;
                    end

                    cont_var = 1;

                end
            end
            
            [secs, keyCode, deltaSecs] = KbWait([], 3);
            keycode = keyCodes(key_pressed-1);
            if keyCode(keycode)
                key_press = key_pressed;
                imagename = char(myFiles(trial_num).name);
                data{pos_data, 3} = key_pressed;
                data{pos_data, 4} = char(myFiles(trial_num).name);
                response_Confirmed = true;
                t3 = GetSecs - t0;
                disp(['response_Confirmed: ', num2str(key_pressed)]); % Add this line
            end
        end
        
        
        if cont_var == true
            
            Screen('Preference', 'SkipSyncTests', 1);
            Screen(window,'TextFont','Arial');
            Screen(window,'TextSize',60);
            DrawFormattedText(window, 'Participant looked away from center.\nTrial aborted.\n\n Press any key to continue.','center', 'center', [255 0 0],[],[],[],2);
            Screen('Flip', window);
            KbStrokeWait;
            newRow = cell2table({subject_n, trial, key_press, imagename, char(condition_opt), -999, -999, -999, -999, block, char(folder(final_condition_order(block*36))), meta_data.numerosity_target(trial_index), meta_data.numerosity_dist(trial_index), meta_data.target_coords_scaled(trial_index), meta_data.distract_coords_scaled(trial_index), factor_view(block*36), factor_task(block*36), 1}, 'VariableNames', variableNames);
            Data = [Data; newRow];
            
            data{pos_data, 7} = [];
            data{pos_data, 8} = [];
            data{pos_data, 9} = [];
            data{pos_data, 10} = block;
            data{pos_data, 11} = char(folder(condition_block(trial)));
            data{pos_data, 12} = meta_data.numerosity_target(trial_index);
            data{pos_data, 13} = meta_data.numerosity_dist(trial_index);
            data{pos_data, 14} = factor_view(block*36);
            data{pos_data, 15} = factor_task(block*36);
                 
            if eyetracking
                Eyelink('message',['trial aborted ' num2str(trial) ' Block ' num2str(block)]);
            end
            cont_var = false;
            
            if eyetracking && fixation_condition
                if trial > 7
                	consec_look_away = look_away(trial)>0 & look_away(trial-1)>0 & look_away(trial-2)>0  & look_away(trial-3)>0 & look_away(trial-4)>0 & look_away(trial-5)>0 & look_away(trial-6)>0;
                else
                    consec_look_away = 0;
                end
                
                recalibrate_eyetracker; % Closes the current file and transfers it and opens the calibration sequence
            end
            
            trial = trial + 1;
            
            continue
        end
        
        
        if eyetracking
            Eyelink('message',['second response registered ' num2str(trial) ' Block ' num2str(block)]);
        end
        
        
        
        
        % Display feedback
        
        Screen('Preference', 'SkipSyncTests', 1);
        Screen(window,'TextFont','Arial');
        Screen(window,'TextSize',80);
        
        disp(meta_data.numerosity_target(trial_index));
        disp(meta_data.numerosity_target(trial_index) + meta_data.numerosity_dist(trial_index));
        
        if ismember(condition_block(trial), [1,3])
            if key_pressed == meta_data.numerosity_target(trial_index)
                DrawFormattedText(window, 'correct','center', 'center', [0 255 0],[],[],[],2);
                answer = 1;
                data{pos_data, 6} = 1;
            else
                DrawFormattedText(window, 'incorrect','center', 'center', [255 0 0],[],[],[],2);
                answer = 0;
                data{pos_data, 6} = 0;
            end
        else
            if key_pressed == meta_data.numerosity_target(trial_index) + meta_data.numerosity_dist(trial_index)
                DrawFormattedText(window, 'correct','center', 'center', [0 255 0],[],[],[],2);
                answer = 1;
                data{pos_data, 6} = 1;
            else
                DrawFormattedText(window, 'incorrect','center', 'center', [255 0 0],[],[],[],2);
                answer = 0;
                data{pos_data, 6} = 0;
            end
        end
        
        
        
        %%
        
        newRow = cell2table({subject_n, trial, key_press, imagename, char(condition_opt), answer, t1, t2, t3, block, char(folder(condition_block(trial))), meta_data.numerosity_target(trial_index), meta_data.numerosity_dist(trial_index), meta_data.target_coords_scaled(trial_index), meta_data.distract_coords_scaled(trial_index), factor_view(block*36), factor_task(block*36), 0}, 'VariableNames', variableNames);
        Data = [Data; newRow];
        
        data{pos_data, 7} = t1;
        data{pos_data, 8} = t2;
        data{pos_data, 9} = t3;
        data{pos_data, 10} = block;
        data{pos_data, 11} = char(folder(condition_block(1)));
        data{pos_data, 12} = meta_data.numerosity_target(trial_index);
        data{pos_data, 13} = meta_data.numerosity_dist(trial_index);
        data{pos_data, 14} = factor_view(block*36);
        data{pos_data, 15} = factor_task(block*36);
        
        
        Screen('Flip', window);
        if eyetracking
            Eyelink('message',['feedback onset ' num2str(trial) ' Block ' num2str(block)]);
        end
        WaitSecs(1);
        
        % Gap Between Intervals
        
        Screen('Flip', window);
        if eyetracking
            Eyelink('message',['feedback offset ' num2str(trial) ' Block ' num2str(block)]);
        end
        WaitSecs(1);
        
        if length(block_trials) == trial
            notdone = 0;
        end
        
        trial = trial + 1;
        
        Screen('Flip', window);
    end
    
    
    % End of block
    
    % Save the .csv data
    saveData(data,data{1,1});
    
    % Save the .csv data
    csvname = strcat('data/','subject_',char(data{1,1}),'.csv');
    writetable(Data, csvname);
    
    % Save fixation
    fix_filename = strcat('data/','fixations_subject_',char(data{1,1}),'_',char(num2str(block)),'.mat');
    save(fix_filename, 'fixation');
    
    % Set textsize
    Screen('TextSize', window ,40);
    
    if block == num_block
        % Display End Instructions
        
        instruction_end_text = 'This is the end of the experiment.\n\nThank you for participating!';
        DrawFormattedText(window, instruction_end_text, 'center', 'center', [255, 255, 255]); % Draw the text in white
        Screen('Flip', window);
        
        if eyetracking
            Eyelink('StopRecording');
            Eyelink('CloseFile');

            try
                fprintf('Receiving data file ''%s''\n', el_filename );
                status=Eyelink('ReceiveFile');
                if status > 0
                    fprintf('ReceiveFile status %d\n', status);
                end
                if 2==exist(el_filename, 'file')
                    fprintf('Data file ''%s'' can be found in ''%s''\n', el_filename, pwd );
                end
            catch rdf
                fprintf('Problem receiving data file ''%s''\n', el_filename );
                rdf;

            end
        end
        
        KbStrokeWait;
        
    elseif ismember(block, [2,4,6])
        block_end = 'This is the end of the block, please take a rest';
        DrawFormattedText(window, block_end,'center', 'center', [255 255 255],[],[],[],2);
        Screen('Flip', window);
        
        if eyetracking
            Eyelink('StopRecording');
            Eyelink('CloseFile');

            try
                fprintf('Receiving data file ''%s''\n', el_filename );
                status=Eyelink('ReceiveFile');
                if status > 0
                    fprintf('ReceiveFile status %d\n', status);
                end
                if 2==exist(el_filename, 'file')
                    fprintf('Data file ''%s'' can be found in ''%s''\n', el_filename, pwd );
                end
            catch rdf
                fprintf('Problem receiving data file ''%s''\n', el_filename );
                rdf;

            end
        end
    
        KbStrokeWait;
        
    end
end


%%
% Close screen
ShowCursor(window);
sca;
Screen('CloseAll');


% Shut down the eyetracker at end of experiment
if eyetracking
    Eyelink('ShutDown');
end
