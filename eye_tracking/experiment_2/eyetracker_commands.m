% set up eyetracker
% window from:
% whichScreen = 0; 
% [window, windowRect] = Screen(whichScreen, 'OpenWindow'); 

Eyelink Initialize;


% get initialization for eyelink
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



% start eyetraking
eyetracking = 1;
if eyetracking
    % set-up eye tracker file
    recal = 0;
    el_filename = [num2str(no) '_' num2str(block) '.edf'];
    Eyelink('OpenFile', el_filename);
    EyelinkDoTrackerSetup(el);
    EyelinkDoDriftCorrection(el);
    error_start = Eyelink('StartRecording');
    while error_start~=0
        error_start = Eyelink('StartRecording');
        WaitSecs(1)
        if error_start == 0
            disp('Problem solved')
        end
    end
end


% send trigger to eyetracker
if eyetracking
    Eyelink('message',['trial_' num2str(i)])
end

% while presenting stimulus (right after screen flip) check fixation (separate function implementation)
j = 0;
%stim_timing = .200;
while (GetSecs-presentation)<stim_timing
    if eyetracking
        j=j+1;
        fixation(j) = check_fixation(el,scH,scW,siz);
    end
end

look_away(i) =  sum(~fixation); % saved per trial; we are summing across all samples recorded during that trial

% response message for eyetracker and check if need to recalibrate (another separate function)
if eyetracking
    Eyelink('message',['resp_' num2str(response(i))])
    
    
    evt = Eyelink('NewestFloatSample');
    x = mean(evt.gx);
    y = mean(evt.gy);
    x = x-scW;
    y = y-scH;
    
    if i > 4 && itrial > 4
        consec_look_away = look_away(i)>0 & look_away(i-1)>0 & look_away(i-2)>0  & look_away(i-3)>0;
    else
        consec_look_away = 0;
    end
    
    recalibrate_eyetracker;
    
end


% stop recording and save file

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

% shut down at end of experiment

if eyetracking
    Eyelink('ShutDown');
end




% stop eyetracker
try
    Eyelink('StopRecording');
    Eyelink('CloseFile');
    status=Eyelink('ReceiveFile');
    Eyelink('ShutDown');
catch
    Eyelink('ShutDown');
end
