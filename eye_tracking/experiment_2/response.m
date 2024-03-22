% function [resp_key,t1,earlyResponse] = response(window,t0,eyetracking,trial,MaxTime)
function [key_pressed,t1, t2, fixation, first_time, first_time_poss_wrong] = response(window,t0,t1,eyetracking,trial, el, first_time, ImSq, fixation_condition, block,fixation)

textcolor = [0 0 0];

confirmcolor = [0.3 0.3 0.3];
numchoices = 4;
numcolor = 1; 

% fixation = 0

% fixation_condition = 1;

spaceKeyCode = KbName('space');

   
%% GET WINDOW PARAMETERS
windowsize=Screen('Rect', window);

width=windowsize(3);
YTop=windowsize(2);
YBottom=windowsize(4);
height=YBottom - YTop;
buttonmargin=height/6;

buttonheight = height-(5*buttonmargin); % 1/4 of screen
buttonwidth = width/(2*(numchoices+ceil(numchoices/2)));
% num of buttons + half the width of each button as spacer
        
%% SET UP BUTTONS
spacerwidth=buttonwidth/2;
   
buttonbottom=(height+buttonheight)/2;
buttontop=(height-buttonheight)/2;
      
   % initialize button matrix and assign coordinates of 1st button
buttons=repmat([width/2-2.75*buttonwidth    buttontop  ...
                width/2-1.75*buttonwidth    buttonbottom], numchoices, 1);
   % adjust X coordinates (Y coordinates are constant)
for i=2:4
    buttons(i,[1 3]) = buttons(i-1,[1 3]) + buttonwidth + spacerwidth; % advance horizontally
end

%% GET USER RESPONSE (new)
earlyResponse = false;
respToBeMade = true;
% response_start_time = GetSecs

if eyetracking
    Eyelink('message',['start accept reaction response ' num2str(trial) ' Block ' num2str(block)]);
end
j = 0;

if first_time
    while respToBeMade == true
        [keyIsDown,secs,keyCode] = KbCheck;

         if keyCode(spaceKeyCode)
            earlyResponse = true;
            respToBeMade = false;
            int_t = GetSecs - t0;
    %     if keyIsDown % printing out key presses to help debug
    %         disp('key press detected')
    %         disp('keyCode:')
    %         disp(find(keyCode))
    %     end
    %    if keyIsDown
    %     if keyCode(10)
    %         ShowCursor;
    %         sca;
    %         return
    %     elseif keyCode(12)
    %         resp_key = 2;
    %         respToBeMade = false;
    %         earlyResponse = true;
    %     elseif keyCode(13)
    %         resp_key = 3;
    %         respToBeMade = false;
    %         earlyResponse = true;
    %     elseif keyCode(14)
    %         resp_key = 4;
    %         respToBeMade = false;
    %         earlyResponse = true;
    %     elseif keyCode(15)
    %         resp_key = 5;
    %         respToBeMade = false;
    %         earlyResponse = true;
    %     elseif keyCode(16)
    %         resp_key = 6;
    %         respToBeMade = false;
    %         earlyResponse = true;
    %     elseif keyCode(17)
    %         resp_key = 7;
    %         respToBeMade = false;
    %         earlyResponse = true;
        % check if 2 seconds have passed since the stimulus was presented
        elseif GetSecs - t0 > 2
            respToBeMade = false;
        end

        scH = height/2   ;         % Height coordinate of the screen's centre?
        scW = width/2    ;     % Width coordinate of the screen's centre?
        siz = 100         ;  % Radius that's "allowed" (degrees of visual angle)

        % While presenting stimulus (right after screen flip) check fixation (separate function implementation)
        if eyetracking && fixation_condition % and fixation condition (-> define this in the main code (run_experiment))
            j=j+1;
            fixation(j) = check_fixation(el,scH,scW,siz); % double check what fixation does and incorporate the arguments accordingly
        end

       % disp('While loops running')

    end
    
    if earlyResponse
        t1 = int_t;
        if eyetracking
            Eyelink('message',['first response registered ' num2str(trial) ' Block ' num2str(block)]);
        end
    else
        t1 = -999;
    end  

end
    % if there was an early response, move to the confirmation screen
    % otherwise, continue with the regular response collection
% disp('while loop done')



%     % DRAW THE BUTTONS    
%     Screen(window,'TextFont','Arial');
%     Screen(window,'TextSize',80);   
%     if numcolor ~= -1
%       for i=2:numchoices+1
%         numX = buttons(i-1,1) + (buttonwidth/3);
%         numY = height/2;
%         Screen('DrawText',window, num2str(i), numX, numY, numcolor);
%       end
%     end
% 
%     % Confirmation Screen
%     Screen(window,'TextSize',40); 
%     conf_instr = 'confirm your answer by pressing your response again';
%     conf_instr_2 = 'press any other keys to re-select your response';
%     Screen('DrawText',window, conf_instr, 200, buttonmargin+100, textcolor);  
%     Screen('DrawText',window, conf_instr_2, 250, buttonmargin+175, textcolor);  
% 
%     for i = 1:6
%         if i == (resp_key-1)
%             Screen('FillRect',window,textcolor, buttons(i,:));
%         else
%             Screen('FillRect',window,confirmcolor, buttons(i,:));
%         end
% 
%     end
% 
%     % key_pressed = resp_key;
%     % return;
% 
%     Screen(window,'TextSize',80);   
%     for i=2:numchoices+1
%         numX = buttons(i-1,1) + (buttonwidth/3);
%         numY = height/2;
%         Screen('DrawText',window, num2str(i), numX, numY, numcolor);
%       end
%     Screen('Flip',window);
%     
%     key_pressed = resp_key;
%     return;
%     
    

%t1 = GetSecs - t0;


if first_time
    
    % Present the mask
    num = randi([0, 9]);
    file_name = ['/home/neuronoodle/Documents/MIRA_JULIAN/noise/structured_noise_0',num2str(num),'.jpg';]  
    image_load = imread(file_name);
    mask_image = Screen('MakeTexture',window,image_load);
    Screen('Preference', 'SkipSyncTests', 1);
    
    [textureIndex,offScreenSize] = Screen('OpenOffscreenWindow', window, 0.5);
    Screen('FillRect', textureIndex, 0.5);
    
    Screen('DrawTexture', window, mask_image, [],ImSq);
      
    Screen('Flip', window);
    if eyetracking
        Eyelink('message',['stimulus offset ' num2str(trial) ' Block ' num2str(block)]);
    end
    
    WaitSecs(0.1);
    [textureIndex,offScreenSize] = Screen('OpenOffscreenWindow', window, 0.5);
    Screen('FillRect', textureIndex, 0.5);
    Screen('DrawTexture', window, textureIndex, [], offScreenSize);
    Screen('Flip', window);
    if eyetracking
        Eyelink('message',['mask offset ' num2str(trial) ' Block ' num2str(block)]);
    end
    WaitSecs(0.5);
end


% WRITE THE QUESTION
Screen(window,'TextFont','Arial');
Screen(window,'TextSize',40); 

Screen('DrawText',window, 'Select your response by pressing the corresponding', 220, buttonmargin+100, textcolor);  
Screen('DrawText',window, 'number on the keyboard', 400, buttonmargin+175, textcolor);  

%Screen('Flip',window,0,1);
%   (moved this to the run_experiment file to avoid multiple triggers)
%     if eyetracking
%         Eyelink('message',['stimulus offset ' num2str(trial)])
%     end
Screen('FillRect',window,textcolor, buttons');


% DRAW THE BUTTONS       
Screen(window,'TextFont','Arial');
Screen(window,'TextSize',80);   
if numcolor ~= -1
  % for i=3:numchoices+2
  for i=1:4     
    numX = buttons(i,1) + (buttonwidth/3);
    numY = height/2;
    Screen('DrawText',window, num2str(i+2), numX, numY, numcolor);
  end
end

Screen('Flip',window);

if eyetracking
    Eyelink('message',['start accept responses ' num2str(trial) ' Block ' num2str(block)]);
end




% While loop checking for response
respToBeMade = true;
while respToBeMade == true
    [keyIsDown,secs,keyCode] = KbCheck;
    if keyCode(10)
        ShowCursor;
        sca;
        return
%     elseif keyCode(12)
%         resp_key = 2;
%         respToBeMade = false;
    elseif keyCode(13)
        resp_key = 3;
        respToBeMade = false;
    elseif keyCode(14)
        resp_key = 4;
        respToBeMade = false;
    elseif keyCode(15)
        resp_key = 5;
        respToBeMade = false;
    elseif keyCode(16)
        resp_key = 6;
        respToBeMade = false;
%     elseif keyCode(17)
%         resp_key = 7;
%         respToBeMade = false;
    end
end

t2 = GetSecs - t0;

% confirmation screen follows initial choice
Screen(window,'TextSize',40); 
conf_instr = 'confirm your answer by pressing your response again';
conf_instr_2 = 'press any other keys to re-select your response';
Screen('DrawText',window, conf_instr, 200, buttonmargin+100, textcolor);  
Screen('DrawText',window, conf_instr_2, 250, buttonmargin+175, textcolor);  

for i = 1:4
    if i == (resp_key-2)
        Screen('FillRect',window,textcolor, buttons(i,:));
    else
        Screen('FillRect',window,confirmcolor, buttons(i,:));
    end

end

Screen(window,'TextSize',80);   
% for i=2:numchoices+1
for i=1:4
    numX = buttons(i,1) + (buttonwidth/3);
    numY = height/2;
    Screen('DrawText',window, num2str(i+2), numX, numY, numcolor);
end
Screen('Flip',window);

if first_time == true
    first_time_poss_wrong = true;
else 
    first_time_poss_wrong = false;
end
    
first_time = false;



key_pressed = resp_key;
% t1 = GetSecs - t0;
if eyetracking
    Eyelink('message',['first response registered ' num2str(trial) ' Block ' num2str(block)]);
end

end