function fixation = check_fixation(el,scH,scW,siz)
% Fixation returns 0 when not fixating, 1 when fixating

evt = Eyelink('NewestFloatSample');

x = evt.gx - scW;
y = evt.gy - scH;

radius = siz;

fixation = 0;

for i = 1:2  % Iterate over both eyes
    if x(i)~=el.MISSING_DATA && y(i)~=el.MISSING_DATA && evt.pa(i)>0
        distance = sqrt(x(i)^2 + y(i)^2);
        % Check if eye i is fixating
        fixation_i = ~(distance > radius);
        % If at least one eye is fixating, set fixation to 1
        if fixation_i == 1
            fixation = 1;
            break
        end
    end
end

end
