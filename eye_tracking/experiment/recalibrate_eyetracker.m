% recalibrate eyetracker

if  consec_look_away
    recal_eye = 1
    look_away(:) = 0;
    recal = recal +1;

        
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

    if recal== 1
        og_el_filename = el_filename(1:end-4);
    end
    el_filename = [og_el_filename '_' num2str(recal) '.edf'];
        
    Eyelink('OpenFile', el_filename);
    EyelinkDoTrackerSetup(el);

    Eyelink('StartRecording');
    
end