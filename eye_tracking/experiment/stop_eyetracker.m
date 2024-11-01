% stop eyetracker
try
    Eyelink('StopRecording');
    Eyelink('CloseFile');
    status=Eyelink('ReceiveFile');
    Eyelink('ShutDown');
catch
    Eyelink('ShutDown');
end