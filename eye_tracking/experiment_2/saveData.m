
% Save data function
function saveData (data, subjectNumber)
    header = {'SubjectNumber','Trial_Num','Response','image_Name','Object_Type','Accuracy','Response_Time_1','Response_Time_2','Response_Time_3','block','image_folder', 'target_n', 'dist_n', 'View', 'Task', 'Aborted'};
    %header = {'SubjectNumber','Trial_Num','Response','image_Name','Object_Type','Accuracy','Response_Time_1','Response_Time_2','block','image_folder', 'target_n', 'dist_n', 'Condition'};
    data_table = cell2table(data, 'VariableNames',header);

    exp_data = strcat('data/','subject_',char(data{1,1}),'.csv');
    writetable(data_table, exp_data);
end
