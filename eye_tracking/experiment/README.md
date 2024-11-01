**Saccades - Eyetracking Experiment**
This repository contains the MATLAB scripts necessary to run the behavioral and eye-tracking experiment associated with the "Zero-shot counting with a dual-stream neural network model" project and paper (Thompson et al., 2024). For questions, please contact the authors.

The experiment consists of a visual counting task while participants' gaze position is tracked. In a 2x2 within-subjects design, participants are shown images and instructed to either "simply count" or to "ignore distracters", while their gaze is either free or fixed to the center of the screen.


**Requirements**
- MATLAB R2023a
- Psychtoolbox Version 3 (Psychtoolbox-3)
- SR Research EyeLink 1000

Note: The code is optimized for Linux and for a computer monitor with a 60 Hz refresh rate, 1280*1024 resolution, 170 LCD. Gaze was recorded at 1000 Hz. To run the experiment, adjust the file paths in run_experiment.m by replacing '/home/neuronoodle/Documents/MIRA_JULIAN/' with the directory where the experiment files are stored. For access to all image folders used in the experiment, please reach out to the authors.


**Main Scripts**
These scripts are essential for running the experiment.

**1. run_experiment.m**
This is the primary script that initializes and runs the experiment.
**Sections:**
- **General Setup** (line 1-158): Psychtoolbox configuration, set up of cell for data saving, subject number prompt, and set up of required parameters and paths (including practice trials, total trials, 2x2 design, image loading, folder allocation).
- **Eyetracker Initialization** (line 159-178): Set up EyeLink 1000 settings.
- **Start Instructions** (line 179-207): Displays instructions.
- **Practice Trials** (line 208-503): Displays stimuli in 4 blocks of 8 trials each, covering the four conditions. Records initial participant responses and fixation data (by calling the "response" function), records participant response confirmation (or, alternatively, re-calls the "response" function), and provides feedback on accuracy. Ensures that trials are repeated if the participant’s gaze deviates from the screen center in "fixed gaze" conditions.
- **End of Practice, Start Actual Trials** (504-975): Same setup as practice trials, but with 8 blocks of 36 trials each, in the predefined randomised order. At the beginning of each block, the eyetracker is started and calibrated. Throughout each block, data is incrementally saved in the "Data" table using the "cell2table" function. At the end of each block, the "Data" table is converted into a .csv file using the "writetable" function (line 896), which ensures regular "checkpoint" saves after each block, while ensuring that, at the end of the experiment, there is only one complete .csv file for the whole experiment. Eyetracking files are saved as .edf. At the very end of the script, the screen and eyetracker are closed.

**2. response.m**
Defines the "response" function, which is called immediately following stimulus presentation and handles how participant responses are collected. 
**Sections:**
- **General Setup** (Lines 1-10): Defines the function's inputs and outputs and sets up some general parameters.
- **Get Window Parameters** (Lines 11-24): Sets up window parameters.
- **Set Up Buttons** (Lines 25-39): Sets up and positions "buttons", which represent possible answers.
- **Get User Response** (Lines 40-117): If this is the first time the "response" function is called in a given trial, a while-loop checks for initial response (spacebar) to record reaction time while, in "fixed gaze" trials, checking fixation using the "check_fixation" function. Initial response is followed by short presentation of a mask image. Then, the buttons are drawn and presented, which is followed by a while loop that checks for the actual numeric response. This is followed by a confirmation screen. The confirmation itself, however, takes place outside of the "response" function, in the while-loop in "run_experiment.m".

**3. check_fixation.m**
Defines the "check_fixation" function, which is called in the "response" function to monitor fixation during "fixed gaze" trials . The function checks whether the gaze deviates from the center of the screen by more than 100 pixels.

**4. recalibrate_eyetracker.m**
Defines the "recalibrate_eyetracker" function, which is called when there have been seven consecutive fixation breaks (aborted trials) in a "fixed gaze" block. It initiates and handles recalibration of the eyetracker.

**5. saveData.m**
Defines the "saveData" function, which was used as a data saving method in an earlier version of the script but is now redundant, as "writetable" in "run_experiment.m" now saves the .csv file. Although "saveData" is still called in the main script, it does not impact the actual data-saving process. However, since it is called, the "saveData.m" file must be present in the directory for the experiment to run without errors.


**Additional/Non-Essential Scripts**
These scripts are not essential for the primary experiment as their relevant functionalities were incorporated into the main "run_experiment.m" script. However, they may offer additional resources for customization or troubleshooting.

**6. eyetracker_commands.m**
Lays out different eyetracker commands. 

**7. stop_eyetracker.m**
Lays out a command to stop the eyetracker and save eyetracking data. 
