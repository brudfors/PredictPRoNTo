# PredictPRoNTo

Predict using the PRoNTo toolbox. Supports both regression and classification, and different predictive models.

A typical example of when these type of predictions could be interesting is when multiple subjects' spatially normalised tissue segmentations from brain MRI data, obtained with for example the SPM12 software, are available. If the subjects have age and/or sex labels, then the approach implemented here can be used for prediciting between subjects.

For detailed information see PredictPRoNTo.m.

## Example

Call `PredictPRoNTo(data)`, where `data` is a cell array with the following form:

data{n,1} = 'subj1feature.nii' or {'subj1feature1.nii','subj1feature2.nii'}  
data{n,2} = floating point number (e.g., 42)  
data{n,3} = logical value (e.g., true)

and `n` is the subject index going from `1` to `N`, the total number of subjects.

## Requirements 

Requires that SPM12 and PRoNTo v2 are on the MATLAB path:
* SPM12:  https://www.fil.ion.ucl.ac.uk/spm/software/download/
* PRoNTo: http://www.mlnl.cs.ucl.ac.uk/pronto/prtsoftware.html (.zip file in this repo!)
