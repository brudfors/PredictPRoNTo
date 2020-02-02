# PredictPRoNTo

Predict using the PRoNTo toolbox. Supports both regression and classification, and different predictive models.

A typical example of when these type of predictions could be interesting is when multiple subjects' spatially normalised tissue segmentations, obtained with for example the SPM12 software, are available. If the subjects have age and/or sex labels, then the approach implemented here can be used for prediciting between subjects.

For detailed information see PredictPRoNTo.m.

## Requirements 

Requires that SPM12 and PRoNTo v2 are on the MATLAB path:
* SPM12:  https://www.fil.ion.ucl.ac.uk/spm/software/download/
* PRoNTo: http://www.mlnl.cs.ucl.ac.uk/pronto/prtsoftware.html
