function PredictPRoNTo(Data,s)
% Predict using the PRoNTo toolbox. Supports both regression and 
% classification, and different predictive models.
%
% FORMAT PredictPRoNTo(Data,s)
%
% INPUT
%
% 'Data' is an Nx2 or Nx3 cell array, where N are the number of subjects.
% Each row of 'data' should contain, in the first column, the path(s) to
% nifti file(s) holding the data that should be used as features in the
% prediction. If each subject has one nifti, then each element of the first
% column should be a string. If there are multiple niftis per subject, then
% each element of the first column should contain a cell array of string.
% The second and third columns of 'data' should hold regression and
% classification targets, as floats and logicals, respectively. One of the
% two can also be given, not both. The code will figure out they are
% regression or classification targets, based on the data type.
%
% 's' is a structure with settings, if not given or empty, uses default
% (described in detail below).
%
% EXAMPLE
%
% -Call PredictPRoNTo(data), where data has the following form:
%     data{n,1} = 'subj1feature.nii' or {'subj1feature1.nii','subj1feature2.nii'}
%     data{n,2} = floating point number (e.g., 42)
%     data{n,3} = logical value (e.g., true)
%
%__________________________________________________________________________
% A typical example of when these type of predictions could be interesting
% is when multiple subjects' spatially normalised tissue segmentations,
% obtained with for example the SPM12 software, are available. If the
% subjects have age and/or sex labels, then the approach implemented here
% can be used for prediciting between subjects.
%
% Requires that SPM12 and PRoNTo v2 are on the MATLAB path:
% SPM12:  https://www.fil.ion.ucl.ac.uk/spm/software/download/
% PRoNTo: http://www.mlnl.cs.ucl.ac.uk/pronto/prtsoftware.html
%__________________________________________________________________________
% Mikael Brudfors (brudfors@gmail.com)
% Copyright (C) 2020 Wellcome Centre for Human Neuroimaging, UCL, London

%--------------------------------------------------------------------------
% Settings
%--------------------------------------------------------------------------

if nargin < 2, s = struct; end
% === Make PRoNTo features (if false, load already generated) =============
if ~isfield(s,'DoProcess'),        s.DoProcess       = true; end
% === Show one feature example ============================================
if ~isfield(s,'DoVisualise'),      s.DoVisualise      = 1; end
% === Show prediction results =============================================
% 0 - Show nothing
% 1 - Print to command window
% 2 - 1 + plot figures (regression plot and ROC curve)
if ~isfield(s,'ShowResults'),      s.ShowResults      = 2; end
% === Folder where to store results =======================================
if ~isfield(s,'DirRes'),           s.DirRes           = './results'; end
% === Number of subjects to include (Inf -> all) ==========================
if ~isfield(s,'N'),                s.N                = Inf; end
% === Include background in feature data ==================================
if ~isfield(s,'IncBg'),            s.IncBg            = true; end
% === FWHM of smoothing of PRoNTo features ================================
if ~isfield(s,'FWHM'),             s.FWHM             = 12; end
% === Apply brain mask to PRoNTo features =================================
if ~isfield(s,'Msk'),              s.Msk              = true; end
% === Cross-validaton setting =============================================
% -1: leave one out, >=1: k-fold
if ~isfield(s,'CrsVal'),           s.CrsVal           = 10; end    
% === Pattern recognition algorithm to employ =============================
% krr - kernel ridge regression (equivalent to a MAP approach to GP regression with fixed prior variance and no explicit noise term)
% gp  - gaussian process regression
% mkl - L1-Multiple Kernel Learning (SWM)
% rvr - relevance vector regression (has an identical functional form to the SVM, but provides probabilistic output)
if ~isfield(s,'Machine'),          s.Machine          = 'gp'; end

%--------------------------------------------------------------------------
% Add PRoNTo to MATLAB path and check that SPM is available
%--------------------------------------------------------------------------

if isempty(fileparts(which('spm'))),    error('SPM12 not on the MATLAB path!'); end
if isempty(fileparts(which('pronto'))), error('PRoNTo not on the MATLAB path!'); end

%--------------------------------------------------------------------------
% Get features for prediction (e.g., normalised, smoothed segmentations)
%--------------------------------------------------------------------------

% Generate PRoNTo features (e.g., from normalised SPM12 segmentations), takes time!
Nii = MakeFeatures(Data, s.DoProcess, s.DirRes, s.N, s.IncBg, s.FWHM);    

% Inspect features
if s.DoVisualise, spm_check_registration(char(Nii{s.DoVisualise})); end

%--------------------------------------------------------------------------
% Load targets (e.g., sex or age)
%--------------------------------------------------------------------------

[TargCls,TargReg] = GetTargets(Data,s.N);

%--------------------------------------------------------------------------
% Predict
%--------------------------------------------------------------------------

if ~isempty(TargReg)
    % Run regression 
    ResReg = Predict(Nii, TargReg, s.DirRes, s.Msk, s.CrsVal, s.Machine);
end

if ~isempty(TargCls)
    % Run classification
    ResClass = Predict(Nii, TargCls, s.DirRes, s.Msk, s.CrsVal, s.Machine);
end

%--------------------------------------------------------------------------
% Display prediction results
%--------------------------------------------------------------------------

if s.ShowResults >= 1
    % Print to command window
    if ~isempty(TargReg)
        disp('--------------------------------------------------------------------------')
        disp('Regression Results')
        disp(ResReg{2})
    end

    if ~isempty(TargCls)
        disp('--------------------------------------------------------------------------')
        disp('Classification Results')
        disp(ResClass{2})
    end
end

if s.ShowResults >= 2
    % Plot to figure    
    if ~isempty(TargReg)
        % Regression plot
        PlotRegRes(ResReg);
        print(gcf,fullfile(s.DirRes,'ResReg.png'),'-dpng','-r300');  
    end

    if ~isempty(TargCls)
        % ROC curve
        prt_plot_ROC(ResClass{1}.PRT, 1, 1);
        print(gcf,fullfile(s.DirRes,'ResClass.png'),'-dpng','-r300');  
    end
end

%--------------------------------------------------------------------------
% Save results
%--------------------------------------------------------------------------

if exist('ResClass','var'), save(fullfile(s.DirRes,'Classification.mat'),'ResClass'); end
if exist('ResReg','var'),   save(fullfile(s.DirRes,'Regression.mat'),'ResReg'); end
end
%==========================================================================

%==========================================================================
function [TargCls,TargReg] = GetTargets(Data,N)
% Get predicion targets

N       = min(size(Data,1),N);
TargCls = zeros(N,1);
TargReg = zeros(N,1);
for n=1:N
    if islogical(Data{n,2}), TargCls(n) = Data{n,2}; end
    if isfloat(Data{n,2}), TargReg(n) = Data{n,2}; end
    if size(Data,2) > 2 && islogical(Data{n,3}), TargCls(n) = Data{n,3}; end
    if size(Data,2) > 2 && isfloat(Data{n,3}),   TargReg(n) = Data{n,3}; end
end
TargCls = {find(TargCls == 0), find(TargCls == 1)};
end
%==========================================================================

%==========================================================================
function Nii = MakeFeatures(Data,DoProcess,DirRes0,N,IncBg,FWHM)
% Make PRoNTo input features (stored as nifti) from nifti files of, e.g.,
% spatially normalised tissue segmentations

if nargin < 4, N     = Inf;  end
if nargin < 5, IncBg = true; end
if nargin < 6, FWHM  = 12;   end

fprintf('Generating features for PRoNTo...\n');

%--------------------------------------------------------------------------
% Get paths to NIfTI segmentations, and targets
%--------------------------------------------------------------------------

% Number of subjects and classesN = min(size(Data,1),N); (excl. background)
N = min(size(Data,1),N);
if iscell(Data{1})
    NumClasses = numel(Data{1});   
else
    NumClasses = 1;
end

% Folder for storing NIfTIs
DirRes = fullfile(DirRes0,'features');
if exist(DirRes,'dir') == 7, rmdir(DirRes,'s'); end; mkdir(DirRes);

% Loop over subjects
Nii = cell(1,N);
for n=1:N
     
    if DoProcess || NumClasses > 1
        fprintf('%i/%i ',n,N)

        % Read class into an array, used as features for PRONTO    
        Image      = [];
        Background = 1;
        for k=1:NumClasses
            PthClass = deblank(Data{n,1}{k});
            Niis     = nifti(PthClass);               
            Image_k  = single(Niis.dat(:,:,:));

            Image = cat(3,Image,Image_k);

            if IncBg, Background = Background - Image_k; end
        end
        clear Image_k

        if IncBg
            % Add background class                               
            Image = cat(3,Image,Background);
        end    

        if FWHM > 0
            % Smooth image
            VoxelSize = sqrt(sum(Niis.mat(1:3,1:3).^2));   
            Image     = SmoothImage(Image,FWHM,VoxelSize);
        end

        PthClass = WriteNIfTI(PthClass,DirRes,Image);    
    else
        PthClass = Data{n,1};
    end
    
    Nii{n} = PthClass;
    
end

save(fullfile(DirRes0,'Nii.mat'),'Nii')

fprintf('\nDone!\n');
end
%==========================================================================

%==========================================================================
function Res = Predict(Nii,Targets,DirRes,Msk,CrsVal,Machine)
% Predict, either regression or classification, using PRoNTo

if nargin < 4, Msk     = true; end
if nargin < 5, CrsVal  = 10;   end
if nargin < 6, Machine = 'gp'; end

% Create results directory
DirRes = fullfile(DirRes,'PRT');
if exist(DirRes,'dir') == 7, rmdir(DirRes,'s'); end; mkdir(DirRes);

% Path to PRT struct
pth_PRT = fullfile(DirRes,'PRT.mat');

% Classification or regression?
if iscell(Targets), Model = 'classify';
else,               Model = 'regress';
end

% Get mask
if Msk    
    % Remove outside of brain voxels
    PthMask = fullfile(fileparts(which('pronto')),'masks/SPM_mask_noeyes.img');
else
    % Keep all voxels
    NiiImage       = nifti(Nii{1});
    ImageDim       = NiiImage.dat.dim;
    ImageMat       = NiiImage.mat;
    PthMask        = fullfile(DirRes,'mask.nii');  
    NiiOut         = nifti;
    NiiOut.dat     = file_array(PthMask,ImageDim,[spm_type('float32') spm_platform('bigend')]);
    NiiOut.mat     = ImageMat;
    NiiOut.mat0    = ImageMat;
    NiiOut.descrip = 'PRONTO mask';
    create(NiiOut);
    NiiOut.dat(:,:,:,:) = ones(ImageDim);
end

% Set number of subjects
N = numel(Nii);

%--------------------------------------------------------------------------
% Step 1. Data
%--------------------------------------------------------------------------

matlabbatch{1}.prt.data.dir_name      = {DirRes};
matlabbatch{1}.prt.data.group.gr_name = 'group';

% Subjects
matlabbatch{1}.prt.data.group.select.modality.mod_name = 'modality';
matlabbatch{1}.prt.data.group.select.modality.subjects = Nii;
matlabbatch{1}.prt.data.group.select.modality.rt_subj  = [];
if strcmpi(Model,'regress')
    matlabbatch{1}.prt.data.group.select.modality.rt_subj  = Targets;
end
matlabbatch{1}.prt.data.group.select.modality.covar    = {''};

% Mask
matlabbatch{1}.prt.data.mask.mod_name = 'modality';
matlabbatch{1}.prt.data.mask.fmask    = {PthMask};
matlabbatch{1}.prt.data.mask.hrfover  = 0;
matlabbatch{1}.prt.data.mask.hrfdel   = 0;

matlabbatch{1}.prt.data.review = 0;

prt_run_design(matlabbatch{1}.prt.data);

%--------------------------------------------------------------------------
% Step 2. Features
%--------------------------------------------------------------------------

matlabbatch{2}.prt.fs.infile                        = {pth_PRT};
matlabbatch{2}.prt.fs.k_file                        = 'kernel';
matlabbatch{2}.prt.fs.modality.mod_name             = 'modality';
matlabbatch{2}.prt.fs.modality.conditions.all_scans = 1;
matlabbatch{2}.prt.fs.modality.voxels.all_voxels    = 1;
matlabbatch{2}.prt.fs.modality.detrend.no_dt        = 1;
matlabbatch{2}.prt.fs.modality.normalise.no_gms     = 1;
matlabbatch{2}.prt.fs.modality.atlasroi             = {''};
matlabbatch{2}.prt.fs.flag_mm                       = 0;

prt_run_fs(matlabbatch{2}.prt.fs);

%--------------------------------------------------------------------------
% Step 3. Model
%--------------------------------------------------------------------------

matlabbatch{3}.prt.model.infile     = {pth_PRT};
matlabbatch{3}.prt.model.model_name = Model;
matlabbatch{3}.prt.model.use_kernel = 1;
matlabbatch{3}.prt.model.fsets      = 'kernel';

if strcmpi(Model,'regress')
    % Regression
    matlabbatch{3}.prt.model.model_type.regression.reg_group.gr_name   = 'group';
    matlabbatch{3}.prt.model.model_type.regression.reg_group.subj_nums = 1:N;

    if strcmpi(Machine,'krr')
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.krr.krr_opt                = 0;
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.krr.krr_args               = 1;
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.krr.cv_type_nested.cv_loso = 1;
    elseif strcmpi(Machine,'gp')
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.gpr.gpr_args = '-l gauss -h';
    elseif strcmpi(Machine,'mkl')
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.sMKL_reg.sMKL_reg_opt           = 0;
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.sMKL_reg.sMKL_reg_args          = 1;
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.sMKL_reg.cv_type_nested.cv_loso = 1;
    elseif strcmpi(Machine,'rvr')
        matlabbatch{3}.prt.model.model_type.regression.machine_rg.rvr = struct([]);
    else
        error('Undefined machine!')
    end
elseif strcmpi(Model,'classify')
    % Classification
    matlabbatch{3}.prt.model.model_type.classification.class(1).class_name                 = 'M';
    matlabbatch{3}.prt.model.model_type.classification.class(1).group.gr_name              = 'group';
    matlabbatch{3}.prt.model.model_type.classification.class(1).group.subj_nums            = Targets{1};
    matlabbatch{3}.prt.model.model_type.classification.class(1).group.conditions.all_scans = 1;

    matlabbatch{3}.prt.model.model_type.classification.class(2).class_name                 = 'F';
    matlabbatch{3}.prt.model.model_type.classification.class(2).group.gr_name              = 'group';
    matlabbatch{3}.prt.model.model_type.classification.class(2).group.subj_nums            = Targets{2};
    matlabbatch{3}.prt.model.model_type.classification.class(2).group.conditions.all_scans = 1;
    
    if strcmpi(Machine,'gp')
        matlabbatch{3}.prt.model.model_type.classification.machine_cl.gpc.gpc_args = '-l erf -h';
    else
        error('Undefined machine!')
    end    
end
    
if CrsVal == -1
    % Leave one out
    matlabbatch{3}.prt.model.cv_type.cv_loso         = 1;
else
    % K-fold
    matlabbatch{3}.prt.model.cv_type.cv_lkso.k_args  = CrsVal;
end
matlabbatch{3}.prt.model.include_allscans            = 0;
matlabbatch{3}.prt.model.sel_ops.data_op_mc          = 1;
matlabbatch{3}.prt.model.sel_ops.use_other_ops.no_op = 1;

prt_run_model(matlabbatch{3}.prt.model);

%--------------------------------------------------------------------------
% Step 4. CV
%--------------------------------------------------------------------------

matlabbatch{4}.prt.cv_model.infile            = {pth_PRT};
matlabbatch{4}.prt.cv_model.model_name        = Model;
matlabbatch{4}.prt.cv_model.perm_test.no_perm = 1;

prt_run_cv_model(matlabbatch{4}.prt.cv_model);

%--------------------------------------------------------------------------
% Return results
%--------------------------------------------------------------------------

Res{1} = load(pth_PRT);

if strcmpi(Model,'classify')
    Res{2} = sprintf('acc=%0.3f, acc_lb=%0.3f, acc_ub=%0.3f', ...
                         Res{1}.PRT.model.output.stats.acc, ...
                         Res{1}.PRT.model.output.stats.acc_lb, ...
                         Res{1}.PRT.model.output.stats.acc_ub);
else
    Res{2} = sprintf('r2=%0.3f, rmse=%0.3f', ...
                         Res{1}.PRT.model.output.stats.r2, ...
                         sqrt(Res{1}.PRT.model.output.stats.mse));   
end

%--------------------------------------------------------------------------
% % Delete Feature_set_modality (because it is HUGE)
%--------------------------------------------------------------------------

delete(fullfile(DirRes,'Feature_set_modality.dat'))
end
%==========================================================================

%==========================================================================
% Utility functions
%==========================================================================

%==========================================================================
function PthTissue = WriteNIfTI(PthTissue,DirWrite,Image,IsBackground)
% Write image data to disk, as nifti files (in single precision)

if nargin < 4, IsBackground = false; end

Nii           = nifti(PthTissue);
[~,nam,ext]   = fileparts(PthTissue);
if IsBackground
    PthTissue = fullfile(DirWrite,['bg' nam ext]);  
else
    PthTissue = fullfile(DirWrite,[nam ext]);  
end

NiiOut         = nifti;
NiiOut.dat     = file_array(PthTissue,size(Image),[spm_type('float32') spm_platform('bigend')]);
NiiOut.mat     = Nii.mat;
NiiOut.mat0    = Nii.mat;
NiiOut.descrip = 'PRONTO input';
create(NiiOut);
NiiOut.dat(:,:,:,:) = Image;
end
%==========================================================================

%==========================================================================
function simg = SmoothImage(img,fwhm,VoxelSize)
% Smooth an image (in memory), taking voxel size into account

if nargin<2, fwhm      = 12; end
if nargin<3, VoxelSize = 1; end

if numel(fwhm) == 1
    fwhm = fwhm*ones(1,3);
end
if numel(VoxelSize) == 1
    VoxelSize = VoxelSize*ones(1,3);
end

simg = zeros(size(img),'single');

fwhm = fwhm./VoxelSize;
s1   = fwhm/sqrt(8*log(2));

x  = round(6*s1(1)); x = -x:x; x = spm_smoothkern(fwhm(1),x,1); x  = x/sum(x);
y  = round(6*s1(2)); y = -y:y; y = spm_smoothkern(fwhm(2),y,1); y  = y/sum(y);
z  = round(6*s1(3)); z = -z:z; z = spm_smoothkern(fwhm(3),z,1); z  = z/sum(z);

i  = (length(x) - 1)/2;
j  = (length(y) - 1)/2;
k  = (length(z) - 1)/2;

spm_conv_vol(img,simg,x,y,z,-[i,j,k]);
end
%==========================================================================

%==========================================================================
function Targets = LoadTargets(Pth,N)
% Load prediction targets
v       = load(Pth);
fn      = fieldnames(v);
Targets = v.(fn{1});
Targets = Targets(:);
Targets = Targets(1:min(numel(Targets),N));
end
%==========================================================================

%==========================================================================
function PlotRegRes(Res)
figure;
T = []; % true
P = []; % predicted
for k=1:numel(Res{1}.PRT.model.output.fold)
    t = Res{1}.PRT.model.output.fold(k).targets;
    p = Res{1}.PRT.model.output.fold(k).predictions;
    T = [T; t];
    P = [P; p];
end

plot(T,P,'k.','MarkerSize',10);
axis image 
grid on

title('Scatter plot')
title(sprintf('Regression Plot / R2 = %0.2f',Res{1}.PRT.model.output.stats.r2))
ylabel('Estimated','FontWeight','bold')
xlabel('True','FontWeight','bold')
end
%==========================================================================