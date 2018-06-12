% This script is the entry point to perform the evaluation.
addpath('../common');
clear;
tic;
DataSet = 'PIOD';
% DataSet = 'BSDSownership';
switch DataSet
    case 'PIOD'
        oriGtPath = '../../data/PIOD/Data/';
        oriResPath = '../../results/';

        opt.method_folder = 'DOOBNet';
        opt.model_name = 'PIOD';
        
        testIdsFilename = '../../data/PIOD/val_doc_2010.txt';
        ImageList = textread(testIdsFilename, '%s'); omit_id = [];
        
    case 'BSDSownership'
        oriGtPath = '../../data/BSDSownership/testfg/';
        oriResPath = '../../results/';
        
        opt.method_folder = 'DOOBNet';
        opt.model_name = 'BSDSownership';  
        
        testIdsFilename = '../../data/BSDSownership/Augmentation/test_ori_iids.lst';
        ImageList = textread(testIdsFilename, '%s'); omit_id = []; 
end


respath = [oriResPath, opt.method_folder, '/', opt.model_name , '/'];
evalPath = [oriResPath, opt.method_folder, '/eval_fig/']; if ~exist(evalPath, 'dir') mkdir(evalPath); end

ImageList(omit_id) = []; 

opt.DataSet = DataSet; 

opt.vis = 1; 
opt.print = 0; 
opt.overwrite = 1; 
opt.visall = 0; 
opt.append = ''; 
opt.validate = 0; 
opt.occ_scale = 1;  % set which scale output for occlusion
opt.w_occ = 1; 
if opt.w_occ; opt.append = '_occ'; end 
opt.scale_id = 0; 
if opt.scale_id ~= 0;
    opt.append = [opt.append, '_', num2str(opt.scale_id)]; 
end

opt.outDir = respath;
opt.resPath = respath;
opt.gtPath = oriGtPath; 
opt.nthresh = 99;   % threshold to calculate precision and recall
                    % it set to 33 in DOC for save runtime but 99 in DOOBNet.
opt.thinpb = 1;     % thinpb means performing nms operation before evaluation.
opt.renormalize = 0; 
opt.fastmode = 0;   % see EvaluateSingle.m

if (~isfield(opt, 'method') || isempty(opt.method)), opt.method = opt.method_folder; end
fprintf('Starting evaluate %s %s, model: %s and %s\n', DataSet, opt.method, opt.model_name, opt.append); 

if opt.validate
    valid_num = 10; 
    ImageList = ImageList(1:valid_num); 
end 

EvaluateBoundary(ImageList, opt);

if opt.vis
    if strfind(opt.append, '_occ'); 
        close all;
        app_name = opt.append; 
        opt.eval_item_name = 'Boundary';
        opt.append = [app_name, '_e'];  plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge'); 
        % if opt.print,        set(gcf, 'PaperPositionMode', 'auto');        print(['-f' num2str(1)], '-dpng', [evalPath, model_name '.png']);    end
        
        opt.eval_item_name = 'Orientation PR';
        opt.append = [app_name, '_poc'];  plot_multi_eval_v2(opt.outDir, opt, opt.method);  title('PRO'); 
        
        opt.append = [app_name, '_aoc'];
        plot_multi_occ_acc_eval(opt.outDir, opt, opt.method);
    end
end
toc;


