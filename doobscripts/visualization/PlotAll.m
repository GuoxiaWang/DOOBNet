addpath('../export_fig');
addpath('../common');
addpath('edge_link');

DataSet = 'PIOD';
% DataSet = 'BSDSownership';
switch DataSet
    case 'PIOD'
        img_dir = '../../data/PIOD/JPEGImages/';
        gt_mat_dir = '../../data/PIOD/Data/';
        res_mat_dir = '../../results/PIOD/';
        export_dir = '../../results/figs/PIOD/';    
        test_ids_filename = '../../data/PIOD/val_doc_2010.txt';

        
    case 'BSDSownership'
        img_dir = '../../BSDSownership/BSDS300/images/train/';
        gt_mat_dir = '../../data/BSDSownership/testfg/';
        res_mat_dir = '../../results/BSDSownership/';
        export_dir = '../../results/figs/BSDSownership/';
        test_ids_filename = '../../data/BSDSownership/Augmentation/test_ori_iids.lst';
         
end

opt.nms_thresh = 0.5;
opt.append = ''; 
opt.w_occ = 1; 
if opt.w_occ; opt.append = '_occ'; end 
opt.nthresh = 33; 
opt.thinpb = 1; 
opt.print = 1;
opt.validate = 1; 

imgids = textread(test_ids_filename, '%s');

if opt.validate
    valid_num = 1; 
    imgids = imgids(1:valid_num); 
end 

nImgs = length(imgids);
for idx = 1:nImgs
    close all;
    
    I = imread([img_dir imgids{idx} '.jpg']);
    
    switch DataSet
        case 'PIOD'
            load([gt_mat_dir imgids{idx} '.mat'], 'bndinfo_pascal');
            gt_img = GetEdgeGT(bndinfo_pascal);
            gt_edge = gt_img(:,:,1);
            gt_ori = gt_img(:,:,2);

        case 'BSDSownership'
            load([gt_mat_dir imgids{idx} '.mat'], 'gtStruct');
            gt_img = cat(3, gtStruct.gt_theta{:});
            % BSDS onwership gt annotation from two anntator.
            % 3, 4 channel is the second annotation,
            % you can change to 1, 2 if you want to plot the first
            % annatation
            gt_edge = gt_img(:,:,3);
            gt_ori = gt_img(:,:,4);
    end
 
    % original image
    figure, imshow(I), truesize(gcf);
    if opt.print, export_fig(sprintf('%s%s_img', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;


    % GT 
    PlotOcclusionArraw(I, gt_edge, gt_ori);
    if opt.print, export_fig(sprintf('%s%s_occ_gt', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-painters', '-tight');  end;

    % GT orientation
    PlotOri(gt_ori);
    if opt.print, export_fig(sprintf('%s%s_ori_gt', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    % GT boundary
    figure, imshow(1-gt_edge), truesize(gcf);
    if opt.print, export_fig(sprintf('%s%s_edge_gt', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    % predicted
    load([res_mat_dir imgids{idx} '.mat'], 'edge_ori');
    res_edge = edge_ori.edge;
    res_ori = edge_ori.ori;
    
%     % predicted orientation before nms
%     PlotOri(res_ori);
%     if opt.print, export_fig(sprintf('%s%s_ori_res_1', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
%     
%     % predicted edge before nms
%     figure, imshow(res_edge), truesize(gcf);
%     if opt.print, export_fig(sprintf('%s%s_edge_res_1', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    res_edge = edge_nms(res_edge, 0);
    mask = find(res_edge<0.01);
    res_ori(mask) = 0;
    
    % predicted orientation
    PlotOri(res_ori);
    if opt.print, export_fig(sprintf('%s%s_ori_res', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    % predicted edge
    figure, imshow(1-res_edge), truesize(gcf);
    if opt.print, export_fig(sprintf('%s%s_edge_res', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    % predicted TP, FP, FN, with occlusion arraw
    res_img = zeros([size(edge_ori.edge), 2], 'single');
    res_img(:,:,1) = edge_ori.edge;
    res_img(:,:,2) = edge_ori.ori; 
    PlotOcclusion(I, res_img, gt_img, opt);
    if opt.print, export_fig(sprintf('%s%s_occ_res', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
end