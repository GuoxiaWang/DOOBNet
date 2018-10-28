function EvaluateBoundary(imglist, varargin)
opt = struct('Evaluate', 'edge', 'resPath', 'gtPath', 'nthresh', 30, ...,
    'renormalize', 1, 'w_occ', 0, 'overwrite', 0, 'dataBase', 'PASCAL', 'scale_id', 0);

opt = CatVarargin(opt, varargin);
append = opt.append;

if exist(fullfile(opt.outDir,['eval_bdry_thr',append,'.txt']), 'file') & ~ opt.overwrite
    return;
end

outDir = opt.outDir;
resPath = opt.resPath;
gtPath = opt.gtPath;
opt.dilate = 0; % do not dilate the ground truth for evaluation 
scale_id = opt.scale_id; 
if scale_id ==0; scale_id = 1; end 

if opt.w_occ
    fname = fullfile(opt.outDir, ['eval', append, '_acc.txt']); 
else
    fname = fullfile(opt.outDir, ['eval_bdry',append,'.txt']); 
end

if ~exist(fname, 'file')  | opt.overwrite 
    % assert(length(resfile) == length(gtfile));
    n = length(imglist);
    do = false(1, n);
    params = cell(1, n);
    for ires = 1:length(imglist)
        prFile = fullfile(outDir, [imglist{ires}, '_ev1',append,'.txt']); 
        if exist(prFile, 'file')& ~opt.overwrite; continue; end 
        do(ires) = true;
        % get results

        resfile = fullfile(resPath, [imglist{ires}, '.mat']);
        edge_maps = load(resfile);
        edge_maps = edge_maps.edge_ori;
        if opt.w_occ
            res_img = zeros([size(edge_maps.edge), 2], 'single');
            res_img(:,:,1) = edge_maps.edge;
            res_img(:,:,2) = edge_maps.ori;
        else
            res_img = zeros([size(edge_maps.edge), 1], 'single');
            res_img(:,:,1) = edge_maps.edge;
        end
        
        if size(edge_maps,2) >= 3 && opt.w_occ
            opt.rank_score = res_img(:,:,1) + edge_maps{scale_id,3};
        end
        
        % get ground truth
        switch opt.DataSet
            case 'PIOD'
                gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                bndinfo_pascal = load(gtfile);
                bndinfo_pascal = bndinfo_pascal.bndinfo_pascal;         
                gt_img = GetEdgeGT(bndinfo_pascal);

              
            case 'BSDSownership'
                gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                gtStruct = load(gtfile);
                gtStruct = gtStruct.gtStruct;
                gt_img = cat(3, gtStruct.gt_theta{:});
        end
        
%         if opt.vis 
%             imagesc(gt_img(:,:,2)); 
%             pause; 
%         end
        
        if ~all(size(res_img(:,:,1)) - size(gt_img(:,:,1)) == 0)
            res_img = imresize(res_img, size(gt_img(:,:,1)), 'bilinear'); 
            opt.rank_score = imresize(opt.rank_score, size(gt_img(:,:,1)), 'bilinear'); 
        end
        params{ires} = {res_img, gt_img, prFile, opt};
        %EvaluateSingle(res_img, gt_img, prFile, opt);
    end

    params = params(do);
    n = length(params);
    parforProgress(n);
    parfor i=1:n,
        feval('EvaluateSingle', params{i}{:});
        parforProgress;
    end
    parforProgress(0);
end

%% collect results
if strfind(append, '_occ'); 
    % opt.overwrite = 1;
    collect_eval_bdry_occ(opt.outDir, append, opt.overwrite);
else
    collect_eval_bdry_v2(opt.outDir, append);
end

%% clean up
% system(sprintf('rm -f %s/*_ev1%s.txt',opt.outDir,append));

end



