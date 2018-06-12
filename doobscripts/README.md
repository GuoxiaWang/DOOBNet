This is the matlab script code for the PIOD and the BSDS ownership dataset and tests successfuly on Matlab2015b. You can use it to evaluate and visulize the occlusion boundary results.

# Evaluation

There are two main interfaces:
*  `EvaluateOcc.m`
   This script perform the evaluation.
   In this script, you need to set following variables:
   ```
   oriGtPath          it specifies the directory that contains mat files of test images, e.g data/PIOD/Data
   oriResPath         it specifies the root directory of prediction results
   testIdsFilename    the relative or absolute path of test ids file, e.g test_ori_iids.lst or val_doc_2010.txt  
   opt.method_folder  the folder name in oriResPath 
   opt.model_name     the folder name in opt.method_folder that contains mat files of your methods prediction
   ...
   ...                you can see the other variables in EvaluateOcc.m
   ```

*  `OccCompCurvesPlot.m`
   This script plots multi algorithms boundary PR curves, object occlusion boundary PR curves, occlusion boundary accuracy recall curves.
   In this script, you need to set following vairables:
   ```
   algs        N x 1 cell, each element is pridiction path
   nms         N x 1 cell, each element is algorithm name which will be present at legend
   export_dir  the directory that you want to save the plots
   ```

# Visulation

The visualization script supports 7 type figures: original image, ground truth boundary map, ground truth occlution orientation map, ground truth occlution boundary map with arrow, and corresponding prediction map.

`PlotAll.m` is the main point. You need to set following variables:
```
img_dir            the original images directory, e.g data/PIOD/JPEGImages
gt_mat_dir         the ground truth annotation directory, e.g data/PIOD/Data
res_mat_dir        the prediction directory, which contans .mat files
export_dir         the directory that you want to save the print eps figure
test_ids_filename  the relative or absolute path of test ids file, e.g test_ori_iids.lst or val_doc_2010.txt

```

# Issues

*  If you meet the error about edgesNmsMex or correspondPixels, you can download [edge box](https://github.com/pdollar/edges) and add the path to matlab path or copy edgesNmsMex.mexa64 and correspondPixels.mexa64 from `edges/private` folder to `common` folder. If you can not run, please recompile the edgesNmsMex and correspondPixels according the README.
*  If you meet the error about convConst or gradientMex, you can download [piotr's toolbox](https://github.com/pdollar/toolbox) and add the path to matlab path or copy convConst.mexa64 and gradientMex.mexa64 from `piotr_toolbox/channels/private` folder to `common` folder. If you can not run, please get more information from author's website.
