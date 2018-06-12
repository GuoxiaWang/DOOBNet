function PlotOcclusionArraw(I, edge,  occ, varargin)
opt = struct('draw_arrows', 1, 'nms_thresh', 0.5, 'dist_ratio', 10);
opt = CatVarargin(opt, varargin);

color = [1 0 0];
bgcolor = [1 1 1];
arrowcolor = [1 0 0];

imsize = size(edge);
arrowdist = ceil(sqrt(imsize(1).^2 + imsize(2).^2)/opt.dist_ratio);

edge = edge_nms(edge, opt.nms_thresh);

figure, imshow(I);
hold on
[frags, ~] = edgelink(edge, 5);
for ifrag = 1:length(frags)
    ind = frags{ifrag};
    
    if (length(ind) < 10)
        continue
    end
    ex = ind(:,2);
    ey = ind(:,1);
    npix = length(ind);
    
    plot(ex, ey, 'Color', bgcolor, 'LineWidth', 3);
    plot(ex, ey, 'Color', color, 'LineWidth', 2);
    
    narrows = ceil(npix/arrowdist);
    epos = ceil((1:narrows) / (narrows+1) * npix);
    
    for j = 1:numel(epos)
        ax = ind(epos(j), 2);
        ay = ind(epos(j), 1);
        
        y1 = ind(max(epos(j)-10,1), 1);
        x1 = ind(max(epos(j)-10,1), 2);
        y2 = ind(min(epos(j),npix), 1);
        x2 = ind(min(epos(j),npix), 2);

        theta1 = atan2(y2-y1, x2-x1);
        theta2 = atan2(y1-y2, x1-x2);
        theta_pred = occ(ay, ax);
        
        abs_diff = abs(theta_pred-theta1);
        abs_diff = mod(abs_diff, 2*pi);
        if(abs_diff <= pi/2 | abs_diff > 3*pi/2)
            theta = theta1;
        else
            theta = theta2;
        end
        
        asx = ax - cos(theta);
        asy = ay - sin(theta);
        
        ax = (ax-1)/(imsize(2)-1);   ay = 1-(ay-1)/(imsize(1)-1);
        asx = (asx-1)/(imsize(2)-1);   asy = 1-(asy-1)/(imsize(1)-1);
        
        if opt.draw_arrows
            annotation('arrow', [asx ax], [asy ay], 'LineStyle', 'none', ...
               'HeadWidth', 18, 'HeadLength', 11, 'Color', arrowcolor, ...
               'HeadStyle', 'vback2');
        end
    end
    
end

truesize(gcf);
hold off;
