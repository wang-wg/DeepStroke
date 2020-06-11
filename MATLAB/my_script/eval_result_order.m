%% This file is use to evaluate the performance of model on the three test font and handwritten data. 
%  该文件用于评估三个测试字体的笔画笔顺分支精度

clc;
% 1. stroke segmentation result. 
% 2. stroke order labeling result.
% 3. stroke order labeling result fixed by stroke segmentation result.
% When bishun_result=1, only evaluate 1.
% When bishun_result=2, evaluate 1 and 2.
% When bishun_result=3, evaluate 1, 2 and 3.

% bishun_result=1时只评估三个字体的 笔画分割结果
% bishun_result=2时只评估三个字体的 笔画分割结果  笔顺标注结果 
% bishun_result=3时评估三个字体的   笔画分割结果  笔顺标注结果  校正的笔顺标注结果
bishun_result=2;
% Number of each font. If you want to test on handwritten data, set test_num=70.
% 每个字体的字符数量，test_num=70则仅仅测试 手写数据集
test_num=6763;
DL_dataset_dir='/home/wwg/data/CCSSD/';

% The three test font name.
% 评测的三个字体
dataset_tpye_arrary= {'HLJ','SS','FZLBJW'};
% Name of all picture.
% 评测字体的所有图片名
valtxt_dir=strcat(DL_dataset_dir,'trainval.txt');
% final_result_save is used to store all result.
% 用于存放三个字体的 笔画分割结果  笔顺标注结果  校正的笔顺标注结果
final_result_save(1:3,1:3)=0;

if test_num==70
    dataset_tpye_arrary= {'hand'};
    valtxt_dir=strcat(DL_dataset_dir,'DATA_GB6763_hand/hand2017/trainval.txt');
end

% The performance in detail for each category will be written in this file.
% 每个类别的结果写在newest_result.txt中
txt_write_in=fopen('newest_result.txt','w');
for dataset_type_num=1:length(dataset_tpye_arrary)
    %% stroke segmentation branch 笔画分支

    num=35;
    dataset_type=dataset_tpye_arrary{dataset_type_num};  %FZLBJW
    datasets=strcat('DATA_GB6763_',dataset_type,'/');
    datasets_year=strcat(dataset_type,'2017/');
    maxlabel=34;
    v3_seg_dir=strcat(DL_dataset_dir,datasets,datasets_year,'exp/train_on_trainval_set/vis/raw_segmentation_results/');
    seg_dir=strcat(DL_dataset_dir,datasets,datasets_year,'/SegmentationClassAug/');    
    fid=fopen(valtxt_dir);

    confcounts = zeros(num);
    count=0;
    VOCopts = GetVOCopts('', '', '', '', 'Stroke');

    for i=1:test_num
        if mod(i,2000)==0
            fprintf('%d/%d\n',i,test_num);
        end   
        tline=fgetl(fid);
        v3_seg=strcat(v3_seg_dir,tline,'.png');
        seg=strcat(seg_dir,tline,'.png');
        gtim = imread(seg);    
        gtim = double(gtim);
        [resim] = imread(v3_seg);
        resim = double(resim);
        szgtim = size(gtim); szresim = size(resim);
        if any(szgtim~=szresim)
            error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
        end
        locs = gtim<255;
        sumim = 1+gtim+resim*num; 
        hs = histc(sumim(locs),1:num*num); 
        count = count + numel(find(locs));
        confcounts(:) = confcounts(:) + hs(:);
    end

    fclose(fid);
    conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
    rawcounts = confcounts;

    % Pixel Accuracy
    overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
    fprintf(txt_write_in,'Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

    % Class Accuracy
    class_acc = zeros(1, num);
    class_count = 0;
    fprintf(txt_write_in,'Accuracy for each class (pixel accuracy)\n');
    for i = 1 : num
        denom = sum(confcounts(i, :));
        if (denom == 0)
            denom = 1;
        end
        class_acc(i) = 100 * confcounts(i, i) / denom; 
        if i == 1
            clname = 'background';
        else
            clname = VOCopts.classes{i-1};
        end

        if ~strcmp(clname, 'void')
            class_count = class_count + 1;
            fprintf(txt_write_in,'%2d  %20s: %6.3f%%\n',i-1, clname, class_acc(i));
        end
    end
    fprintf(txt_write_in,'-------------------------\n');
    avg_class_acc = sum(class_acc) / class_count;
    fprintf(txt_write_in,'Mean Class Accuracy: %6.3f%%\n', avg_class_acc);

    % Pixel IOU
    accuracies = zeros(VOCopts.nclasses,1);
    fprintf(txt_write_in,'Accuracy for each class (intersection/union measure)\n');

    real_class_count = 0;

    for j=1:num

        gtj=sum(confcounts(j,:));
        resj=sum(confcounts(:,j));
        gtjresj=confcounts(j,j);
        % The accuracy is: true positive / (true positive + false positive + false negative) 
        % which is equivalent to the following percentage:
        denom = (gtj+resj-gtjresj);

        if denom == 0
            denom = 1;
        end

        accuracies(j)=100*gtjresj/denom;

        clname = 'background';
        if (j>1), clname = VOCopts.classes{j-1};end;

        if ~strcmp(clname, 'void')
            real_class_count = real_class_count + 1;
        else
            if denom ~= 1
                fprintf(1, 'WARNING: this void class has denom = %d\n', denom);
            end
        end

        if ~strcmp(clname, 'void')
            fprintf(txt_write_in,'%2d  %20s: %6.3f%%\n',j-1,clname,accuracies(j));
        end
    end

    % accuracies = accuracies(1:end);
    % avacc = mean(accuracies);
    avacc = sum(accuracies) / real_class_count;

    fprintf(txt_write_in,'-------------------------\n');
    fprintf(txt_write_in,'Average accuracy: %6.3f%%\n',avacc);
    fixIouTmp=sum(rawcounts');
    fixIouTmp1=fixIouTmp/sum(fixIouTmp);
    fixIouTmp2=fixIouTmp(2:maxlabel+1)/sum(fixIouTmp(2:maxlabel+1));
    fixIou=fixIouTmp1*accuracies;
    fixIouFont=fixIouTmp2*accuracies(2:maxlabel+1);
    fprintf(txt_write_in,'fixIou: %6.3f%%\n',fixIou);
    fprintf(txt_write_in,'fixIou of font ground: %6.3f%%\n',fixIouFont);

    final_result_save(1,dataset_type_num)=fixIouFont;
    
    %% stroke order labeling branch 笔顺分支
    for bishun=2:bishun_result
        num=32;
        datasets=strcat('DATA_GB6763_',dataset_type,'/');
        datasets_year=strcat(dataset_type,'2017/');
        maxlabel=31;
        if bishun==2
            v3_seg_dir=strcat(DL_dataset_dir,datasets,datasets_year,...
                'exp/train_on_trainval_set/vis/raw_segmentation_results_order/');
        elseif bishun==3
            v3_seg_dir=strcat(DL_dataset_dir,datasets,datasets_year,...
                'exp/train_on_trainval_set/vis/raw_segmentation_results_order_fix/');
        end

        seg_dir=strcat(DL_dataset_dir,datasets,datasets_year,'/OrderSegmentationClassAug/');

        fid=fopen(valtxt_dir);

        confcounts = zeros(num);
        count=0;
        VOCopts = GetVOCopts('', '', '', '', 'StrokeOrder');

        for i=1:test_num

            if mod(i,2000)==0
                fprintf('%d/%d\n',i,test_num);
            end   

            tline=fgetl(fid);
            v3_seg=strcat(v3_seg_dir,tline,'.png');

            seg=strcat(seg_dir,tline,'.png');

            [gtim] = imread(seg); 

            gtim = double(gtim);
            [resim] = imread(v3_seg);
            resim = double(resim);
        %    maxlabel = max(resim(:));

            szgtim = size(gtim); szresim = size(resim);
            if any(szgtim~=szresim)
                error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
            end

            %pixel locations to include in computation
            locs = gtim<255;

            % joint histogram
            sumim = 1+gtim+resim*num; 
            hs = histc(sumim(locs),1:num*num); 
            count = count + numel(find(locs));
            confcounts(:) = confcounts(:) + hs(:);
        end

        fclose(fid);

        % confusion matrix - first index is true label, second is inferred label
        % conf = zeros(num);
        conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
        rawcounts = confcounts;

        % Pixel Accuracy
        overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
        fprintf(txt_write_in,'Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

        % Class Accuracy
        class_acc = zeros(1, num);
        class_count = 0;
        fprintf(txt_write_in,'Accuracy for each class (pixel accuracy)\n');
        for i = 1 : num
            denom = sum(confcounts(i, :));
            if (denom == 0)
                denom = 1;
            end
            class_acc(i) = 100 * confcounts(i, i) / denom; 
            if i == 1
              clname = 'background';
            else
              clname = VOCopts.classes{i-1};
            end

            if ~strcmp(clname, 'void')
                class_count = class_count + 1;
                fprintf(txt_write_in,'%2d  %20s: %6.3f%%\n',i-1, clname, class_acc(i));
            end
        end
        fprintf(txt_write_in,'-------------------------\n');
        avg_class_acc = sum(class_acc) / class_count;
        fprintf(txt_write_in,'Mean Class Accuracy: %6.3f%%\n', avg_class_acc);

        accuracies = zeros(32,1);
        fprintf(txt_write_in,'Accuracy for each class (intersection/union measure)\n');

        real_class_count = 0;

        for j=1:num

           gtj=sum(confcounts(j,:));
           resj=sum(confcounts(:,j));
           gtjresj=confcounts(j,j);
           denom = (gtj+resj-gtjresj);

           if denom == 0
             denom = 1;
           end

           accuracies(j)=100*gtjresj/denom;

           clname = 'background';
           if (j>1), clname = VOCopts.classes{j-1};end;

           if ~strcmp(clname, 'void')
               real_class_count = real_class_count + 1;
           else
               if denom ~= 1
                   fprintf(1, 'WARNING: this void class has denom = %d\n', denom);
               end
           end

           if ~strcmp(clname, 'void')
               fprintf(txt_write_in,'%2d  %20s: %6.3f%%\n',j-1,clname,accuracies(j));
           end
        end

        avacc = sum(accuracies) / real_class_count;

        fprintf(txt_write_in,'-------------------------\n');
        fprintf(txt_write_in,'Average accuracy: %6.3f%%\n',avacc);
        fixIouTmp=sum(rawcounts');
        fixIouTmp1=fixIouTmp/sum(fixIouTmp);
        fixIouTmp2=fixIouTmp(2:maxlabel+1)/sum(fixIouTmp(2:maxlabel+1));
        fixIou=fixIouTmp1*accuracies;
        fixIouFont=fixIouTmp2*accuracies(2:maxlabel+1);
        fprintf(txt_write_in,'fixIou: %6.3f%%\n',fixIou);
        fprintf(txt_write_in,'fixIou of font ground: %6.3f%%\n',fixIouFont);

        final_result_save(bishun,dataset_type_num)=fixIouFont;
    end
end
disp(final_result_save)
fprintf(txt_write_in, '%f ',final_result_save);
fclose(txt_write_in);