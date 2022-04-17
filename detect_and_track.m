clc; clear;
gpuDevice(1);
image_path = "UAV-benchmark-M";
trained_detector_path = 'workspace_resnet18_augmented.mat';
testDataTb = load('vehicleDataTable_2.mat');
testDataTb = testDataTb.vehicleDataTable;
testDataTb.imageFileName = fullfile(image_path,testDataTb.imageFileName);
label = {['car'], ['truck'],['bus']};
imdsTest = imageDatastore(testDataTb{:,'imageFileName'});
bldsTest = boxLabelDatastore(testDataTb(:,label));
testData = combine(imdsTest,bldsTest);
inputSize = [540 1024 3];
testData = transform(testData,@(data)preprocessData(data,inputSize));

pretrained = load(trained_detector_path);
detector = pretrained.detector;
detectionResults = detect(detector,testData,'MinibatchSize',1);


image_path = 'UAV-benchmark-M';
all_images = testDataTb.imageFileName;
all_bboxes = detectionResults.Boxes;
img_tracker = ImageTracker(0.1,5,50,10);
tracker_count = [];
figure;
for i=1:height(all_images)
    clf;
    complete_image_path = testDataTb.imageFileName{i,:};
    image_labels = detectionResults.Labels{i,:};
    measure = all_bboxes{i,:};
    measure(:,1) = measure(:,1)+0.5*measure(:,3);
    measure(:,2) = measure(:,2)+0.5*measure(:,4);
    measure = measure(:,1:2);
    img_tracker.update(measure,image_labels);
    tracker_count = [tracker_count;size(img_tracker.trackers,1)];
    measure(:,3) = 1;
    I = imread(complete_image_path);
    for j=1:size(img_tracker.trackers)
        %display(img_tracker.trackers(j).trace(end,:));
        %display(img_tracker.trackers(j).id);
        I = insertObjectAnnotation(I,'circle',[img_tracker.trackers(j).trace(end,:),1],...
            append(string(img_tracker.trackers(j).label),': ',string(img_tracker.trackers(j).id)));
    end
    imshow(I);
    hold on;
    pause(0.2);
end


function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
sz = size(data{1});
rout = affineOutputView(sz,tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);

% Sanitize box data, if needed.
%data{2} = helperSanitizeBoxes(data{2}, sz);

% Warp boxes.
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
sz = size(data{1},[1 2]);
scale = targetSize(1:2)./sz;
data{1} = imresize(data{1},targetSize(1:2));

% Sanitize box data, if needed.
%data{2} = helperSanitizeBoxes(data{2}, sz);

boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);

% Resize boxes.
data{2} = bboxresize(boxEstimate,scale);
end