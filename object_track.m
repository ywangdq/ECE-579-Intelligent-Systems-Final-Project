image_file = load('vehicleDataTable_Trackinguse.mat');
image_path = 'UAV-benchmark-M';
all_images = image_file.vehicleDataTable.imageFileName;
all_bboxes = image_file.vehicleDataTable.bbox;
img_tracker = ImageTracker(0.1,5,100,5);
figure;
for i=1:height(all_images)
    clf;
    complete_image_path = fullfile(image_path,all_images{i,:});
    measure = all_bboxes{i,:};
    measure(:,1) = measure(:,1)+0.5*measure(:,3);
    measure(:,2) = measure(:,2)+0.5*measure(:,4);
    measure = measure(:,1:2);
    img_tracker.update(measure);
    measure(:,3) = 1;
    I = imread(complete_image_path);
    for j=1:size(img_tracker.trackers)
        display(img_tracker.trackers(j).trace(end,:));
        display(img_tracker.trackers(j).id);
        I = insertObjectAnnotation(I,'circle',[img_tracker.trackers(j).trace(end,:),1],img_tracker.trackers(j).id);
    end
    imshow(I);
    hold on;
    pause(0.2);
end