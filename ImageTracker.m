classdef ImageTracker<handle
    properties
        tracker_id % tracker id
        trackers % vehicle trackers
        dt % delta time
        tracesize % maximum length of trace
        dist_threshold % maximum matching distance
        frame_threshold % maximum missing frames
    end
    methods
        function obj=ImageTracker(delta_t, tracesize, dist_threshold, frame_threshold)
            obj.tracker_id = 0;
            obj.trackers = [];
            obj.dt = delta_t;
            obj.tracesize=tracesize;
            obj.dist_threshold = dist_threshold;
            obj.frame_threshold = frame_threshold;
        end
        function r=update(obj, measures, labels)
            % measures in the size of nx2
            if size(obj.trackers,1)==0
                for i=1:size(measures,1)
                    obj.tracker_id = obj.tracker_id +1;
                    new_tracker = Tracker(obj.tracker_id,obj.dt,obj.tracesize,measures(i,:),labels(i,:));
                    obj.trackers=[obj.trackers;new_tracker];
                end
            end
            cost = [];
            for i=1:size(obj.trackers,1)
                cost=[cost;reshape(vecnorm(obj.trackers(i).prediction-measures,2,2),[1,size(measures,1)])];
            end
            [M,uR,uC] = matchpairs(cost,1000);
            tracker_to_remove = [];
            for i=1:size(M,1)
                if cost(M(i,1),M(i,2))>obj.dist_threshold
                    obj.trackers(M(i,1)).num_frames = obj.trackers(M(i,1)).num_frames+1;
                    if obj.trackers(M(i,1)).num_frames>obj.frame_threshold
                        tracker_to_remove = [tracker_to_remove;M(i,1)];
                    end
                else
                    pred_res=obj.trackers(M(i,1)).predict(measures(M(i,2),:));
                    obj.trackers(M(i,1)).trace = [obj.trackers(M(i,1)).trace;pred_res];
                end
            end
            for i=1:size(uR,1)
                obj.trackers(uR(i,1)).num_frames = obj.trackers(uR(i,1)).num_frames+1;
                if obj.trackers(uR(i,1)).num_frames>obj.frame_threshold
                    tracker_to_remove = [tracker_to_remove;uR(i,1)];
                end
            end
            obj.trackers(tracker_to_remove)=[];
            for i=1:size(uC)
                obj.tracker_id = obj.tracker_id +1;
                newTracker = Tracker(obj.tracker_id,obj.dt,obj.tracesize,measures(uC(i),:),labels(uC(i),:));
                obj.trackers=[obj.trackers;newTracker];
            end
        end
    end
end