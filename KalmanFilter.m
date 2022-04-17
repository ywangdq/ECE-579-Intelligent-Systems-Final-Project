classdef KalmanFilter<handle
    properties
        dt % delta time
        x  % state vector
        y  % observation vector
        u  % acceleration
        A  % state transition matrix
        B  % state transition matrix
        Q  % state noise covariance
        H  % output matrix
        R  % measure noise covariance
        P  % error covariance
        predictedState
        predictedErrorCov
        kalmanGain
    end
    methods
        function obj=KalmanFilter(delta_t,measure)
            obj.dt = delta_t;
            obj.x = [reshape(measure,[2,1]);0;0];
            obj.y = [0;0];
            obj.u = 0;
            obj.A = [1 0 obj.dt 0;
                     0 1 0 obj.dt;
                     0 0 1 0;
                     0 0 0 1];
            obj.B = [obj.dt^2/2;obj.dt^2/2;obj.dt;obj.dt];
            obj.Q = [obj.dt^4/4 0 obj.dt^3/2 0;
                     0 obj.dt^4/4 0 obj.dt^3/2;
                     obj.dt^3/2 0 obj.dt^2 0;
                     0 obj.dt^3/2 0 obj.dt^2];
            obj.H = [1 0 0 0;0 1 0 0];
            obj.R = [1 0;0 1];
            obj.P = [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];
        end
        function r=predict(obj)
            obj.predictedState = obj.A*obj.x+obj.B*obj.u;
            obj.predictedErrorCov = obj.A*obj.P*obj.A'+obj.Q;
            r = obj.predictedState([1:2]);
        end
        function r=update(obj, measure)
            obj.kalmanGain = obj.predictedErrorCov*obj.H'*...
                pinv(obj.H*obj.predictedErrorCov*obj.H'+obj.R);
            obj.x = obj.predictedState + obj.kalmanGain*...
                (measure-obj.H*obj.predictedState);
            obj.P = (eye(4)-obj.kalmanGain*obj.H)*obj.predictedErrorCov;
        end
    end
end