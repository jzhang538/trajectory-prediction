%% Process dataset into mat files %%

clear;
clc;

%% Inputs:
% Locations of raw input files:
us101_1 = '/home/henry/Desktop/raw/us101-0750-0805.txt';
us101_2 = '/home/henry/Desktop/raw/us101-0805-0820.txt';
us101_3 = '/home/henry/Desktop/raw/us101-0820-0835.txt';
i80_1 = '/home/henry/Desktop/raw/i80-1600-1615.txt';
i80_2 = '/home/henry/Desktop/raw/i80-1700-1715.txt';
i80_3 = '/home/henry/Desktop/raw/i80-1715-1730.txt';


%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: Lane Id
7: Velocity
8: Acceleration
9: Lateral maneuver
10: Longitudinal maneuver
11: Num_nbrs
12-50: Neighbor Car Ids
%}



%% Load data and add dataset id
disp('Loading data...')
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

for k = 1:6
    traj{k} = traj{k}(:,[1,2,3,6,7,15,13,14]);
    if k <=3
        traj{k}(traj{k}(:,6)>=6,6) = 6;
    end
end

vehTrajs{1} = containers.Map;
vehTrajs{2} = containers.Map;
vehTrajs{3} = containers.Map;
vehTrajs{4} = containers.Map;
vehTrajs{5} = containers.Map;
vehTrajs{6} = containers.Map;

vehTimes{1} = containers.Map;
vehTimes{2} = containers.Map;
vehTimes{3} = containers.Map;
vehTimes{4} = containers.Map;
vehTimes{5} = containers.Map;
vehTimes{6} = containers.Map;

%% Parse fields (listed above):
disp('Parsing fields...')


m1 = 0;
i1 = 0;

for ii = 1:6
    vehIds = unique(traj{ii}(:,2));

    for v = 1:length(vehIds)
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:);
    end

    timeFrames = unique(traj{ii}(:,3));

    for v = 1:length(timeFrames)
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end
    
    for k = 1:length(traj{ii}(:,1))        
        time = traj{ii}(k,3);
        dsId = traj{ii}(k,1);
        vehId = traj{ii}(k,2);
        vehtraj = vehTrajs{ii}(int2str(vehId));
        ind = find(vehtraj(:,3)==time);
        ind = ind(1);
        lane = traj{ii}(k,6);
        
        
        % Get lateral maneuver:
        ub = min(size(vehtraj,1),ind+40);
        lb = max(1, ind-40);
        if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6)
            traj{ii}(k,9) = 3;
        elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
            traj{ii}(k,9) = 2;
        else
            traj{ii}(k,9) = 1;
        end
        
        
        % Get longitudinal maneuver:
        ub = min(size(vehtraj,1),ind+50);
        lb = max(1, ind-30);
        if ub==ind || lb ==ind
            traj{ii}(k,10) =1;
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
            if vFut/vHist <0.8
                traj{ii}(k,10) =2;
            else
                traj{ii}(k,10) =1;
            end
        end
        
        for l = 12:100
        	traj{ii}(k,l) = -1;
        end

        % Get Nbr Vehicles:
        num_nbrs = 0;
        dis = 150;
        t = vehTimes{ii}(int2str(time));
        frameEgo = t(t(:,6) == lane,:);
        frameL = t(t(:,6) == lane-1,:);
        frameR = t(t(:,6) == lane+1,:);

        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                dy = frameL(l,5)-traj{ii}(k,5);
            	dx = frameL(l,4)-traj{ii}(k,4);
                if (dx*dx + dy*dy) < (dis*dis)
                	num_nbrs = num_nbrs + 1;              
                    traj{ii}(k,11+num_nbrs) = frameL(l,2);
                end
            end
        end
        for l = 1:size(frameEgo,1)
            dy = frameEgo(l,5)-traj{ii}(k,5);
        	dx = frameEgo(l,4)-traj{ii}(k,4);
            if (dx*dx + dy*dy) < (dis*dis) && dy~=0
            	num_nbrs = num_nbrs + 1;
            	% disp(num_nbrs);
            	% disp(11+num_nbrs);
                traj{ii}(k,11+num_nbrs) = frameEgo(l,2);
            end
        end
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                dy = frameR(l,5)-traj{ii}(k,5);
            	dx = frameR(l,4)-traj{ii}(k,4);
                if (dx*dx + dy*dy) < (dis*dis)
                	num_nbrs = num_nbrs + 1;
                    traj{ii}(k,11+num_nbrs) = frameR(l,2);
                end
            end
        end
        
        traj{ii}(k,11) = num_nbrs;
    end
end

disp(m1)
disp(i1)

%% Split train, validation, test
disp('Splitting into train, validation and test sets...')

trajAll = [traj{1};traj{2};traj{3};traj{4};traj{5};traj{6}];
clear traj;

trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:6
    ul1 = round(0.7*max(trajAll(trajAll(:,1)==k,3)));
    ul2 = round(0.8*max(trajAll(trajAll(:,1)==k,3)));
    
    trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :)];
    trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :)];
    trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :)];
end


tracksTr = {};
for k = 1:6
    trajSet = trajTr(trajTr(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,4,5,7,8])';
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

tracksVal = {};
for k = 1:6
    trajSet = trajVal(trajVal(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,4,5,7,8])';
        tracksVal{k,carIds(l)} = vehtrack;
    end
end

tracksTs = {};
for k = 1:6
    trajSet = trajTs(trajTs(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,4,5,7,8])';
        tracksTs{k,carIds(l)} = vehtrack;
    end
end


%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing

disp('Filtering edge cases...')

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    if tracksTr{trajTr(k,1),trajTr(k,2)}(1,101) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>=t+70
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);


indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if tracksVal{trajVal(k,1),trajVal(k,2)}(1,101) <= t && tracksVal{trajVal(k,1),trajVal(k,2)}(1,end)>=t+70
        indsVal(k) = 1;
    end
end
trajVal = trajVal(find(indsVal),:);


indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if tracksTs{trajTs(k,1),trajTs(k,2)}(1,101) <= t && tracksTs{trajTs(k,1),trajTs(k,2)}(1,end)>=t+70
        indsTs(k) = 1;
    end
end
trajTs = trajTs(find(indsTs),:);

%% Save mat files:
disp('Saving mat files...')

traj = trajTr;
tracks = tracksTr;
save('TrainSet','traj','tracks');

traj = trajVal;
tracks = tracksVal;
save('ValSet','traj','tracks');

traj = trajTs;
tracks = tracksTs;
save('TestSet','traj','tracks');