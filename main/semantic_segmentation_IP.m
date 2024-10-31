scan_filenames = listdir('./data_odometry_velodyne/dataset/sequences');
calib_path = './data_odometry_velodyne/dataset/sequences/calib';
num_scans = length(scan_filenames);
iou = {};
miou = 0;
class = {};
iou_class = {};
center = {[],[],[],[],[],[],[]};
xx=[-0.06;0.27;-0.08];

hWaitbar = waitbar(0, 'Processing...');
for i_all = 1:num_scans-1
    file_idx = i_all;
    file = num2str(file_idx, '%06.f');
    laser_path = ['./data_odometry_velodyne/dataset/sequences/',file,'.bin'];
    pos_path = ['./data_odometry_velodyne/dataset/sequences/',file,'.txt'];
    image_path = ['./data_odometry_velodyne/dataset/sequences/',file,'.png'];
    label_path = ['./data_odometry_velodyne/dataset/sequences/',file,'.label'];
    
    delta = 5; 
    delta_ex = 5; 
    repetitive = 3.5;

    max_distance = 80;
    alpha = 1.1;
    step_size = 20;
    k = alpha/step_size;
    %% 
    fid=fopen(laser_path,'rb');
    [a,count1]=fread(fid,'float32');
    fclose(fid);
    x = a(1:4:end);
    y = a(2:4:end);
    z = a(3:4:end);
    data = pointCloud([x y z]);
    %% 
    fid=fopen(pos_path,'r');
    C  = textscan(fid,'%s %f %f %f %f %f','delimiter', ' ');
    fclose(fid);
    if isempty(C)
        iou{i_all}=[];
        continue;
    end
    
    objects = [];
    for o = 1:numel(C{1})
      lbl = C{1}(o); % for converting: cell -> string
      objects(o).type = lbl{1};  % 'Car', 'Pedestrian', ...
      objects(o).score = C{2}(o);
      objects(o).x1 = C{3}(o); % left
      objects(o).y1 = C{4}(o); % top
      objects(o).x2 = C{5}(o); % right
      objects(o).y2 = C{6}(o); % bottom
      objects(o).pos = [objects(o).x1-delta,objects(o).y1-delta,objects(o).x2+delta,objects(o).y2+delta];    
    end
    %% 
    fid1=fopen(calib_path);
    fgetl(fid1);
    fgetl(fid1);
    c1=textscan(fid1,'%*s %f %f %f %f %f %f %f %f %f %f %f %f',1);
    fgetl(fid1);
    fgetl(fid1);
    c3=textscan(fid1,'%*s %f %f %f %f %f %f %f %f %f %f %f %f',1);
    fclose(fid1);
    c1=cell2mat(c1);c1=reshape(c1,[4,3]);c1=c1';
    c3=cell2mat(c3);c3=reshape(c3,[4,3]);c3=c3';
    c3(4,:)=[0 0 0 1];
    h=c1*c3;
    i=0;
    objects_line={};
    keys=[];
    object_pointcloud_all = {};segment_org_idx = {};l_object = {};object_pointcloud_all_ex = {};segment_org_idx_ex = {};l_object_ex = {};
    while (1)
        i = i+1;
        if i > numel(objects)
            break;
        end
        [object_pointcloud_all{i},segment_org_idx{i},l_object{i}] = inverse_projection(data,h,objects(i).pos,cor,i,max_distance,alpha,step_size);
        instance_seg1(i).pos = [objects(i).pos(1)-delta_ex,objects(i).pos(2)-delta_ex,objects(i).pos(3)+delta_ex,objects(i).pos(4)+delta_ex];
        [object_pointcloud_all_ex{i},segment_org_idx_ex{i},l_object_ex{i}] = inverse_projection(data,h,instance_seg1(i).pos,cor,i,max_distance,alpha,step_size);
    end
    object_pointcloud_all1=object_pointcloud_all;
    segment_org_idx1=segment_org_idx;
    %% 
    signal1=[];
    seeds = {};
    for i = 0:numel(segment_org_idx)*6-1
        segment_idx = segment_org_idx{fix(i/6)+1}{rem(i,6)+1};
        if isempty(segment_idx)
           continue
        end
        for j = fix(i/6)+2:numel(segment_org_idx)
            for k = 1:6
                if isempty(segment_org_idx{j}{k})
                    continue
                end
                if length(segment_idx) >length(segment_org_idx{j}{k})
                    max_one = segment_idx;
                    min_one = segment_org_idx{j}{k};
                    signal = 0;
                else
                    max_one = segment_org_idx{j}{k};
                    min_one = segment_idx;
                    signal = 1;
                end
                result = length(intersect (min_one,max_one));
                if result/length(max_one) > 0.5 && result/length(max_one) ~= 1
                    if result/length(min_one) > 0.9 
                        every_len1 = sum(cellfun(@numel,segment_org_idx1{fix(i/6)+1}));
                        every_len2 = sum(cellfun(@numel,segment_org_idx1{j}));
                        if length(segment_idx) / every_len1 > length(segment_org_idx{j}{k}) / every_len2
                            segment_org_idx{j}{k} = [];
                            object_pointcloud_all{j}{k} = [];
                        else
                            segment_org_idx{fix(i/6)+1}{rem(i,6)+1} = [];
                            object_pointcloud_all{fix(i/6)+1}{rem(i,6)+1} = [];
                        end
                    else    
                        if signal == 1
                            segment_org_idx{fix(i/6)+1}{rem(i,6)+1} = [];
                            object_pointcloud_all{fix(i/6)+1}{rem(i,6)+1} = [];
                        else 
                            segment_org_idx{j}{k} = [];
                            object_pointcloud_all{j}{k} = [];
                        end
                    end
                    signal1=[signal1;fix(i/6)+1,rem(i,6)+1,j,k];
                end
           end
        end
    end
    for i = 1:numel(segment_org_idx)
        intersection_1 = [];
        intersection_2 = [];
        for j = 1:6
            length1(j) = length(object_pointcloud_all{i}{j});
        end
        index = find(length1 ~= 0,2);
        index = index';
        if length(index) > 1 && length1(index(1))/length1(index(2)) < repetitive
            for j = 1:6
                intersection_1(j) = length(intersect (object_pointcloud_all{i}{index(1)},object_pointcloud_all_ex{i}{j}));
            end
            if all(intersection_1 == 0) || length(find(intersection_1 == max(intersection_1))) >1
                break;
            end
            intersection_1_dif = length(object_pointcloud_all_ex{i}{find(intersection_1 == max(intersection_1))}) - length(object_pointcloud_all{i}{index(1)});
            for j = 1:6
                intersection_2(j) = length(intersect (object_pointcloud_all{i}{index(2)},object_pointcloud_all_ex{i}{j}));
            end
            if all(intersection_2 == 0) || length(find(intersection_2 == max(intersection_2))) >1
                break;
            else
                intersection_2_dif = length(object_pointcloud_all_ex{i}{find(intersection_2 == max(intersection_2))}) - length(object_pointcloud_all{i}{index(2)});
            end
            if intersection_1_dif > intersection_2_dif
                object_pointcloud_all{i}{index(1)} = [];
                segment_org_idx{i}{index(1)} = [];
                length1(index(1)) = 0;
            else
                object_pointcloud_all{i}{index(2)} = [];
                segment_org_idx{i}{index(2)} = [];
                length1(index(2)) = 0;
            end
        end
        index_max = find(length1 == max(length1));
        if isempty(object_pointcloud_all{i}{index_max(1)}) == 0
            seeds{i} = segment_org_idx{i}{index_max(1)};
        else
            seeds{i} = [];
        end
    end
    % % hold off
    %% 
        IDX = DBSCAN_fin(seeds,data.Location,0.2,5);
        data_colors = zeros(data.Count, 3);
        org_lb_set = unique(IDX);
        data_lb_set = LABEL_INDEXES(unique(IDX)+1);
        for ith = 1:numel(data_lb_set)
            semantic_idx = data_lb_set(ith);
            org_idx = org_lb_set(ith);
            semantic_color = LABEL_COLORMAP_uint((semantic_idx == LABEL_INDEXES), :);
            points_idx_this_semantic = find(org_idx == IDX);
            data_colors(points_idx_this_semantic, :) = repmat(semantic_color, numel(points_idx_this_semantic), 1);
        end
        data.Color = uint8(data_colors);
    %% 
        [pc_lb, pc_id] = readLabel(label_path);
        [pc_id_set, num_each_id] = unique(pc_id);
        num_id = length(pc_id_set);
        iou_i = [];
        for jth = 2:numel(org_lb_set)%org_lb_set
            org_idx = org_lb_set(jth);
            points_my = find(org_idx == IDX);
            points_idx_this_instance = {};
            iou_pre = [];
            apd = 0;
            semantic_right = false;
            for ith = 2:numel(pc_id_set)
                instance_idx = pc_id_set(ith);
                points_idx_this_instance{ith} = find(instance_idx == pc_id);
                iou_pre(ith) = length(intersect(points_idx_this_instance{ith},points_my));
            end
            iou_idx = find(iou_pre == max(iou_pre));
            iou_target = iou_pre(iou_idx);
            if isempty(iou_target) || iou_target(1) == 0 
                continue;
            else
                point_center = [mean(x(points_my)) mean(y(points_my)) mean(z(points_my))];
                point_distance = norm(point_center-xx);
                apd = iou_target(1)/(length(points_my)+length(points_idx_this_instance{iou_idx(1)})-iou_target(1));
                if apd < 0.05
                    continue;
                end
                iou_i = [iou_i,apd];
            end
            if find(strcmp(objects(jth-1).type,class)) 
                iou_class{find(strcmp(objects(jth-1).type,class))} = [iou_class{find(strcmp(objects(jth-1).type,class))},apd];
            else
                class{end+1} = objects(jth-1).type;
                iou_class{end+1} = apd;
            end

            if point_distance < 10
                center{1} = [center{1},apd];
            elseif point_distance > 10 && point_distance < 20
                center{2} = [center{2},apd];
            elseif point_distance > 20 && point_distance < 30
                center{3} = [center{3},apd];
            elseif point_distance > 30 && point_distance < 40
                center{4} = [center{4},apd];
            elseif point_distance > 40 && point_distance < 50
                center{5} = [center{5},apd];
            elseif point_distance > 50 && point_distance < 60
                center{6} = [center{6},apd];
            else
                center{7} = [center{7},apd];
            end

        end
        iou{i_all} = iou_i;
        waitbar(i_all/(num_scans-1), hWaitbar, sprintf('Processing... %.1f%%', (i_all/(num_scans-1))*100));
end

for i = 1:numel(iou_class)
    array = iou_class{i}>0.05;
    array_target{i} = iou_class{i}(array);
end
iou_length = cellfun(@length,array_target);
miou = cellfun(@mean,array_target);
miou_distance = cellfun(@mean,center);
close(hWaitbar);