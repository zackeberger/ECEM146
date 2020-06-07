% Import 300x400 picture of Bruin Bear, then display it
pic = imread('../img/UCLA_Bruin.jpg');

% Initialize K-means algorithm; K > 0
K = 16;

prototypes = zeros(3,K);            % Initialize prototypes to zero
prototypes(:,1) = [229, 249, 250];  % Set the first prototype to first pixel


% Set the rest of the prototypes using furthest-first heuristic
for k = 2 : K
    
    % Start with the first pixel as the furthest pixel
    % fpd = furthest_pixel_distance
    furthest_pixel = [229;249;250];
    fpd = 0;
    
    % Iterate over all pixels
    for r = 1 : 300
        for c = 1 : 400
            
            current_pixel = double([pic(r,c,1);pic(r,c,2);pic(r,c,3)]);
            
            % Find the min distance of current_pixel to previously computed prototypes
            % Denote it cpd. Start with distance to first prototype
            cpd = norm(current_pixel - prototypes(:,1))^2;
            
            % Iterate over all previously computed prototypes
            for p = 2 : (k - 1)
                % Obtain distance to next prototype (npd)
                npd = norm(current_pixel - prototypes(:,p))^2;
                if npd < cpd
                    cpd = npd;
                end  
            end
            
            % If the closest prototype distance is the furthest pixel
            % distance, we have a new furthest pixel
            if cpd > fpd
                fpd = cpd;
                furthest_pixel = current_pixel;
            end
            
        end
    end
    
    % Set the furthest pixel as the next prototype
    prototypes(:,k) = furthest_pixel;
    
end


% Run K-means for 10 iterations
J = zeros(1,10);                    % Track objective for each iteration
pixel_indices = zeros(300,400);     % Each pixel refers to its closest prototype
Num_Iterations = 10;

for i = 1 : Num_Iterations
    
    % Assign each pixel to closest prototype
    for r = 1 : 300
        for c = 1 : 400
            
            current_pixel = double([pic(r,c,1);pic(r,c,2);pic(r,c,3)]);
            
            % Calculate closest prototype.
            % Start with distance to first.
            % prototype
            cpd = norm(current_pixel - prototypes(:,1))^2;
            closest_prototype = 1;
            
            % Iterate over all other prototypes
            for p = 2 : K
                npd = norm(current_pixel - prototypes(:,p))^2;
                if npd < cpd
                    cpd = npd;
                    closest_prototype = p;
                end 
            end
            
            % Set closest prototype
            pixel_indices(r,c) = closest_prototype;
    
        end
    end
    
    % Now, pixel_indicies contains the number prototype that each pixel is
    % a part of. Re-estimate center of each cluster using this info
    prototypes = zeros(3,K);    % Refresh the prototype values
    num_samples = zeros(1,K);   % Number of datapoints in each cluster
    
    for r = 1 : 300
        for c = 1 : 400
            
            current_pixel = double([pic(r,c,1);pic(r,c,2);pic(r,c,3)]);
            
            num_samples(1,pixel_indices(r,c)) = num_samples(1,pixel_indices(r,c)) + 1;
            prototypes(:, pixel_indices(r,c)) = prototypes(:, pixel_indices(r,c)) + current_pixel;
            
        end
    end
    
    prototypes = prototypes ./ num_samples;
    
    
    
    % Calculate objective function
    total_dist = 0;
    for r = 1 : 300
        for c = 1 : 400
                
            for k = 1 : K
           
                current_pixel = double([pic(r,c,1);pic(r,c,2);pic(r,c,3)]);
                
                if pixel_indices(r,c) == k
                    total_dist = total_dist + norm(current_pixel - prototypes(:,k))^2;
                end
                
            end
        end
    end
   
    J(i) = total_dist;
end

% Set the picture to correct pixels
for r = 1 : 300
    for c = 1 : 400
        pic(r,c,:) = prototypes(:, pixel_indices(r,c));
    end
end

imshow(pic);

% Plot objective function vs. iterations
% iterations = [1,2,3,4,5,6,7,8,9,10];
% scatter(iterations, J, 'filled');

% title("J Value vs. Iteration");
% xlabel("Iteration Number");
% ylabel("J Value");
