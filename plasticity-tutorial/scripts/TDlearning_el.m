function TDlearning_el

% A simple grid world TD learning simulation
% TD(lambda)
%
%  by Naoshige Uchida

epsi = 0.3;              % epsilon greedy parameter
gamma = 0.99;            % discount factor
alpha = 0.6;             % learning rate
lambda = 0.2;            % eligibility trace

env.start = [4,3];
env.goal  = [1,4];
n_episode = 18;
pause_time = 0.2;

%  action 1 : East
%  action 2 : West
%  action 3 : South
%  action 4 : North

env.protect = zeros(4,5);

% possible actions
env.action(:,:,1) = [1 1 1 1 0;
    1 1 1 0 0;
    1 1 1 0 0;
    1 1 1 1 0];
env.action(:,:,2) = [0 1 1 1 1;
    0 1 1 1 0;
    0 1 1 1 0;
    0 1 1 1 1];
env.action(:,:,3) = [0 0 1 1 1;
    1 0 0 1 1;
    0 0 1 0 1;
    0 0 0 0 0];
env.action(:,:,4) = [0 0 0 0 0;
    0 0 1 1 1;
    1 0 0 1 1;
    0 0 1 0 1];

plotvaluefun = 1;

cum_r = zeros(1,n_episode);
V     =  zeros([size(env.protect)]);
V(env.goal(1),env.goal(2)) = 20;

figure(1)
set(gcf,'doublebuffer','on','color','w')
% colormap('jet')
colormap(yellowblue(255))

vidfile = VideoWriter('Grid_TD.mp4','MPEG-4');
vidfile.FrameRate = 3;  % Default 30
vidfile.Quality = 50;    % Default 75
open(vidfile);

sf = 22050;                 % sample frequency (Hz)


for i=1:n_episode
    state = env.start;
    nextaction = egreedy(state,V,env,epsi);
    reward=0;
    xx=[state(2)];
    yy=[state(1)];
    EL = zeros([size(env.protect)]);
    hold off
    if plotvaluefun
        subplot(1,2,1)

        imagesc(V,[-22 22])
        hold on; plot(xx,yy,'-w'); plot(state(2),state(1),'ow',...
            'markerfacecolor','w','markersize',8)
        draw_walls

        % colorbar
        title('Value')
        axis off
        drawnow

        subplot(1,2,2)
        imagesc(EL,[0 1]); hold on
        title('Eligibility Trace')
        axis off
        draw_walls

        if i==1
            waitforbuttonpress
        end
    end
    pause(pause_time)

    SoundFile = zeros(1,sf/vidfile.FrameRate);
    while reward ~= -100
        subplot(1,2,1)
        if state == env.goal
            if plotvaluefun
                gensound(2000,0.1)
            end
            break
        end


        if state == env.goal
            if plotvaluefun
                s = gensound(2000,0.1);
            end
            break
            SoundFile(1:length(s)) = s;
        end

        action = nextaction;
        prex = state(2);
        prey = state(1);

        [state, reward] = grid_move(state,action,env);

        xx=[xx state(2)];
        yy=[yy state(1)];
        X = zeros([size(env.protect)]);
        X(prey,prex) = 1;

        EL = EL.*lambda + X; 

        % TD-learning
        % TD-error
        TDerror = reward + gamma *V(state(1),state(2)) - V(prey,prex);

        % update the value function
        V = V + alpha * EL * TDerror;


        if plotvaluefun
            imagesc(V,[-22 22])
            hold on; plot(xx,yy,'-w'); plot(state(2),state(1),'ow',...
                'markerfacecolor','w','markersize',8)
            draw_walls

            % colorbar
            title('Value')
            axis off
            drawnow
        end
        subplot(1,2,2)
        imagesc(EL,[0 1])
        hold on; plot(xx,yy,'-w'); plot(state(2),state(1),'ow',...
            'markerfacecolor','w','markersize',8)
        draw_walls
        % colorbar
        title('Eligibility Trace')
        axis off
        drawnow

        pause(pause_time)

        nextaction = egreedy(state,V,env,epsi);
        cum_r(i) = reward + cum_r(i);

        F(i) = getframe(gcf);
        % writeVideo(vidfile,F(i),SoundFile);
        writeVideo(vidfile,F(i));

    end
end

figure
plot(cum_r)

close(vidfile)


function action = egreedy(state,V,env,epsi);

poss_action = find(shiftdim(env.action(state(1),state(2),:)));
for i = 1:4
    if ifany(i,poss_action)
        next_state = gridmove2(state,i);
        v(i) = V(next_state(1),next_state(2));
    else
        v(i) = NaN;
    end
end

if rand > (1-epsi)
    action = poss_action(ceil(rand*(length(poss_action))));
else
    [maxV ,action] = find(v==max(v));
    if length(action)>1
        action = action(ceil(rand*length(action)));
    end
end


function draw_walls

plot([.5 2.5],[1.5 1.5],'w','LineWidth',2)
plot([1.5 3.5],[2.5 2.5],'w','LineWidth',2)
plot([.5 2.5],[3.5 3.5],'w','LineWidth',2)
plot([3.5 4.5],[3.5 3.5],'w','LineWidth',2)
plot([4.5 4.5],[1.5 3.5],'w','LineWidth',2)
plot([.5 5.5],[.5 .5],'w','LineWidth',2)
plot([.5 5.5],[4.5 4.5],'w','LineWidth',2)
plot([.5 5],[.5 .45],'w','LineWidth',2)
plot([5.5 5.5],[.5 4.5],'w','LineWidth',2)



function state = gridmove2(state,action)

%  action 1 : East
%  action 2 : West
%  action 3 : South
%  action 4 : North
%
% state:  x,y coordinate
%
prexs = state(2); preys = state(1);

switch action
    case 1
        xs = prexs + 1;
        ys = preys;
    case 2
        xs = prexs - 1;
        ys = preys;
    case 3
        xs = prexs;
        ys = preys + 1;
    case 4
        xs = prexs;
        ys = preys - 1;
end
state = [ys,xs];

function gensound(f,dur)

cf = f;                  % carrier frequency (Hz)
sf = 22050;                 % sample frequency (Hz)
d = dur;                    % duration (s)
n = sf * d;                 % number of samples
s = (1:n) / sf;             % sound data preparation
s = sin(2 * pi * cf * s);   % sinusoidal modulation
s = s./500;                 % lower the volume
sound(s, sf);               % sound presentation


function [state, reward] = grid_move(state,action,environment)

%
%  Grid world maze
%
%  NISS2000    programed by K.Samejima  Aug.1, 2000
%
%
protect = environment.protect;
start = environment.start;
goal = environment.goal;

%
% state:  x,y coordinate
%
prexs = state(2); preys = state(1);

%  action 1 : East
%  action 2 : West
%  action 3 : South
%  action 4 : North

switch action
    case 1
        xs = prexs + 1;
        ys = preys;
    case 2
        xs = prexs - 1;
        ys = preys;
    case 3
        xs = prexs;
        ys = preys + 1;
    case 4
        xs = prexs;
        ys = preys - 1;
end
reward = -1;
if  xs  < 1 |  xs > size(protect,2)  |  ys < 1 |  ys>size(protect,1)
    xs = prexs;
    ys = preys;
else
    if protect(ys,xs) == 1
        xs = prexs;
        ys = preys;
    end
    if goal == [xs,ys];
        reward = 0;
        xs = start(2);
        ys = start(1);
    end
end

state = [ys,xs];



function cmap = yellowblue(m)
% FIREICE LightCyan-Cyan-Blue-Black-Red-Yellow-LightYellow Colormap
%
%  FIREICE(M) Creates a colormap with M colors
%
%   Inputs:
%       M - (optional) an integer between 1 and 256 specifying the number
%           of colors in the colormap. Default is 64.
%
%   Outputs:
%       CMAP - an Mx3 colormap matrix
%
%   Example:
%       imagesc(peaks(500))
%       colormap(fireice), colorbar
%
%   Example:
%       imagesc(interp2(rand(10),2))
%       colormap(fireice); colorbar
%
% See also: hot, jet, hsv, gray, copper, bone, cold, vivid, colors
%
% Author: Joseph Kirk
% Email: jdkirk630@gmail.com
% Release: 1.0
% Date: 07/29/09


% Default Colormap Size
if ~nargin
    m = 32;
end

% Blue-Black-Yellow
clrs = [ 0 0.75 1;0 0 0; 1 1 0];
clrs = [ 0 0 0;0 0 0; 1 1 0];

y = -1:1;
if mod(m,2)
    delta = min(1,2/(m-1));
    half = (m-1)/2;
    yi = delta*(-half:half)';
else
    delta = min(1,2/m);
    half = m/2;
    yi = delta*nonzeros(-half:half);
end
cmap = interp2(1:3,y,clrs,1:3,yi);

