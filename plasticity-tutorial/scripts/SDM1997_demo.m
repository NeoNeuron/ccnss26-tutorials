% SDM1997_demo
 
% Temporal difference model of dopamine neurons
% Shultz, Dayan & Montague (1997)
% eligibility traces are added to the original model

clear

gamma = 0.97;   % discount factor
alpha = .6;     % learning rate
lambda = 0.8;   % eligibility trace

n_trials = 25;

t_step = 5; % time steps per second
t_max = 5*t_step;   % Total time of simulation
t_stim = 1*t_step;  % Timing of stimulus
t_rew = 3*t_step;   % Timing of reward

T = ([1:t_max]-t_stim)/t_step; % Time in seconds

% stimulus (x): complete serial compound
X = zeros(t_max,t_max);
for i = 1:t_max-t_stim-1
    X(i+t_stim,i+t_stim) = 1;
end

% weight vector
w = zeros(1,t_max);
W = w;

% reward function
r = zeros(1,t_max); % reward function
r(t_rew) = 1;

% value function
v = zeros(1,t_max);

figure(1)
set(gcf,'color','w')

for n = 1:n_trials
    % Initialize the eligibility trace
    el = zeros(1,t_max);

    % TD error (initialize)
    D = zeros(1,t_max);

    % compute value
    v = w*X;

    
    for t = 1:t_max-1
        
        d = r(t)+gamma*v(t+1)-v(t); % calculate TD error (d)

        el = el*lambda + X(t,:);    % eligibility trace

        dw = alpha*el*d;            % weight update

        w = w + dw;
        
        v = w*X;

        D(t) = d; % compute value

        % plot
        LW = 2; % line width
        fsize = 15; % font size
        fsize2 = 25; % font size
        
        % Stimulus (x)
        subplot(11,1,1)
        imagesc(T,1,X(t,:),[0 1])
        set(gca,'xtick',[0 1 2 3],'tickdir','out','ytick',[],'fontsize',fsize)
        box off
        ylabel('$x $','interpreter','latex','Rotation',0,'fontsize',fsize2)
        colormap(gray)
        title(sprintf('Trial %1.0d',n))
        ax = axis;

        % Eligibility trace (e)
        subplot(11,1,2)
        imagesc(T,1,el,[0 1])
        set(gca,'xtick',[0 1 2 3],'tickdir','out','ytick',[],'fontsize',fsize2)
        box off
        ylabel('$e $','interpreter','latex','Rotation',0)
        colormap(gray)
        ax = axis;

        % Reward
        subplot(11,1,3:5)
        plot(T(1:t),r(1:t),'linewidth',LW)
        ylabel('$R$','interpreter','latex','Rotation',0,'fontsize',fsize2)
        box off
        axis([-1 4 -0.3 1])
        set(gca,'xtick',[0 1 2 3],'tickdir','out','ytick',[0 1],'fontsize',fsize)

        % TD error
        subplot(11,1,6:8)
        plot(T(1:t),D(1:t),'linewidth',LW)
        ylabel('$\delta$','interpreter','latex','Rotation',0,'fontsize',fsize2)
        box off
        axis([-1 4 -0.3 1])
        set(gca,'xtick',[0 1 2 3],'tickdir','out','ytick',[0 1],'fontsize',fsize)

        % Value
        subplot(11,1,9:11)
        plot(T,v,'linewidth',LW)
        ylabel('$V$','interpreter','latex','Rotation',0,'fontsize',fsize2)
        box off
        axis([-1 4 0 1])
        set(gca,'xtick',[0 1 2 3],'tickdir','out','ytick',[0 1],'fontsize',fsize)
        xlabel('Time (sec)')
        axis([-1 4 -0.3 1])

        drawnow

    end
    
end
