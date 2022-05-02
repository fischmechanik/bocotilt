clear all;

% Paths...
PATH_FIELDTRIP = '/home/plkn/fieldtrip-master/';
PATH_EEGCAP_MODELS = '/mnt/data_dump/bocotilt/0_eegcap_models/';
PATH_WORKING_DIRECTORY = '/home/plkn/Desktop/elecpos_practice/'; 

% Init eeglab
addpath(PATH_FIELDTRIP);
ft_defaults;

% Set subject
subject = 'VP04';

% Load a headshape
head_surface = ft_read_headshape([PATH_EEGCAP_MODELS, subject, '/Model.obj']);

% Convert units to mm
head_surface = ft_convert_units(head_surface, 'mm');

% Plot
ft_plot_mesh(head_surface);

% Manually select fiducial coordinates
cfg = [];
cfg.method = 'headshape';
fiducials = ft_electrodeplacement(cfg, head_surface);


% Convert to CTF-coordinates using feducials
cfg = []
cfg.method = 'fiducial';
cfg.coordsys = 'ctf';
cfg.fiducial.nas = fiducials.elecpos(1, :); %  nasion
cfg.fiducial.lpa = fiducials.elecpos(2, :); % left pre-auricular point
cfg.fiducial.rpa = fiducials.elecpos(3, :); % right pre-auricular point
head_surface = ft_meshrealign(cfg, head_surface);

% Plot CFT axes
ft_plot_axes(head_surface);
ft_plot_mesh(head_surface);

% Save
save([PATH_WORKING_DIRECTORY, 'head_surface_fiducials_added.mat'], 'head_surface');

% Load
load([PATH_WORKING_DIRECTORY, 'head_surface_fiducials_added.mat']);

% Identify electrode positions
cfg = [];
cfg.method = 'headshape';
cfg.channel = {'FCz', 'Cz', 'POz'};
elec = ft_electrodeplacement(cfg, head_surface);




