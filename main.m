% Author: reda-kamal
%% Three-domain Stefan problem with finite contact resistance
%  Explicit finite-difference solver for a contact-resistance Stefan problem.
%  Plots per case: profiles, profile deviation from T_f, h_eff(t), interface flux q(t)

% Ensure helper functions in ./functions are on the path
func_dir = fullfile(fileparts(mfilename('fullpath')), 'functions');
if exist(func_dir, 'dir')
    paths = strsplit(path, pathsep);
    if ~any(strcmp(func_dir, paths))
        addpath(func_dir);
    end
end

clear; close all; clc;

%% --- USER CONTROLS -----------------------------------------------------
t_phys = 0.1;                 % representative time for temperature profiles [s]
R_c    = 2e-5;                % contact resistance [m^2 K/W] (shared across cases)

% Explicit finite-difference snapshot controls (passed to explicit solver)
explicit_opts = struct( ...
    'CFL',            0.30, ...    % stability number for FTCS updates
    'wall',           struct( ...  % wall mesh: physical extent & resolution
                           'length', 7.5e-3, ...   % [m] domain size
                           'cells',  600 ...       % number of uniform cells
                           ), ...
    'fluid',          struct( ...  % solid+liquid mesh controls
                           'length', 8.0e-3, ...   % [m] domain size for fluid side
                           'cells',  800, ...      % uniform cells across solid+liquid
                           'min_cells', 500 ...    % guard minimum when overriding opts
                           ), ...
    'min_seed_cells', 1, ...       % initial solid thickness in cell units
    'history_dt',     2.5e-4, ...  % seconds between saved flux samples (approx.)
    'flux_smoothing', 5, ...       % odd moving-average window for q(t)
    'nsave',          4000 ...     % fallback when history_dt <= 0
    );

%% --- WALL (sapphire-like, shared) --------------------------------------
k_w   = 40;     rho_w = 3980;  c_w = 750;   % sapphire-ish

%% === CASE A: WATER/ICE ================================================
A.k_s = 2.22;     A.rho_s = 917;   A.c_s = 2100;     % ice
A.k_l = 0.6;      A.rho_l = 998;   A.c_l = 4180;     % water
A.L   = 333.5e3;                                   % J/kg
A.Tf  = 0;        A.Tw_inf = -15;   A.Tl_inf = -15; % °C

%% === CASE B: TIN (metal) ==============================================
B.k_s = 66;       B.rho_s = 7310;  B.c_s = 230;      % solid tin
B.k_l = 31;       B.rho_l = 6980;  B.c_l = 300;      % liquid tin
B.L   = 59.2e3;                                   % J/kg
B.Tf  = 231.93;   B.Tw_inf = 50;  B.Tl_inf = B.Tf - 5;

%% --- RUN BOTH CASES ----------------------------------------------------
sim_opts = struct('explicit', explicit_opts);

caseA = run_vam_case('Water/Ice + Sapphire', k_w,rho_w,c_w, A, R_c, t_phys, sim_opts);
caseB = run_vam_case('Tin (liq/sol) + Sapphire', k_w,rho_w,c_w, B, R_c, t_phys, sim_opts);

%% --- PLOTS: per case (profiles + difference + conductance + flux) -----
plot_profiles(caseA);
plot_diff_profile(caseA);
plot_conductance(caseA, R_c, 0.1);          % window you can change
plot_flux(caseA, R_c, 0.1);                  % one call per case, correct Rc

plot_profiles(caseB);
plot_diff_profile(caseB);
plot_conductance(caseB, R_c, 0.1);
plot_flux(caseB, R_c, 0.1);
