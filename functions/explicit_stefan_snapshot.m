function snap = explicit_stefan_snapshot(k_w,rho_w,c_w, M, R_c, t_end, params, opts)
%EXPLICIT_STEFAN_SNAPSHOT Explicit 1-D three-domain Stefan snapshot with Rc.
% PDEs:  ∂t T = α ∂xx T in each region
% x=0: single flux q0 through series resistance Rw + Rc + Rs
% x=S(t): Ts=Tl=Tf at the face; Stefan law with one-sided slopes.
% Far field: Tw(-Lw)=Tw_inf, Tl(Lf)=Tl_inf
% OPTS is an optional struct supporting fields:
%   CFL              - explicit stability number (default 0.30)
%   wall             - struct with wall mesh controls:
%                        length [m], dx [m], cells, extent_factor, min_length
%   fluid            - struct with solid/liquid mesh controls (same fields)
%   min_cells        - legacy fallback for both domains (default 400)
%   nodes_per_diff   - legacy heuristic for # nodes per diffusion length
%   domain_factor    - legacy fallback for extent_factor (default 5)
%   min_length       - legacy fallback for domain length (default 2e-3 m)
%   min_seed_cells   - # of solid cells to seed at the interface (default 1)
%   nsave            - max stored history points if history_dt unused (2000)
%   history_dt       - desired spacing of saved flux samples [s] (optional)
%   flux_smoothing   - odd window length for moving-average smoothing

    if nargin < 7 || isempty(params)
        params = struct();
    end

    if nargin < 8 || isempty(opts)
        opts = struct();
    end

% Unpack
k_s=M.k_s; rho_s=M.rho_s; c_s=M.c_s;
k_l=M.k_l; rho_l=M.rho_l; c_l=M.c_l;
L  =M.L;   Tf=M.Tf;       Tw_inf=M.Tw_inf; Tl_inf=M.Tl_inf;

% Diffusivities
aw = k_w/(rho_w*c_w);
as = k_s/(rho_s*c_s);
al = k_l/(rho_l*c_l);

% Extract optional VAM calibration pieces needed for seeding
has_lam = isfield(params,'lam');
has_S0e = isfield(params,'S0_e');
has_E0e = isfield(params,'E0_e');
has_t0e = isfield(params,'t0_e');
has_mu  = isfield(params,'mu');
use_vam_seed = has_lam && has_S0e && has_E0e && has_t0e && has_mu;

if use_vam_seed
    lam  = params.lam;
    Ti   = params.Ti;
    S0_e = params.S0_e;
    E0_e = params.E0_e;
    t0_e = params.t0_e;
    mu   = params.mu;
else
    Ti = Tf;
    lam = NaN; S0_e = NaN; E0_e = NaN; t0_e = 0; mu = NaN; %#ok<NASGU>
end

if isfield(params,'Tf'), Tf = params.Tf; end
if isfield(params,'Tw_inf'), Tw_inf = params.Tw_inf; end
if isfield(params,'Tl_inf'), Tl_inf = params.Tl_inf; end

% Determine a seed time so the explicit grid starts with one full solid cell.
% This avoids the "no-solid" start of the analytic solution while keeping the
% numerical domain consistent with the VAM calibration.

% Mesh/temporal controls (user-friendly fields override legacy heuristics)
nodes_per_diff = get_opt(opts, 'nodes_per_diff', []);
min_cells_legacy = get_opt(opts, 'min_cells', 400);
domain_factor = get_opt(opts, 'domain_factor', 5);
min_length_legacy = get_opt(opts, 'min_length', 2e-3);

wall_opts  = get_opt(opts, 'wall', struct());
fluid_opts = get_opt(opts, 'fluid', struct());

min_seed_cells = get_opt(opts, 'min_seed_cells', get_opt(fluid_opts, 'min_seed_cells', 1));
nsave          = get_opt(opts, 'nsave', 2000);
history_dt     = get_opt(opts, 'history_dt', 0);
flux_window    = get_opt(opts, 'flux_smoothing', 0);
if flux_window > 1
    flux_window = floor(flux_window);
    if mod(flux_window,2) == 0
        flux_window = flux_window + 1;
    end
end

% Domain extents (default to multiples of the diffusion length unless
% overridden explicitly)
wall_extent_factor = get_opt(wall_opts, 'extent_factor', domain_factor);
fluid_extent_factor = get_opt(fluid_opts, 'extent_factor', domain_factor);

wall_min_length = get_opt(wall_opts, 'min_length', min_length_legacy);
fluid_min_length = get_opt(fluid_opts, 'min_length', min_length_legacy);

wall_length_user = get_opt(wall_opts, 'length', get_opt(wall_opts, 'extent', []));
wall_length_fixed = ~isempty(wall_length_user);
if wall_length_fixed
    wall_length = max(wall_length_user, wall_min_length);
else
    wall_length = max(wall_extent_factor*sqrt(aw*t_final_phys), wall_min_length);
end

fluid_length_user = get_opt(fluid_opts, 'length', get_opt(fluid_opts, 'extent', []));
fluid_length_fixed = ~isempty(fluid_length_user);
if fluid_length_fixed
    fluid_length = max(fluid_length_user, fluid_min_length);
else
    fluid_length = max(fluid_extent_factor*sqrt(max(as,al)*t_final_phys), fluid_min_length);
end

wall_min_cells = get_opt(wall_opts, 'min_cells', min_cells_legacy);
fluid_min_cells = get_opt(fluid_opts, 'min_cells', min_cells_legacy);

wall_cells = get_opt(wall_opts, 'cells', []);
if isempty(wall_cells)
    wall_dx = get_opt(wall_opts, 'dx', []);
    if ~isempty(wall_dx) && wall_dx > 0
        wall_cells = max(3, round(wall_length / wall_dx));
    else
        wall_cells = wall_min_cells;
        if ~isempty(nodes_per_diff)
            wall_cells = max(wall_cells, ceil(nodes_per_diff * wall_extent_factor));
        end
    end
else
    wall_cells = max(3, round(wall_cells));
end
wall_cells = max(wall_cells, wall_min_cells);

fluid_cells = get_opt(fluid_opts, 'cells', []);
if isempty(fluid_cells)
    fluid_dx = get_opt(fluid_opts, 'dx', []);
    if ~isempty(fluid_dx) && fluid_dx > 0
        fluid_cells = max(3, round(fluid_length / fluid_dx));
    else
        fluid_cells = fluid_min_cells;
        if ~isempty(nodes_per_diff)
            fluid_cells = max(fluid_cells, ceil(nodes_per_diff * fluid_extent_factor));
        end
    end
else
    fluid_cells = max(3, round(fluid_cells));
end
fluid_cells = max(fluid_cells, fluid_min_cells);

Nw = wall_cells;
Nf = fluid_cells;

% Seed the numerical fields either from VAM calibration or a uniform slab.
seed_time = 0;
t_final_phys = t_end;
if use_vam_seed
    for iter = 1:5
        if wall_length_fixed
            Lw = wall_length;
        else
            Lw = max(wall_length, wall_extent_factor*sqrt(aw*t_final_phys));
        end
        if fluid_length_fixed
            Lf = fluid_length;
        else
            Lf = max(fluid_length, fluid_extent_factor*sqrt(max(as,al)*t_final_phys));
        end
        dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;  % 0^- at +dxw/2
        dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;  % 0^+ at +dxf/2

        seed_thickness = min_seed_cells * dxf;
        seed_time_new = ((seed_thickness + S0_e)^2) / (4*lam^2*as) - t0_e;
        seed_time_new = max(0, seed_time_new);
        t_final_phys = max(t_end, seed_time_new);

        if abs(seed_time_new - seed_time) < 1e-12
            seed_time = seed_time_new;
            break;
        end

        seed_time = seed_time_new;
    end

    seed_time = max(seed_time, 0);
    t_final_phys = max(t_end, seed_time);
    if ~wall_length_fixed
        wall_length = max(wall_length, wall_extent_factor*sqrt(aw*t_final_phys));
    end
    if ~fluid_length_fixed
        fluid_length = max(fluid_length, fluid_extent_factor*sqrt(max(as,al)*t_final_phys));
    end
    Lw = wall_length;
    Lf = fluid_length;
    dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;
    dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;
    seed_thickness = min_seed_cells * dxf;
    seed_time = ((seed_thickness + S0_e)^2) / (4*lam^2*as) - t0_e;
    seed_time = max(0, seed_time);
    t_final_phys = max(t_end, seed_time);

    tpe_seed = seed_time + t0_e;
    den_w_e = 2*sqrt(aw*tpe_seed);
    den_s_e = 2*sqrt(as*tpe_seed);
    den_l_e = 2*sqrt(al*tpe_seed);
    erf_lam = erf(lam);
    erf_mu  = erf(mu);

    Se_seed = 2*lam*sqrt(as*tpe_seed) - S0_e;

    Tw = Ti + (Ti - Tw_inf) .* erf( (xw - E0_e) ./ den_w_e );

    Tfld = zeros(Nf,1);
    solid_mask  = xf <= Se_seed;
    liquid_mask = ~solid_mask;
    Tfld(solid_mask) = Ti + (Tf - Ti) .* erf( (xf(solid_mask) + S0_e) ./ den_s_e ) ./ erf_lam;
    Tfld(liquid_mask) = Tl_inf + (Tf - Tl_inf) .* ...
        ( erf( (xf(liquid_mask) + S0_e) ./ den_l_e ) - 1 ) ./ (erf_mu - 1);

    S_real = min((Nf-1)*dxf, max(dxf, Se_seed));
    S_seed = S_real;

    seed_info.Se_vam = Se_seed;
else
    if wall_length_fixed
        Lw = wall_length;
    else
        Lw = max(wall_length, wall_extent_factor*sqrt(aw*max(t_final_phys,1e-16)));
    end
    if fluid_length_fixed
        Lf = fluid_length;
    else
        Lf = max(fluid_length, fluid_extent_factor*sqrt(max(as,al)*max(t_final_phys,1e-16)));
    end
    dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;
    dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;
    seed_thickness = min_seed_cells * dxf;
    S_seed = min((Nf-1)*dxf, max(dxf, seed_thickness));

    Tw = Tw_inf * ones(Nw,1);
    Tfld = Tl_inf * ones(Nf,1);
    solid_mask = xf <= S_seed;
    Tfld(solid_mask) = Tf;

    S_real = S_seed;
    seed_info.Se_vam = NaN;
end

seed_info.time   = seed_time;
seed_info.thickness = S_real;
seed_info.cell_width = dxf;

% Track the physical and relative times separately so we can report a history
% measured from t = 0 while integrating from t = seed_time.
t_phys = seed_time;
t_rel  = 0;
sim_duration = max(t_end - seed_time, 0);
seed_info.duration = sim_duration;

% Explicit time step (CFL)
CFL = get_opt(opts, 'CFL', 0.3);
dt_base = CFL * min( dxw^2/(2*aw), dxf^2/(2*max(as,al)) );
if sim_duration <= 0
    nsteps = 0;
    dt_base = 0;
else
    nsteps = max(1, ceil(sim_duration/dt_base));
    dt_base = min(dt_base, sim_duration/nsteps);
end
t  = 0;

% Precompute coefficients for the nominal dt (reuse unless final step smaller)
Rw = dxw/(2*k_w);     Rs = dxf/(2*k_s);
coeff = local_coeffs(dt_base, aw, as, al, dxw, dxf);
curr_dt = dt_base;

% history buffers (downsampled only)
if history_dt > 0
    n_hist_max = max(2, ceil(sim_duration/history_dt) + 2);
    stride = 1; %#ok<NASGU>
else
    stride = max(1, max(1, floor(max(nsteps,1)/max(nsave,1))));
    n_hist_max = max(2, ceil(max(nsteps,1)/stride) + 1);
end
t_hist = zeros(n_hist_max,1);
q_hist = zeros(n_hist_max,1);
Tw_face_hist = zeros(n_hist_max,1);
Ts_face_hist = zeros(n_hist_max,1);
ksave  = 0;
last_save_time = -inf;

% Record initial (seed) flux sample for diagnostics
q_seed = (Tw(1) - Tfld(1)) / (Rw + R_c + Rs);
Tw_face_seed = Tw(1) - Rw*q_seed;
Ts_face_seed = Tfld(1) + Rs*q_seed;
ksave = ksave + 1;
t_hist(ksave) = t_rel;
q_hist(ksave) = (Tw_face_seed - Ts_face_seed)/R_c;
Tw_face_hist(ksave) = Tw_face_seed;
Ts_face_hist(ksave) = Ts_face_seed;
last_save_time = t_rel;

for n = 1:nsteps
    if sim_duration == 0
        dt_step = 0;
    else
        dt_step = min(curr_dt, sim_duration - t);
    end
    if dt_step <= 0
        break;
    end
    if abs(dt_step - curr_dt) > eps(curr_dt)
        coeff = local_coeffs(dt_step, aw, as, al, dxw, dxf);
        curr_dt = dt_step;
    end

    % Adjacent face index from current continuous front
    m = max(1, min(Nf-1, floor(S_real/dxf)));

    % ===== x=0 contact resistance: common flux q0 through series =====
    q0 = (Tw(1) - Tfld(1)) / (Rw + R_c + Rs);   % + toward +x (into fluid)
    Tw_face = Tw(1) - Rw*q0;
    Ts_face = Tfld(1) + Rs*q0;

    % Ghosts consistent with q0
    Tghost_w = Tw(1)   - Rw*q0;   % wall face
    Tghost_f = Tfld(1) + Rs*q0;   % solid face

    % ===== WALL update (FTCS) =====
    Tw_new = Tw;
    if Nw >= 2
        Tw_new(1) = Tw(1) + coeff.wall_edge * (Tghost_w - 2*Tw(1) + Tw(2));
    end
    if Nw > 2
        iL=2; iR=Nw-1;
        Tw_new(iL:iR) = Tw(iL:iR) + coeff.wall_bulk .* ...
            ( Tw(iL+1:iR+1) - 2*Tw(iL:iR) + Tw(iL-1:iR-1) );
    end
    Tw_new(end) = Tw_inf;
    Tw = Tw_new;

    % ===== FLUID update (solid + liquid) =====
    Tn = Tfld;

    % Leftmost solid cell uses the contact ghost
    if Nf >= 2
        Tn(1) = Tfld(1) + coeff.solid_edge * ( Tfld(2) - 2*Tfld(1) + Tghost_f );
    end

    % Solid interior 2..m-1
    if m > 2
        jL=2; jR=m-1;
        Tn(jL:jR) = Tfld(jL:jR) + coeff.solid_bulk .* ...
            ( Tfld(jL+1:jR+1) - 2*Tfld(jL:jR) + Tfld(jL-1:jR-1) );
    end

    % Near-front solid cell j=m (face Dirichlet at Tf)
    if m >= 1
        if m == 1, Tm1 = Tghost_f; else, Tm1 = Tfld(m-1); end
        lap_m = coeff.near_face_coeff * ( Tm1 - 3*Tfld(m) + 2*Tf );
        Tn(m) = Tfld(m) + coeff.as_dt * lap_m;
    end

    % Near-front liquid cell j=m+1 (face Dirichlet at Tf)
    if m+1 <= Nf && m+2 <= Nf
        lap_mp1 = coeff.near_face_coeff * ( Tfld(m+2) - 3*Tfld(m+1) + 2*Tf );
        Tn(m+1) = Tfld(m+1) + coeff.al_dt * lap_mp1;
    end

    % Liquid interior m+2..Nf-1
    if m+2 <= Nf-1
        jL=m+2; jR=Nf-1;
        Tn(jL:jR) = Tfld(jL:jR) + coeff.liquid_bulk .* ...
            ( Tfld(jL+1:jR+1) - 2*Tfld(jL:jR) + Tfld(jL-1:jR-1) );
    end

    % Far-right Dirichlet
    Tn(end) = Tl_inf;

    % Commit temperatures
    Tfld = Tn;

    % ===== Stefan update (one-sided slopes) =====
    try
        ds = update_interface_from_T_gradient(Tfld, xf, m, k_s, k_l, rho_s, L, dt_step);
    catch err
        if ~strcmp(err.identifier, 'update_interface_from_T_gradient:IndexError')
            rethrow(err);
        end

        if m >= 2
            As_ = Tfld(m)   - Tf;  Bs_ = Tfld(m-1) - Tf;
            grad_s = (Bs_ - 9*As_) / (3*dxf);
        elseif m >= 1
            grad_s = coeff.two_over_dx * (Tf - Tfld(m));
        else
            grad_s = 0;
        end

        if m+2 <= Nf
            Al_ = Tfld(m+1) - Tf;  Bl_ = Tfld(m+2) - Tf;
            grad_l = coeff.grad_upwind * (9*Al_ - Bl_);
        elseif m+1 <= Nf
            grad_l = coeff.two_over_dx * (Tfld(m+1) - Tf);
        else
            grad_l = 0;
        end

        ds = dt_step * ( k_s*grad_s - k_l*grad_l ) / (rho_s*L );
    end

    S_real = S_real + ds;
    S_real = min( (Nf-1)*dxf, max( dxf, S_real ) );

    % Advance clocks after the state has been updated
    t = t + dt_step;
    t_phys = t_phys + dt_step;
    t_rel  = t_phys - seed_time;

    % Reconstruct contact flux/temps at the updated time level
    q1 = (Tw(1) - Tfld(1)) / (Rw + R_c + Rs);
    Tw_face_new = Tw(1) - Rw*q1;
    Ts_face_new = Tfld(1) + Rs*q1;
    q_contact = (Tw_face_new - Ts_face_new)/R_c;

    % Save history either on stride counts or at roughly uniform dt
    should_save = false;
    if history_dt > 0
        if (t_rel - last_save_time) >= history_dt*(1 - 1e-8) || n == nsteps
            should_save = true;
        end
    else
        if mod(n, stride)==0 || n == nsteps
            should_save = true;
        end
    end

    if should_save
        ksave = ksave + 1;
        t_hist(ksave) = t_rel;
        q_hist(ksave) = q_contact;
        Tw_face_hist(ksave) = Tw_face_new;
        Ts_face_hist(ksave) = Ts_face_new;
        last_save_time = t_rel;
    end
end

% Pack snapshot
m        = max(1, min(Nf-1, floor(S_real/dxf)));
snap.x   = [xw; xf];
snap.T   = [Tw; Tfld];
snap.S   = S_real;
snap.t   = t_end;
snap.t_rel = t_rel;
snap.t_offset = seed_time;
snap.seed = seed_info;
snap.grid = struct('dx_wall',dxw,'dx_fluid',dxf,'N_wall',Nw,'N_fluid',Nf,...
    'L_wall',Lw,'L_fluid',Lf);
snap.history = struct('history_dt',history_dt,'flux_window',flux_window,...
    'nsave',nsave);

% Trim history to what we actually saved
t_hist = t_hist(1:ksave);
q_hist = q_hist(1:ksave);
snap.q.t   = t_hist;
snap.q.val = q_hist;
snap.q.Tw_face = Tw_face_hist(1:ksave);
snap.q.Ts_face = Ts_face_hist(1:ksave);
snap.q.t_phys = t_hist + seed_time;

% Optional moving-average smoothing of the stored flux (for plotting only)
if flux_window > 1
    snap.q.val = moving_average(snap.q.val, flux_window);
end
end

function ds = update_interface_from_T_gradient(T, y, i_s, k_s, k_l, rho_s, L_latent, dt)
%UPDATE_INTERFACE_FROM_T_GRADIENT Advance interface via one-sided gradients.
    i_l = i_s + 1;
    if (i_s - 1) < 1 || (i_l + 1) > numel(T)
        error('update_interface_from_T_gradient:IndexError', ...
              'Interface too close to boundary for one-sided gradients.');
    end

    dy_s = y(i_s)   - y(i_s - 1);
    dy_l = y(i_l+1) - y(i_l);

    dTdy_s = (T(i_s)   - T(i_s - 1)) / dy_s;
    dTdy_l = (T(i_l+1) - T(i_l))     / dy_l;

    ds_dt = (k_s * dTdy_s - k_l * dTdy_l) / (rho_s * L_latent);
    ds = dt * ds_dt;
end

function coeff = local_coeffs(dt, aw, as, al, dxw, dxf)
%LOCAL_COEFFS Precompute finite-difference coefficients for efficiency.
    coeff.wall_bulk   = aw*dt/(dxw^2);
    coeff.wall_edge   = 2*coeff.wall_bulk;
    coeff.solid_bulk  = as*dt/(dxf^2);
    coeff.solid_edge  = 2*coeff.solid_bulk;
    coeff.liquid_bulk = al*dt/(dxf^2);
    coeff.near_face_coeff = 4/(3*dxf^2);
    coeff.as_dt = as*dt;
    coeff.al_dt = al*dt;
    coeff.two_over_dx = 2/dxf;
    coeff.grad_upwind = 1/(3*dxf);
end

function val = get_opt(opts, field, default)
%GET_OPT Fetch an option from a struct with a default fallback.
    if isstruct(opts) && isfield(opts, field) && ~isempty(opts.(field))
        val = opts.(field);
    else
        val = default;
    end
end

function y = moving_average(y, window)
%MOVING_AVERAGE Simple centered moving average with odd window length.
    n = numel(y);
    window = max(1, floor(window));
    if window <= 1 || n <= 2
        return;
    end
    if mod(window,2) == 0
        window = window + 1;
    end
    half = floor(window/2);
    y_sm = zeros(size(y));
    for i = 1:n
        j0 = max(1, i-half);
        j1 = min(n, i+half);
        y_sm(i) = mean(y(j0:j1));
    end
    y = y_sm;
end
