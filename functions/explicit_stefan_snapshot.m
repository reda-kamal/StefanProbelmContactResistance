function snap = explicit_stefan_snapshot(k_w,rho_w,c_w, M, R_c, t_end, params, opts)
%EXPLICIT_STEFAN_SNAPSHOT Explicit 1-D three-domain Stefan snapshot with Rc.
% PDEs:  ∂t T = α ∂xx T in each region
% x=0: single flux q0 through series resistance Rw + Rc + Rs
% x=S(t): Ts=Tl=Tf at the face; Stefan law with one-sided slopes.
% Far field: Tw(-Lw)=Tw_inf, Tl(Lf)=Tl_inf
% OPTS is an optional struct supporting fields:
%   CFL, nodes_per_diff, min_cells, domain_factor, min_seed_cells,
%   min_length, nsave

    if nargin < 7
        error('explicit_stefan_snapshot:MissingParams', ...
              'Pass the calibrated VAM parameters to seed the solver.');
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

% Extract VAM calibration pieces needed for seeding
lam  = params.lam;
Ti   = params.Ti;
S0_e = params.S0_e;
E0_e = params.E0_e;
t0_e = params.t0_e;
Tf   = params.Tf;
Tw_inf = params.Tw_inf;
Tl_inf = params.Tl_inf;
mu     = params.mu;

% Determine a seed time so the explicit grid starts with one full solid cell.
% This avoids the "no-solid" start of the analytic solution while keeping the
% numerical domain consistent with the VAM calibration.

% Resolution tuned by nodes-per-diffusion-length (reduces work v. fixed 2000)
nodes_per_diff = get_opt(opts, 'nodes_per_diff', 200);
min_cells      = get_opt(opts, 'min_cells', 400);
domain_factor  = get_opt(opts, 'domain_factor', 5);
min_seed_cells = get_opt(opts, 'min_seed_cells', 1);
min_length     = get_opt(opts, 'min_length', 2e-3);
nsave          = get_opt(opts, 'nsave', 2000);

Nw = max(ceil(nodes_per_diff*domain_factor), min_cells);
Nf = max(ceil(nodes_per_diff*domain_factor), min_cells);

% Fixed-point iteration to align the seed time (one full solid cell) and the
% truncation lengths used for the semi-infinite domains.
seed_time = 0;
t_final_phys = t_end;
for iter = 1:5
    Lw = max(domain_factor*sqrt(aw*t_final_phys), min_length);
    Lf = max(domain_factor*sqrt(max(as,al)*t_final_phys), min_length);
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
Lw = max(domain_factor*sqrt(aw*t_final_phys), min_length);
Lf = max(domain_factor*sqrt(max(as,al)*t_final_phys), min_length);
dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;
dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;
seed_thickness = min_seed_cells * dxf;
seed_time = ((seed_thickness + S0_e)^2) / (4*lam^2*as) - t0_e;
seed_time = max(0, seed_time);
t_final_phys = max(t_end, seed_time);

% Evaluate the early-time VAM solution at the seed time for ICs.
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

% Ensure the seed matches the targeted thickness; clamp interface inside grid.
S_real = min((Nf-1)*dxf, max(dxf, Se_seed));
S_seed = S_real;

seed_info.Se_vam = Se_seed;
seed_info.time   = seed_time;
seed_info.thickness = S_seed;
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
nsteps = max(1, ceil(sim_duration/dt_base));
if sim_duration == 0
    dt_base = 0;
else
    dt_base = min(dt_base, sim_duration/nsteps);
end
t  = 0;

% Precompute coefficients for the nominal dt (reuse unless final step smaller)
Rw = dxw/(2*k_w);     Rs = dxf/(2*k_s);
coeff = local_coeffs(dt_base, aw, as, al, dxw, dxf);
curr_dt = dt_base;

% history buffers (downsampled only)
stride = max(1, floor(nsteps/nsave));
t_hist = zeros(ceil(nsteps/stride),1);
q_hist = zeros(ceil(nsteps/stride),1);
Tw_face_hist = zeros(ceil(nsteps/stride),1);
Ts_face_hist = zeros(ceil(nsteps/stride),1);
ksave  = 0;

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
    t = t + dt_step;
    t_phys = t_phys + dt_step;
    t_rel  = t_phys - seed_time;

    % Adjacent face index from current continuous front
    m = max(1, min(Nf-1, floor(S_real/dxf)));

    % ===== x=0 contact resistance: common flux q0 through series =====
    q0 = (Tw(1) - Tfld(1)) / (Rw + R_c + Rs);   % + toward +x (into fluid)
    Tw_face = Tw(1) - Rw*q0;
    Ts_face = Tfld(1) + Rs*q0;

    % Save downsampled q(t)
    if mod(n, stride)==0
        ksave = ksave + 1;
        t_hist(ksave) = t_rel;
        q_hist(ksave) = (Tw_face - Ts_face)/R_c;
        Tw_face_hist(ksave) = Tw_face;
        Ts_face_hist(ksave) = Ts_face;
    end

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

    S_real = S_real + dt_step * ( k_s*grad_s - k_l*grad_l ) / (rho_s*L);
    S_real = min( (Nf-1)*dxf, max( dxf, S_real ) );
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

% Trim history to what we actually saved
t_hist = t_hist(1:ksave);
q_hist = q_hist(1:ksave);
snap.q.t   = t_hist;
snap.q.val = q_hist;
snap.q.Tw_face = Tw_face_hist(1:ksave);
snap.q.Ts_face = Ts_face_hist(1:ksave);
snap.q.t_phys = t_hist + seed_time;
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
