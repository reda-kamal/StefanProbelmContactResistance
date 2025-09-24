function snap = explicit_stefan_snapshot(k_w,rho_w,c_w, M, R_c, t_end)
%EXPLICIT_STEFAN_SNAPSHOT Explicit 1-D three-domain Stefan snapshot with Rc.
% PDEs:  ∂t T = α ∂xx T in each region
% x=0: single flux q0 through series resistance Rw + Rc + Rs
% x=S(t): Ts=Tl=Tf at the face; Stefan law with one-sided slopes.
% Far field: Tw(-Lw)=Tw_inf, Tl(Lf)=Tl_inf

% Unpack
k_s=M.k_s; rho_s=M.rho_s; c_s=M.c_s;
k_l=M.k_l; rho_l=M.rho_l; c_l=M.c_l;
L  =M.L;   Tf=M.Tf;       Tw_inf=M.Tw_inf; Tl_inf=M.Tl_inf;

% Diffusivities
aw = k_w/(rho_w*c_w);
as = k_s/(rho_s*c_s);
al = k_l/(rho_l*c_l);

% Semi-infinite truncations (~5 diffusion lengths at t_end)
Lw = max(5*sqrt(aw*t_end), 2e-3);
Lf = max(5*sqrt(max(as,al)*t_end), 2e-3);

% Resolution tuned by nodes-per-diffusion-length (reduces work v. fixed 2000)
nodes_per_diff = 200;
min_cells = 400;
Nw = max(ceil(nodes_per_diff*5), min_cells);
Nf = max(ceil(nodes_per_diff*5), min_cells);
dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;  % 0^- at +dxw/2
dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;  % 0^+ at +dxf/2

% ICs
Tw   = Tw_inf*ones(Nw,1);
Tfld = Tl_inf*ones(Nf,1);

% Seed a thin solid so a front exists
seed_cells = 2;
m0 = max(1, min(Nf-1, seed_cells));
S_real = m0*dxf;
Tfld(1:m0) = Tf;

% Explicit time step (CFL)
CFL = 0.3;
dt_base = CFL * min( dxw^2/(2*aw), dxf^2/(2*max(as,al)) );
nsteps = max(1, ceil(t_end/dt_base));
dt_base = min(dt_base, t_end/nsteps);
t  = 0;

% Precompute coefficients for the nominal dt (reuse unless final step smaller)
Rw = dxw/(2*k_w);     Rs = dxf/(2*k_s);
coeff = local_coeffs(dt_base, aw, as, al, dxw, dxf);
curr_dt = dt_base;

% history buffers (downsampled only)
nsave  = 2000;
stride = max(1, floor(nsteps/nsave));
t_hist = zeros(ceil(nsteps/stride),1);
q_hist = zeros(ceil(nsteps/stride),1);
ksave  = 0;

for n = 1:nsteps
    dt_step = min(curr_dt, t_end - t);
    if dt_step <= 0
        break;
    end
    if abs(dt_step - curr_dt) > eps(curr_dt)
        coeff = local_coeffs(dt_step, aw, as, al, dxw, dxf);
        curr_dt = dt_step;
    end
    t = t + dt_step;

    % Adjacent face index from current continuous front
    m = max(1, min(Nf-1, floor(S_real/dxf)));

    % ===== x=0 contact resistance: common flux q0 through series =====
    q0 = (Tw(1) - Tfld(1)) / (Rw + R_c + Rs);   % + toward +x (into fluid)

    % Save downsampled q(t)
    if mod(n, stride)==0
        ksave = ksave + 1;
        t_hist(ksave) = t;
        q_hist(ksave) = q0;
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
    S_real = min( (Nf-1)*dxf, max( 1*dxf, S_real ) );
end

% Pack snapshot
m        = max(1, min(Nf-1, floor(S_real/dxf)));
snap.x   = [xw; xf];
snap.T   = [Tw; Tfld];
snap.S   = S_real;
snap.t   = t;

% Trim history to what we actually saved
t_hist = t_hist(1:ksave);
q_hist = q_hist(1:ksave);
snap.q.t   = t_hist;
snap.q.val = q_hist;
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
