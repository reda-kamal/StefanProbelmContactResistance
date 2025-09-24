% Author: reda-kamal
%% Three-domain Stefan problem with finite contact resistance
%  VAM calibrations (early, late) vs explicit constant-Rc solver
%  Plots per case: profiles, profile difference, heff(t), interface flux q(t)
clear; close all; clc;

%% --- USER CONTROLS -----------------------------------------------------
t_phys = 0.1;                 % representative time for temperature profiles [s]
R_c    = 2e-5;                % contact resistance [m^2 K/W] (shared across cases)

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
caseA = run_vam_case('Water/Ice + Sapphire', k_w,rho_w,c_w, A, R_c, t_phys);
caseB = run_vam_case('Tin (liq/sol) + Sapphire', k_w,rho_w,c_w, B, R_c, t_phys);

%% --- PLOTS: per case (profiles + difference + conductance + flux) -----
plot_profiles(caseA);
plot_diff_profile(caseA);
plot_conductance(caseA, R_c, 0.1);          % window you can change
plot_flux(caseA, R_c, 0.1);                  % one call per case, correct Rc

plot_profiles(caseB);
plot_diff_profile(caseB);
plot_conductance(caseB, R_c, 0.1);
plot_flux(caseB, R_c, 0.1);

%% ====================== FUNCTIONS =====================================
function out = run_vam_case(label, k_w,rho_w,c_w, M, R_c, t_phys)
    % Unpack material M (solid/liquid domain)
    k_s = M.k_s; rho_s = M.rho_s; c_s = M.c_s;
    k_l = M.k_l; rho_l = M.rho_l; c_l = M.c_l;
    L   = M.L;   Tf    = M.Tf;    Tw_inf = M.Tw_inf;  Tl_inf = M.Tl_inf;

    % Diffusivities
    alpha_w = k_w/(rho_w*c_w);
    alpha_s = k_s/(rho_s*c_s);
    alpha_l = k_l/(rho_l*c_l);

    % Solve for lambda, Ti (shared by both VAMs within this case)
    f = @(x) stefan_eqs(x,alpha_w,alpha_s,alpha_l,...
                        k_w,k_s,k_l,L,rho_s, Tw_inf,Tl_inf,Tf);
    opts = optimoptions('fsolve','Display','off');
    x_guess = [0.1,  0.5*(Tf + Tw_inf)];
    [sol,~,flag] = fsolve(f,x_guess,opts);
    if flag<=0, error('FSOLVE did not converge for case: %s', label); end

    lam = sol(1);
    Ti  = sol(2);
    mu  = lam*sqrt(alpha_s/alpha_l);
    h_c = 1/R_c;

    % Split hc onto each side
    hcw = h_c*(Tf - Tw_inf)/(Ti - Tw_inf);
    hcs = h_c*(Tf - Tw_inf)/(Tf - Ti);

    % ---- EARLY-TIME calibration
    S0_e = 2*lam*k_s/(hcs*sqrt(pi))*exp(-lam^2)/erf(lam);
    E0_e = (S0_e/lam)*sqrt(alpha_w/alpha_s)* ...
           sqrt( log( (2*lam*k_w*sqrt(alpha_s)) / (hcw*sqrt(pi*alpha_w)*S0_e) ) );
    t0_e = S0_e^2 /(4*alpha_s*lam^2);

    % ---- LATE-TIME calibration
    S0_l = k_s/hcs;
    E0_l = k_w/hcw;
    t0_l = S0_l^2 /(4*alpha_s*lam^2);

    % Physical fronts at t_phys
    tpe = t_phys + t0_e;
    tpl = t_phys + t0_l;
    Se  = 2*lam*sqrt(alpha_s*tpe) - S0_e;
    Sl  = 2*lam*sqrt(alpha_s*tpl) - S0_l;

    % ===== Effective-conductance sandwich check (at t_phys) =====
    gfun = @(chi) (2*chi.*exp(-chi.^2))./(sqrt(pi)*erf(chi));
    phi_s  = S0_e/(2*sqrt(alpha_s*tpe));
    phi_w  = E0_e/(2*sqrt(alpha_w*tpe));
    hs_e   = (k_s/S0_e) * gfun(phi_s);
    hw_e   = (k_w/E0_e) * gfun(phi_w);
    he_e   = 1/(1/hw_e + 1/hs_e);

    phi_sL = S0_l/(2*sqrt(alpha_s*tpl));
    phi_wL = E0_l/(2*sqrt(alpha_w*tpl));
    hs_l   = (k_s/S0_l) * gfun(phi_sL);
    hw_l   = (k_w/E0_l) * gfun(phi_wL);
    he_l   = 1/(1/hw_l + 1/hs_l);

    assert( he_e + 1e-9 >= h_c && h_c + 1e-9 >= he_l, 'heff sandwich violated');

    % ---- x-mesh for profile plots ----
    Lw_e = 5*sqrt(alpha_w*tpe);  Ll_e = 5*sqrt(alpha_l*tpe);
    Lw_l = 5*sqrt(alpha_w*tpl);  Ll_l = 5*sqrt(alpha_l*tpl);
    x_min = -max(Lw_e,Lw_l); x_max = max([Se,Sl]) + max(Ll_e,Ll_l);
    knots = sort([x_min, 0, Se, Sl, x_max]);
    pts_per_seg = 1000;
    x = [];
    for j = 1:(numel(knots)-1)
        a = knots(j); b = knots(j+1);
        seg = linspace(a,b,pts_per_seg);
        if j < (numel(knots)-1), seg = seg(1:end-1); end
        x = [x, seg]; %#ok<AGROW>
    end

    % Evaluate temperatures for both VAMs at t_phys (physical space/time)
    erf_safe = @(z) erf(z);

    % Early-time
    den_w_e = 2*sqrt(alpha_w*tpe);
    den_s_e = 2*sqrt(alpha_s*tpe);
    den_l_e = 2*sqrt(alpha_l*tpe);
    Te = nan(size(x));
    Iw = (x <= 0);     Is = (x > 0) & (x <= Se);   Il = (x > Se);
    Te(Iw) = Ti + (Ti - Tw_inf).* erf_safe( (x(Iw) - E0_e)./den_w_e );
    Te(Is) = Ti + (Tf - Ti)     .* erf_safe( (x(Is) + S0_e)./den_s_e ) ./ erf_safe(lam);
    Te(Il) = Tl_inf + (Tf - Tl_inf).* ...
              ( erf_safe( (x(Il) + S0_e)./den_l_e ) - 1 ) ./ (erf_safe(mu) - 1);

    % Late-time
    den_w_l = 2*sqrt(alpha_w*tpl);
    den_s_l = 2*sqrt(alpha_s*tpl);
    den_l_l = 2*sqrt(alpha_l*tpl);
    Tl = nan(size(x));
    Iw  = (x <= 0);    IsL = (x > 0) & (x <= Sl);  IlL = (x > Sl);
    Tl(Iw)  = Ti + (Ti - Tw_inf).* erf_safe( (x(Iw)  - E0_l)./den_w_l );
    Tl(IsL) = Ti + (Tf - Ti)     .* erf_safe( (x(IsL) + S0_l)./den_s_l ) ./ erf_safe(lam);
    Tl(IlL) = Tl_inf + (Tf - Tl_inf).* ...
               ( erf_safe( (x(IlL) + S0_l)./den_l_l ) - 1 ) ./ (erf_safe(mu) - 1);

    % Temperature-profile difference (late - early)
    Tdiff = Tl - Te;

    % >>> run explicit solver up to t_phys (stores q(t) history)
    snap = explicit_stefan_snapshot(k_w, rho_w, c_w, M, R_c, t_phys);

    % Pack outputs
    out.label  = label;
    out.params = struct('lam',lam,'mu',mu,'Ti',Ti,'Tf',Tf,'Tw_inf',Tw_inf,'Tl_inf',Tl_inf, ...
        'alpha_w',alpha_w,'alpha_s',alpha_s,'alpha_l',alpha_l, ...
        'rho_w',rho_w,'rho_s',rho_s,'rho_l',rho_l, ...
        'c_w',c_w,'c_s',c_s,'c_l',c_l, ...
        'k_w',k_w,'k_s',k_s,'k_l',k_l, ...
        'S0_e',S0_e,'E0_e',E0_e,'t0_e',t0_e,'S0_l',S0_l,'E0_l',E0_l,'t0_l',t0_l, ...
        'Se',Se,'Sl',Sl);
    out.x     = x;
    out.Te    = Te;
    out.Tl    = Tl;
    out.Tdiff = Tdiff;
    out.num   = snap;
end

function plot_profiles(caseX)
    x = caseX.x; Te = caseX.Te; Tl = caseX.Tl;
    Se = caseX.params.Se; Sl = caseX.params.Sl;
    Tw = caseX.params.Tw_inf; Tf = caseX.params.Tf; Tl_inf = caseX.params.Tl_inf;

    figure('Name',['Profiles @ t_{phys} — ', caseX.label]); hold on; box on; grid on;
    plot(x, Te, 'k--','LineWidth',1.6, 'DisplayName','VAM (0-time)');
    plot(x, Tl, 'b-' ,'LineWidth',1.7, 'DisplayName','VAM (\infty-time)');
    xline(0,  'k:' ,'LineWidth',1.0, 'DisplayName','Wall–solid');
    xline(Se, 'k--','LineWidth',1.0, 'DisplayName','S^{(0)}');
    xline(Sl, 'b--','LineWidth',1.0, 'DisplayName','S^{(\infty)}');

    if isfield(caseX,'num') && ~isempty(caseX.num)
        xn = caseX.num.x;  Tn = caseX.num.T;  Sn = caseX.num.S;
        plot(xn, Tn, '.', 'MarkerSize', 6, 'DisplayName','Explicit numeric');
        xline(Sn, 'm--','LineWidth',1.2, 'DisplayName','S^{num}');
    end

    xlabel('Physical coordinate  x  [m]');
    ylabel('Temperature  [^{\circ}C]');
    title(['Two VAM vs explicit @ t = t_{phys} — ', caseX.label]);
    legend('Location','SouthEast');
    ylim([min([Tw Tf Tl_inf])-5, max([Tw Tf Tl_inf])+5]);
    xlim([min(x), max(x)]);
end

function plot_diff_profile(caseX)
    x = caseX.x; Tdiff = caseX.Tdiff;
    Se = caseX.params.Se; Sl = caseX.params.Sl;

    figure('Name',['Profile difference @ t_{phys} — ', caseX.label]); hold on; box on; grid on;
    plot(x, Tdiff, 'm-','LineWidth',1.7);
    yline(0,'k:'); xline(0,'k:','LineWidth',1.0);
    xline(Se,'k--','LineWidth',1.0); xline(Sl,'b--','LineWidth',1.0);
    xlabel('Physical coordinate  x  [m]');
    ylabel('\Delta T(x) = T^{(\infty)} - T^{(0)}  [^{\circ}C]');
    title(['Difference of VAM profiles @ t = t_{phys} — ', caseX.label]);
    legend({'\Delta T(x)','0','Wall–solid','S^{(0)}','S^{(\infty)}'}, 'Location','SouthEast');
    xlim([min(x), max(x)]);
end

function F = stefan_eqs(x,aw,as,al,kw,ks,kl,L,rhos,Tw_inf,Tl_inf,Tf)
    lam = x(1); Ti  = x(2);
    mu  = lam*sqrt(as/al);
    F1 = kw*sqrt(as)*(Ti - Tw_inf) ...
       - ks*sqrt(aw)/erf(lam)*(Tf - Ti);
    termL = kl*(Tf - Tl_inf)/(lam*sqrt(pi)) * ( - mu / erfcx(mu) );
    termS = ks*(Tf - Ti)/sqrt(pi) * exp(-lam^2)/erf(lam);
    F2 = rhos*L*as*lam + termL - termS;
    F = [F1; F2];
end

function snap = explicit_stefan_snapshot(k_w,rho_w,c_w, M, R_c, t_end)
% Explicit 1-D three-domain Stefan snapshot with finite contact resistance.
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

% Grids (cell-centred; first nodes at ±Δx/2)
Nw = 2000; dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;  % 0^- at +dxw/2
Nf = 2000; dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;  % 0^+ at +dxf/2

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
dt  = CFL * min( dxw^2/(2*aw), dxf^2/(2*max(as,al)) );
nsteps = max(1, ceil(t_end/dt));
dt = min(dt, t_end/nsteps);
t  = 0;

% history buffers (downsampled only)
nsave  = 2000;
stride = max(1, floor(nsteps/nsave));
t_hist = zeros(ceil(nsteps/stride),1);
q_hist = zeros(ceil(nsteps/stride),1);
ksave  = 0;

for n = 1:nsteps
    t = t + dt;

    % Adjacent face index from current continuous front
    m = max(1, min(Nf-1, floor(S_real/dxf)));

    % ===== x=0 contact resistance: common flux q0 through series =====
    Rw = dxw/(2*k_w);     Rs = dxf/(2*k_s);
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
        Tw_new(1) = Tw(1) + aw*dt/(dxw^2) * 2*(Tghost_w - 2*Tw(1) + Tw(2));
    end
    if Nw > 2
        iL=2; iR=Nw-1;
        Tw_new(iL:iR) = Tw(iL:iR) + aw*dt/(dxw^2) .* ...
            ( Tw(iL+1:iR+1) - 2*Tw(iL:iR) + Tw(iL-1:iR-1) );
    end
    Tw_new(end) = Tw_inf;
    Tw = Tw_new;

    % ===== FLUID update (solid + liquid) =====
    Tn = Tfld;

    % Leftmost solid cell uses the contact ghost
    if Nf >= 2
        Tn(1) = Tfld(1) + as*dt/(dxf^2) * 2*( Tfld(2) - 2*Tfld(1) + Tghost_f );
    end

    % Solid interior 2..m-1
    if m > 2
        jL=2; jR=m-1;
        Tn(jL:jR) = Tfld(jL:jR) + as*dt/(dxf^2) .* ...
            ( Tfld(jL+1:jR+1) - 2*Tfld(jL:jR) + Tfld(jL-1:jR-1) );
    end

    % Near-front solid cell j=m (face Dirichlet at Tf)
    if m >= 1
        if m == 1, Tm1 = Tghost_f; else, Tm1 = Tfld(m-1); end
        lap_m = (4/(3*dxf^2))*( Tm1 - 3*Tfld(m) + 2*Tf );
        Tn(m) = Tfld(m) + as*dt*lap_m;
    end

    % Near-front liquid cell j=m+1 (face Dirichlet at Tf)
    if m+1 <= Nf && m+2 <= Nf
        lap_mp1 = (4/(3*dxf^2))*( Tfld(m+2) - 3*Tfld(m+1) + 2*Tf );
        Tn(m+1) = Tfld(m+1) + al*dt*lap_mp1;
    end

    % Liquid interior m+2..Nf-1
    if m+2 <= Nf-1
        jL=m+2; jR=Nf-1;
        Tn(jL:jR) = Tfld(jL:jR) + al*dt/(dxf^2) .* ...
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
        grad_s = (2/dxf)*(Tf - Tfld(m));
    else
        grad_s = 0;
    end

    if m+2 <= Nf
        Al_ = Tfld(m+1) - Tf;  Bl_ = Tfld(m+2) - Tf;
        grad_l = (9*Al_ - Bl_) / (3*dxf);
    elseif m+1 <= Nf
        grad_l = (2/dxf)*(Tfld(m+1) - Tf);
    else
        grad_l = 0;
    end

    S_real = S_real + dt * ( k_s*grad_s - k_l*grad_l ) / (rho_s*L);
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

function plot_conductance(caseX, R_c, t_max)
    if nargin < 3 || isempty(t_max), t_max = 0.1; end
    p  = caseX.params;   hc = 1./R_c;

    alpha_w = p.alpha_w;  alpha_s = p.alpha_s;
    S0_e = p.S0_e;  E0_e = p.E0_e;  t0_e = p.t0_e;
    S0_l = p.S0_l;  E0_l = p.E0_l;  t0_l = p.t0_l;
    kw = p.k_w; ks = p.k_s;
    gfun = @(chi) (2.*chi.*exp(-chi.^2)) ./ (sqrt(pi).*erf(chi));
    t  = linspace(0, t_max, 1000);

    tpe   = t + t0_e;
    phi_s = S0_e ./ (2*sqrt(alpha_s*tpe));
    phi_w = E0_e ./ (2*sqrt(alpha_w*tpe));
    hs_e  = (ks./S0_e) .* gfun(phi_s);
    hw_e  = (kw./E0_e) .* gfun(phi_w);
    he_e  = 1 ./ (1./hw_e + 1./hs_e); he_e(1) = hc;

    tpl    = t + t0_l;
    phi_sL = S0_l ./ (2*sqrt(alpha_s*tpl));
    phi_wL = E0_l ./ (2*sqrt(alpha_w*tpl));
    hs_l   = (ks./S0_l) .* gfun(phi_sL);
    hw_l   = (kw./E0_l) .* gfun(phi_wL);
    he_l   = 1 ./ (1./hw_l + 1./hs_l);

    figure('Name',['h_{eff}(t) — ', caseX.label]); hold on; grid on; box on;
    plot(t, he_e, 'k--','LineWidth',1.6, 'DisplayName','VAM (t=0 calibration)');
    plot(t, he_l, 'b-' ,'LineWidth',1.7, 'DisplayName','VAM (t=\infty calibration)');
    yline(hc, 'r-.','LineWidth',1.4, 'DisplayName','true h_c (=1/R_c)');
    plot(0, hc, 'ko', 'MarkerFaceColor','k', 'DisplayName','he^{(0)}(0)=h_c');
    xlabel('time t [s]');
    ylabel('effective conductance h_{eff}(t) [W m^{-2} K^{-1}]');
    title(['Effective contact conductance vs time — ', caseX.label]);
    legend('Location','SouthEast');
    xlim([0, t_max]);
end

function plot_flux(caseX, R_c, t_max)
% Plot q(t) at x=0 using the contact law for all three: early-VAM, late-VAM, explicit
    if nargin<3 || isempty(t_max), t_max = max(0.1, caseX.num.t); end
    Nt = 600;  t = linspace(0,t_max,Nt);
    p = caseX.params;

    % VAM contact-law fluxes q = (Tw(0^-)-Ts(0^+))/Rc
    [~,~,q0_rc] = vam_face_temps_and_q(p,'early', t, R_c);
    [~,~,qI_rc] = vam_face_temps_and_q(p,'late',  t, R_c);

    % Explicit flux history (same contact law: ghosts enforce Tw_face - Ts_face = Rc*q)
    th = caseX.num.q.t;  qh = caseX.num.q.val;

    figure('Name',['Interface flux vs time — ',caseX.label]); hold on; grid on; box on;
    plot(t, q0_rc, 'k--','LineWidth',1.6, 'DisplayName','VAM^{(0)}: q=\Delta T/R_c');
    plot(t, qI_rc, 'b-' ,'LineWidth',1.7, 'DisplayName','VAM^{(\infty)}: q=\Delta T/R_c');
    plot(th, qh,  '-',  'LineWidth',1.6, 'Color',[0.95 0.65 0.2], 'DisplayName','Explicit (const R_c)');
    xlabel('time t [s]');
    ylabel('interface heat flux q(0,t) [W m^{-2}]');
    title(['Interface flux vs time — ', caseX.label]);
    legend('Location','SouthEast');
    xlim([0, t_max]);
end

function [Tw0m,Ts0p,q_rc] = vam_face_temps_and_q(p, which, t, R_c)
% Face temperatures from VAM (physical time), then flux by contact law q = ΔT/Rc
    alpha_w = p.alpha_w; alpha_s = p.alpha_s;
    lam = p.lam; Tf = p.Tf; Ti = p.Ti; Tw_inf = p.Tw_inf;
    if strcmp(which,'early')
        t0 = p.t0_e; S0 = p.S0_e; E0 = p.E0_e;
    else
        t0 = p.t0_l; S0 = p.S0_l; E0 = p.E0_l;
    end
    den_w = 2*sqrt(alpha_w*(t+t0));
    den_s = 2*sqrt(alpha_s*(t+t0));
    Tw0m = Ti + (Ti - Tw_inf) .* erf( (-E0)./den_w );
    Ts0p = Ti + (Tf - Ti)     .* erf( (+S0)./den_s ) ./ erf(lam);
    q_rc = (Tw0m - Ts0p) ./ R_c;
end
