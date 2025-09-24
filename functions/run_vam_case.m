function out = run_vam_case(label, k_w, rho_w, c_w, M, R_c, t_phys)
%RUN_VAM_CASE Solve VAM calibrations and explicit reference for a material case.
%
% OUT = RUN_VAM_CASE(LABEL, k_w, rho_w, c_w, M, R_c, t_phys) computes both
% the early- and late-time VAM profiles along with an explicit finite-
% difference snapshot for a single material case. Material properties are
% provided via struct M with fields k_s, rho_s, c_s, k_l, rho_l, c_l, L,
% Tf, Tw_inf, and Tl_inf. The function returns a struct containing the
% calibrated parameters, spatial profiles, and explicit solver snapshot.

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
