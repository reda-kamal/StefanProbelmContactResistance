function out = run_vam_case(label, k_w, rho_w, c_w, M, R_c, t_phys, opts)
%RUN_VAM_CASE Solve VAM calibrations and explicit reference for a material case.
%
% OUT = RUN_VAM_CASE(LABEL, k_w, rho_w, c_w, M, R_c, t_phys, OPTS) computes both
% the early- and late-time VAM profiles along with an explicit finite-
% difference snapshot for a single material case. Material properties are
% provided via struct M with fields k_s, rho_s, c_s, k_l, rho_l, c_l, L,
% Tf, Tw_inf, and Tl_inf. The function returns a struct containing the
% calibrated parameters, spatial profiles, and explicit solver snapshot.
% OPTS is an optional struct with fields:
%   profile_pts_per_seg   - points per plotting segment (default 400)
%   profile_extent_factor - domain multiple of sqrt(alpha*t) (default 5)
%   explicit              - struct forwarded to explicit snapshots.
%                           May include a nested 'refine' struct with fields
%                           max_iters, factor, cfl_shrink, tol_abs_T, tol_rel_T,
%                           tol_abs_q, tol_rel_q, history_shrink, and min_CFL
%                           to control the automatic VAM-bounded refinement.

    if nargin < 8 || isempty(opts)
        opts = struct();
    end

    profile_pts_per_seg = get_opt(opts, 'profile_pts_per_seg', 400);
    profile_extent_factor = get_opt(opts, 'profile_extent_factor', 5);
    explicit_opts = get_opt(opts, 'explicit', struct());

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
    Lw_e = profile_extent_factor*sqrt(alpha_w*tpe);  Ll_e = profile_extent_factor*sqrt(alpha_l*tpe);
    Lw_l = profile_extent_factor*sqrt(alpha_w*tpl);  Ll_l = profile_extent_factor*sqrt(alpha_l*tpl);
    x_min = -max(Lw_e,Lw_l); x_max = max([Se,Sl]) + max(Ll_e,Ll_l);
    knots = sort([x_min, 0, Se, Sl, x_max]);
    pts_per_seg = profile_pts_per_seg;
    nseg = numel(knots) - 1;
    npts = nseg*pts_per_seg - (nseg - 1);
    x = zeros(1, npts);
    idx = 1;
    for j = 1:nseg
        seg = linspace(knots(j),knots(j+1),pts_per_seg);
        if j < nseg
            seg = seg(1:end-1);
        end
        seg_len = numel(seg);
        x(idx:idx+seg_len-1) = seg;
        idx = idx + seg_len;
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
    erf_lam = erf_safe(lam);
    Te(Is) = Ti + (Tf - Ti)     .* erf_safe( (x(Is) + S0_e)./den_s_e ) ./ erf_lam;
    erf_mu = erf_safe(mu);
    Te(Il) = Tl_inf + (Tf - Tl_inf).* ...
              ( erf_safe( (x(Il) + S0_e)./den_l_e ) - 1 ) ./ (erf_mu - 1);

    % Late-time
    den_w_l = 2*sqrt(alpha_w*tpl);
    den_s_l = 2*sqrt(alpha_s*tpl);
    den_l_l = 2*sqrt(alpha_l*tpl);
    Tl = nan(size(x));
    Iw  = (x <= 0);    IsL = (x > 0) & (x <= Sl);  IlL = (x > Sl);
    Tl(Iw)  = Ti + (Ti - Tw_inf).* erf_safe( (x(Iw)  - E0_l)./den_w_l );
    Tl(IsL) = Ti + (Tf - Ti)     .* erf_safe( (x(IsL) + S0_l)./den_s_l ) ./ erf_lam;
    Tl(IlL) = Tl_inf + (Tf - Tl_inf).* ...
               ( erf_safe( (x(IlL) + S0_l)./den_l_l ) - 1 ) ./ (erf_mu - 1);

    % Temperature-profile difference (late - early)
    Tdiff = Tl - Te;

    params_struct = struct('lam',lam,'mu',mu,'Ti',Ti,'Tf',Tf,'Tw_inf',Tw_inf, ...
        'Tl_inf',Tl_inf,'S0_e',S0_e,'E0_e',E0_e,'t0_e',t0_e, ...
        'S0_l',S0_l,'E0_l',E0_l,'t0_l',t0_l, ...
        'alpha_w',alpha_w,'alpha_s',alpha_s,'alpha_l',alpha_l, ...
        'Se',Se,'Sl',Sl);

    % >>> run numerical solvers up to t_phys (stores q(t) history)
    [snap_explicit, meta_explicit] = run_refined_snapshot(@explicit_stefan_snapshot, ...
        'explicit', explicit_opts, k_w, rho_w, c_w, M, R_c, t_phys, ...
        params_struct, x, Te, Tl);
    snap_explicit.meta = meta_explicit;

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
    out.num   = snap_explicit;
    out.diagnostics = meta_explicit;
end

function val = get_opt(opts, field, default)
%GET_OPT Fetch an option from a struct with a default fallback.
    if isstruct(opts) && isfield(opts, field) && ~isempty(opts.(field))
        val = opts.(field);
    else
        val = default;
    end
end

function [snap, meta] = run_refined_snapshot(solver_fn, method_key, base_opts, ...
        k_w, rho_w, c_w, M, R_c, t_phys, params_struct, x_ref, Te_ref, Tl_ref)
%RUN_REFINED_SNAPSHOT Execute solver with optional adaptive refinement.

    refine_cfg = get_opt(base_opts, 'refine', struct());
    cfg.max_iters   = get_opt(refine_cfg, 'max_iters',   3);
    cfg.factor      = get_opt(refine_cfg, 'factor',      1.5);
    cfg.cfl_shrink  = get_opt(refine_cfg, 'cfl_shrink',  0.75);
    cfg.tol_abs_T   = get_opt(refine_cfg, 'tol_abs_T',   0.15);
    cfg.tol_rel_T   = get_opt(refine_cfg, 'tol_rel_T',   0.01);
    cfg.tol_abs_q   = get_opt(refine_cfg, 'tol_abs_q',   200);
    cfg.tol_rel_q   = get_opt(refine_cfg, 'tol_rel_q',   0.01);
    cfg.history_shrink = get_opt(refine_cfg, 'history_shrink', 0.75);
    cfg.min_CFL     = get_opt(refine_cfg, 'min_CFL',     0.05);

    curr_opts = base_opts;
    adjustments = struct('iter',{},'CFL',{},'wall_cells',{},'fluid_cells',{});
    ok = false;
    bounds_diag = struct();
    diag_history = cell(max(1,cfg.max_iters),1);

    for iter = 1:max(1,cfg.max_iters)
        snap = solver_fn(k_w, rho_w, c_w, M, R_c, t_phys, params_struct, curr_opts);
        [ok, bounds_diag] = check_snapshot_bounds(snap, x_ref, Te_ref, Tl_ref, ...
            params_struct, R_c, t_phys, cfg);
        bounds_diag.iteration = iter;
        diag_history{iter} = bounds_diag;
        if ok
            break;
        end

        if iter == cfg.max_iters
            break;
        end

        [curr_opts, adj] = refine_options(curr_opts, cfg);
        adj.iter = iter + 1;
        adjustments(end+1) = adj; %#ok<AGROW>
    end

    meta = struct();
    meta.method = method_key;
    if isfield(bounds_diag, 'iteration')
        iter_count = bounds_diag.iteration;
    else
        iter_count = 1;
    end
    meta.refinement = struct('iterations', iter_count, ...
                             'success', ok, ...
                             'max_iters', cfg.max_iters, ...
                             'adjustments', adjustments);
    meta.bounds = bounds_diag;
    meta.bounds_history = diag_history(1:iter_count);
    meta.options = curr_opts;
    meta.initial_options = base_opts;
    meta.refine_cfg = cfg;
end

function [ok, diag] = check_snapshot_bounds(snap, x_ref, Te_ref, Tl_ref, params, R_c, t_phys, cfg)
%CHECK_SNAPSHOT_BOUNDS Compare numeric profiles/flux with VAM sandwich.

    diag = struct();

    % --- Temperature profile bounds ---
    if isfield(snap, 'x') && isfield(snap, 'T')
        Te_interp = interp1(x_ref, Te_ref, snap.x, 'linear', 'extrap');
        Tl_interp = interp1(x_ref, Tl_ref, snap.x, 'linear', 'extrap');
        lower_env = min(Te_interp, Tl_interp);
        upper_env = max(Te_interp, Tl_interp);
        Tvals = snap.T(:);
        upper_env = upper_env(:);
        lower_env = lower_env(:);
        env_span  = max(upper_env - lower_env, [], 'omitnan');
        if isempty(env_span) || isnan(env_span)
            env_span = 0;
        end
        over_err  = Tvals - upper_env;
        under_err = lower_env - Tvals;
        max_over  = max([0; over_err(:)], [], 'omitnan');
        max_under = max([0; under_err(:)], [], 'omitnan');
        profile_violation = max(max_over, max_under);
        tol_profile = max(cfg.tol_abs_T, cfg.tol_rel_T * max(env_span, 1e-6));
        ok_profile = profile_violation <= tol_profile + 1e-12;
        [~, idx_over] = max(over_err);
        [~, idx_under] = max(under_err);
        diag.profile = struct('max_over', max_over, 'max_under', max_under, ...
            'max_violation', profile_violation, 'tol', tol_profile, ...
            'index_over', idx_over, 'index_under', idx_under, ...
            'ok', ok_profile);
    else
        ok_profile = true;
        diag.profile = struct('max_over',0,'max_under',0,'max_violation',0,'tol',0,'ok',true);
    end

    % --- Flux history bounds ---
    if isfield(snap, 'q') && isfield(snap.q, 'val')
        q_vals = snap.q.val(:);
        if isempty(q_vals)
            ok_flux = true;
            diag.flux = struct('max_over',0,'max_under',0,'max_violation',0,'tol',0,'ok',true);
        else
            if isfield(snap.q, 't_phys')
                t_hist = snap.q.t_phys(:)';
            elseif isfield(snap.q, 't')
                if isfield(snap, 't_offset')
                    t_hist = snap.q.t(:)' + snap.t_offset;
                else
                    t_hist = snap.q.t(:)';
                end
            else
                t_hist = linspace(0, t_phys, numel(q_vals));
            end
            [~,~,q_early] = vam_face_temps_and_q(params, 'early', t_hist, R_c);
            [~,~,q_late]  = vam_face_temps_and_q(params, 'late',  t_hist, R_c);
            lower_q = min(q_early, q_late);
            upper_q = max(q_early, q_late);
            lower_q = lower_q(:);
            upper_q = upper_q(:);
            span_q  = max(upper_q - lower_q, [], 'omitnan');
            if isempty(span_q) || isnan(span_q)
                span_q = 0;
            end
            q_over  = q_vals - upper_q;
            q_under = lower_q - q_vals;
            max_over_q  = max([0; q_over], [], 'omitnan');
            max_under_q = max([0; q_under], [], 'omitnan');
            flux_violation = max(max_over_q, max_under_q);
            tol_flux = max(cfg.tol_abs_q, cfg.tol_rel_q * max(span_q, 1e-6));
            ok_flux = flux_violation <= tol_flux + 1e-6;
            [~, idx_over_q] = max(q_over);
            [~, idx_under_q] = max(q_under);
            diag.flux = struct('max_over',max_over_q,'max_under',max_under_q, ...
                'max_violation',flux_violation,'tol',tol_flux, ...
                'index_over',idx_over_q,'index_under',idx_under_q, ...
                'ok', ok_flux);
        end
    else
        ok_flux = true;
        diag.flux = struct('max_over',0,'max_under',0,'max_violation',0,'tol',0,'ok',true);
    end

    ok = ok_profile && ok_flux;
    diag.ok_profile = ok_profile;
    diag.ok_flux = ok_flux;
    diag.ok = ok;
end

function [next_opts, adj] = refine_options(curr_opts, cfg)
%REFINE_OPTIONS Increase spatial resolution / reduce CFL for retry.

    next_opts = curr_opts;

    if isfield(next_opts, 'CFL') && next_opts.CFL > cfg.min_CFL
        next_opts.CFL = max(cfg.min_CFL, next_opts.CFL * cfg.cfl_shrink);
    end

    if isfield(next_opts, 'history_dt') && next_opts.history_dt > 0
        next_opts.history_dt = next_opts.history_dt * cfg.history_shrink;
    end

    next_opts.wall  = upscale_cells(get_opt_struct(next_opts, 'wall'),  cfg.factor);
    next_opts.fluid = upscale_cells(get_opt_struct(next_opts, 'fluid'), cfg.factor);

    adj = struct('CFL', get_opt(next_opts,'CFL',[]), ...
        'wall_cells', get_opt(next_opts.wall,'cells',[]), ...
        'fluid_cells', get_opt(next_opts.fluid,'cells',[]));
end

function s = get_opt_struct(opts, field)
    if isstruct(opts) && isfield(opts, field) && ~isempty(opts.(field))
        s = opts.(field);
    else
        s = struct();
    end
end

function sub = upscale_cells(sub, factor)
%UPSCALE_CELLS Ensure cell count increases by refinement factor.

    if ~isstruct(sub)
        sub = struct();
    end
    curr_cells = get_opt(sub, 'cells', []);
    if isempty(curr_cells)
        curr_cells = get_opt(sub, 'min_cells', []);
        if isempty(curr_cells)
            curr_cells = 400;
        end
    end
    new_cells = max(curr_cells + 2, ceil(curr_cells * factor));
    sub.cells = new_cells;
    sub.min_cells = max(get_opt(sub, 'min_cells', 0), new_cells);
end
