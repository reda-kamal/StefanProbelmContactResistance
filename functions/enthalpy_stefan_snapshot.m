function snap = enthalpy_stefan_snapshot(k_w,rho_w,c_w, M, R_c, t_end, params, opts)
%ENTHALPY_STEFAN_SNAPSHOT One-dimensional enthalpy solver with contact Rc.
%   This solver mirrors the explicit finite-difference grid/controls but
%   advances the solid/liquid side using an enthalpy formulation.  The
%   temperature/phase state is recovered from the cell enthalpy after each
%   update, which permits a moving freezing front without explicitly
%   tracking the interface location.

    if nargin < 8 || isempty(opts)
        opts = struct();
    end

    % Unpack material properties
    k_s = M.k_s; rho_s = M.rho_s; c_s = M.c_s;
    k_l = M.k_l; rho_l = M.rho_l; c_l = M.c_l;
    Tf    = M.Tf;    Tw_inf = M.Tw_inf; Tl_inf = M.Tl_inf;

    % Diffusivities (for time-step sizing)
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

    % Mesh controls (same knobs as explicit solver)
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

    wall_extent_factor = get_opt(wall_opts, 'extent_factor', domain_factor);
    fluid_extent_factor = get_opt(fluid_opts, 'extent_factor', domain_factor);

    wall_min_length = get_opt(wall_opts, 'min_length', min_length_legacy);
    fluid_min_length = get_opt(fluid_opts, 'min_length', min_length_legacy);

    wall_length_user = get_opt(wall_opts, 'length', get_opt(wall_opts, 'extent', []));
    wall_length_fixed = ~isempty(wall_length_user);
    fluid_length_user = get_opt(fluid_opts, 'length', get_opt(fluid_opts, 'extent', []));
    fluid_length_fixed = ~isempty(fluid_length_user);

    wall_min_cells = get_opt(wall_opts, 'min_cells', min_cells_legacy);
    fluid_min_cells = get_opt(fluid_opts, 'min_cells', min_cells_legacy);

    wall_cells = get_opt(wall_opts, 'cells', []);
    if isempty(wall_cells)
        wall_dx = get_opt(wall_opts, 'dx', []);
        if ~isempty(wall_dx) && wall_dx > 0
            wall_cells = max(3, round(max(wall_length_user, wall_min_length)/wall_dx));
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
            fluid_cells = max(3, round(max(fluid_length_user, fluid_min_length)/fluid_dx));
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

    seed_time = 0;
    t_final_phys = t_end;
    for iter = 1:5 %#ok<FXUP>
        if wall_length_fixed
            Lw = max(wall_length_user, wall_min_length);
        else
            Lw = max(wall_extent_factor*sqrt(aw*max(t_final_phys,eps)), wall_min_length);
        end
        if fluid_length_fixed
            Lf = max(fluid_length_user, fluid_min_length);
        else
            Lf = max(fluid_extent_factor*sqrt(max(as,al)*max(t_final_phys,eps)), fluid_min_length);
        end

        dxw = Lw/Nw;   dxf = Lf/Nf;
        seed_thickness = min_seed_cells * dxf;
        seed_time_new = ((seed_thickness + S0_e)^2)/(4*lam^2*as) - t0_e;
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
        Lw = max(wall_extent_factor*sqrt(aw*max(t_final_phys,eps)), wall_min_length);
    else
        Lw = max(wall_length_user, wall_min_length);
    end
    if ~fluid_length_fixed
        Lf = max(fluid_extent_factor*sqrt(max(as,al)*max(t_final_phys,eps)), fluid_min_length);
    else
        Lf = max(fluid_length_user, fluid_min_length);
    end

    dxw = Lw/Nw;   xw = -((1:Nw)' - 0.5)*dxw;
    dxf = Lf/Nf;   xf =  ((1:Nf)' - 0.5)*dxf;

    seed_thickness = min_seed_cells * dxf;
    seed_time = ((seed_thickness + S0_e)^2)/(4*lam^2*as) - t0_e;
    seed_time = max(0, seed_time);
    t_final_phys = max(t_end, seed_time);

    % Evaluate the early-time VAM solution at the seed time for ICs
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

    S_seed = min((Nf-1)*dxf, max(dxf, Se_seed));

    % Convert initial fluid temperature to enthalpy relative to Tl_inf
    rhoLcpL = rho_l * c_l;
    rhoScpS = rho_s * c_s;
    L_vol   = rho_s * M.L;
    H = zeros(Nf,1);
    for j = 1:Nf
        if solid_mask(j)
            H(j) = -L_vol + rhoScpS * (Tfld(j) - Tf);
        else
            H(j) = rhoLcpL * (Tfld(j) - Tl_inf);
        end
    end

    % Explicit time step (CFL with both solid/liquid diffusivities)
    CFL = get_opt(opts, 'CFL', 0.3);
    alpha_max = max(as, al);
    dt_base = CFL * min( dxw^2/(2*aw), dxf^2/(2*alpha_max) );
    sim_duration = max(t_end - seed_time, 0);
    if sim_duration <= 0
        nsteps = 0;
        dt_base = 0;
    else
        nsteps = max(1, ceil(sim_duration/dt_base));
        dt_base = min(dt_base, sim_duration/nsteps);
    end

    t  = 0;
    t_phys = seed_time;
    t_rel  = 0;

    coeff = local_coeffs(dt_base, aw, dxw);
    curr_dt = dt_base;

    Rw = dxw/(2*k_w);

    % history buffers
    if history_dt > 0
        n_hist_max = max(2, ceil(sim_duration/history_dt) + 2);
    else
        stride = max(1, max(1, floor(max(nsteps,1)/max(nsave,1)))); %#ok<NASGU>
        n_hist_max = max(2, ceil(max(nsteps,1)/stride) + 1);
    end
    t_hist = zeros(n_hist_max,1);
    q_hist = zeros(n_hist_max,1);
    Tw_face_hist = zeros(n_hist_max,1);
    Ts_face_hist = zeros(n_hist_max,1);
    ksave  = 0;
    last_save_time = -inf;

    % Initial state (seed)
    [Tfluid, phi_liq, k_cell] = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol);
    [q_seed, Tw_face_seed, Ts_face_seed] = contact_flux(Tw(1), Tfluid(1), phi_liq(1), Tf, Rw, R_c, dxf, k_cell(1));
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
            coeff = local_coeffs(dt_step, aw, dxw);
            curr_dt = dt_step;
        end

        % Reconstruct temperature/phase/conductivity
        [Tfluid, phi_liq, k_cell] = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol);

        % Contact flux with supercooling-aware interface handling
        [q0, Tw_face, Ts_face] = contact_flux(Tw(1), Tfluid(1), phi_liq(1), Tf, Rw, R_c, dxf, k_cell(1));

        % Wall update
        Tw_new = Tw;
        if Nw >= 2
            Tw_new(1) = Tw(1) + coeff.wall_edge * (Tw_face - 2*Tw(1) + Tw(2));
        end
        if Nw > 2
            iL=2; iR=Nw-1;
            Tw_new(iL:iR) = Tw(iL:iR) + coeff.wall_bulk .* ...
                ( Tw(iL+1:iR+1) - 2*Tw(iL:iR) + Tw(iL-1:iR-1) );
        end
        Tw_new(end) = Tw_inf;
        Tw = Tw_new;

        % Fluid enthalpy update via fluxes
        % Compute fluxes at interior faces
        q_faces = zeros(Nf+1,1); % 1..Nf+1 with 1=left boundary, Nf+1=right boundary
        q_faces(1) = q0;  % positive into fluid
        for j = 1:Nf-1
            k_face = harmonic_mean(k_cell(j), k_cell(j+1));
            q_faces(j+1) = -k_face * (Tfluid(j+1) - Tfluid(j)) / dxf;
        end
        k_last = max(k_cell(end), eps);
        q_faces(Nf+1) = -k_last * (Tl_inf - Tfluid(end)) / (0.5*dxf);

        for j = 1:Nf
            qL = q_faces(j);
            qR = q_faces(j+1);
            H(j) = H(j) + dt_step * (qL - qR) / dxf;
        end

        % Advance clocks
        t = t + dt_step;
        t_phys = t_phys + dt_step;
        t_rel  = t_phys - seed_time;

        % Recompute state at updated level for history
        [Tfluid, phi_liq, k_cell] = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol);
        [q_contact, Tw_face_new, Ts_face_new] = contact_flux(Tw(1), Tfluid(1), phi_liq(1), Tf, Rw, R_c, dxf, k_cell(1));

        should_save = false;
        if history_dt > 0
            if (t_rel - last_save_time) >= history_dt*(1 - 1e-8) || n == nsteps
                should_save = true;
            end
        else
            if mod(n, max(1,floor(max(nsteps,1)/max(nsave,1))))==0 || n == nsteps
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

    [Tfluid, phi_liq, k_cell] = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol); %#ok<ASGLU>

    % Estimate freezing front location where liquid fraction drops below 0.5
    idx = find(phi_liq < 0.5, 1, 'first');
    if isempty(idx)
        S_est = dxf * (phi_liq(end) >= 0.5) * Nf;
    elseif idx == 1
        S_est = dxf/2;
    else
        frac = (0.5 - phi_liq(idx-1)) / max(phi_liq(idx-1) - phi_liq(idx), eps);
        S_est = (idx-1 + max(0,min(1,frac)) - 0.5) * dxf;
    end
    S_est = max(dxf, min((Nf-1)*dxf, S_est));

    snap.x   = [xw; xf];
    snap.T   = [Tw; Tfluid];
    snap.Tw  = Tw;
    snap.Tf  = Tfluid;
    snap.H   = H;
    snap.phi = phi_liq;
    snap.S   = S_est;
    snap.t   = t_end;
    snap.t_rel = t_rel;
    snap.t_offset = seed_time;
    snap.seed = struct('time',seed_time,'thickness',S_seed,'cell_width',dxf,'Se_vam',Se_seed);
    snap.grid = struct('dx_wall',dxw,'dx_fluid',dxf,'N_wall',Nw,'N_fluid',Nf,'L_wall',Lw,'L_fluid',Lf);
    snap.history = struct('history_dt',history_dt,'flux_window',flux_window,'nsave',nsave);

    t_hist = t_hist(1:ksave);
    q_hist = q_hist(1:ksave);
    snap.q.t   = t_hist;
    snap.q.val = q_hist;
    snap.q.Tw_face = Tw_face_hist(1:ksave);
    snap.q.Ts_face = Ts_face_hist(1:ksave);
    snap.q.t_phys = t_hist + seed_time;

    if flux_window > 1
        snap.q.val = moving_average(snap.q.val, flux_window);
    end
end

function coeff = local_coeffs(dt, aw, dxw)
    coeff.wall_bulk = aw*dt/(dxw^2);
    coeff.wall_edge = 2*coeff.wall_bulk;
end

function val = get_opt(opts, field, default)
    if isstruct(opts) && isfield(opts, field) && ~isempty(opts.(field))
        val = opts.(field);
    else
        val = default;
    end
end

function [T, phi, k_cell] = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol)
%ENTHALPY_STATE Map cell enthalpy to temperature, liquid fraction, and k.
    H_tf = rhoLcpL * (Tf - Tl_inf);
    N = numel(H);
    T = zeros(N,1);
    phi = zeros(N,1);
    k_cell = zeros(N,1);
    for j = 1:N
        Hj = H(j);
        if Hj >= H_tf
            % Liquid heated above Tf
            T(j) = Tf + (Hj - H_tf)/rhoLcpL;
            phi(j) = 1;
            k_cell(j) = k_l;
        elseif Hj >= 0
            % Supercooled liquid (no phase change yet)
            T(j) = Tl_inf + Hj/rhoLcpL;
            phi(j) = 1;
            k_cell(j) = k_l;
        elseif Hj >= -L_vol
            % Mushy/phase-change zone (T held at Tf)
            phi(j) = max(0, min(1, 1 + Hj/L_vol));
            T(j) = Tf;
            k_cell(j) = phi(j)*k_l + (1-phi(j))*k_s;
        else
            % Fully solid and cooling below Tf
            T(j) = Tf + (Hj + L_vol)/rhoScpS;
            phi(j) = 0;
            k_cell(j) = k_s;
        end
    end
end

function k_face = harmonic_mean(kL, kR)
    if kL <= 0 && kR <= 0
        k_face = 0;
    elseif kL <= 0
        k_face = kR;
    elseif kR <= 0
        k_face = kL;
    else
        k_face = 2*kL*kR/(kL + kR);
    end
end

function [q0, Tw_face, Ts_face] = contact_flux(Tw_cell, T_cell, phi_cell, Tf, Rw, R_c, dxf, k_cell)
%CONTACT_FLUX Compute wall contact flux honoring supercooling/mushy physics.
    phi_cell = min(max(phi_cell, 0), 1);
    tol = 1e-8;

    if phi_cell > tol && phi_cell < 1 - tol
        % Mushy control volume: enforce interface temperature at Tf and
        % collapse the half-cell resistance (uniform Tf assumption).
        q0 = (Tw_cell - Tf) / (Rw + R_c);
        Tw_face = Tw_cell - Rw*q0;
        Ts_face = Tf;
    else
        k_eff = max(k_cell, eps);
        Rs_eff = dxf / (2*k_eff);
        q0 = (Tw_cell - T_cell) / (Rw + R_c + Rs_eff);
        Tw_face = Tw_cell - Rw*q0;
        Ts_face = T_cell + Rs_eff*q0;
    end
end

function y = moving_average(y, window)
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
