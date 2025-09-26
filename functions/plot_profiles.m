function plot_profiles(caseX)
%PLOT_PROFILES Plot early/late VAM and explicit temperature profiles.
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
        num_struct = caseX.num;
        if isfield(num_struct, 'x') && isfield(num_struct, 'T')
            xn = num_struct.x;  Tn = num_struct.T;  Sn = num_struct.S;
            plot(xn, Tn, '.', 'MarkerSize', 6, 'DisplayName','Explicit numeric');
            xline(Sn, 'm--','LineWidth',1.2, 'DisplayName','S^{num}');
            warn_if_unbounded(num_struct, 'numeric profile');
        elseif isfield(num_struct,'explicit')
            snap = num_struct.explicit;
            [x_line, T_line, x_pts, T_pts] = reconstruct_numeric_profile(snap, caseX.params);
            explicit_color = [0.60, 0.00, 0.80];
            if ~isempty(x_line)
                plot(x_line, T_line, 'Color', explicit_color, 'LineWidth', 1.6, ...
                    'DisplayName','Explicit numeric');
                if ~isempty(x_pts)
                    plot(x_pts, T_pts, '.', 'Color', explicit_color, 'MarkerSize', 5, ...
                        'HandleVisibility','off');
                end
            else
                plot(snap.x, snap.T, '.', 'MarkerSize', 6, 'DisplayName','Explicit numeric');
            end
            xline(snap.S, 'm--','LineWidth',1.2, 'DisplayName','S^{num}_{exp}');
            warn_if_unbounded(snap, 'explicit profile');
        end
    end

    xlabel('Physical coordinate  x  [m]');
    ylabel('Temperature  [^{\circ}C]');
    title(['Two VAM vs explicit @ t = t_{phys} — ', caseX.label]);
    legend('Location','SouthEast');
    ylim([min([Tw Tf Tl_inf])-5, max([Tw Tf Tl_inf])+5]);
    xlim([min(x), max(x)]);
end

function warn_if_unbounded(snap, label)
    if ~isstruct(snap)
        return;
    end
    if isfield(snap, 'meta') && isstruct(snap.meta) && isfield(snap.meta, 'bounds')
        b = snap.meta.bounds;
        if isstruct(b) && isfield(b,'ok') && ~b.ok
            profile_violation = NaN;
            if isfield(b,'profile') && isstruct(b.profile) && isfield(b.profile,'max_violation')
                profile_violation = b.profile.max_violation;
            end
            flux_violation = NaN;
            if isfield(b,'flux') && isstruct(b.flux) && isfield(b.flux,'max_violation')
                flux_violation = b.flux.max_violation;
            end
            warning('plot_profiles:NotBounded', ...
                'Numerical %s exceeds VAM envelope (ΔT=%g°C, Δq=%g W/m^2).', ...
                label, profile_violation, flux_violation);
        end
    end
end

function [x_line, T_line, x_pts, T_pts] = reconstruct_numeric_profile(snap, params)
%RECONSTRUCT_NUMERIC_PROFILE Build a piecewise profile with the freezing front.
    x_line = [];
    T_line = [];
    x_pts = [];
    T_pts = [];
    if ~isstruct(snap) || ~isfield(snap,'x') || ~isfield(snap,'T')
        return;
    end

    x_pts = snap.x(:);
    T_pts = snap.T(:);

    if ~isfield(snap,'grid') || ~isstruct(snap.grid) || ~isfield(snap,'S')
        return;
    end

    Nw = snap.grid.N_wall;
    Nf = snap.grid.N_fluid;
    if isempty(Nw) || isempty(Nf) || numel(x_pts) ~= (Nw + Nf)
        return;
    end

    xw = x_pts(1:Nw);
    Tw = T_pts(1:Nw);
    xf = x_pts(Nw+1:end);
    Tf_cells = T_pts(Nw+1:end);

    dxf = snap.grid.dx_fluid;
    if isempty(dxf) || ~isfinite(dxf) || dxf <= 0
        return;
    end

    m = max(1, min(Nf-1, floor(snap.S/dxf)));
    x_solid = xf(1:m);
    T_solid = Tf_cells(1:m);
    x_liquid = xf(m+1:end);
    T_liquid = Tf_cells(m+1:end);

    Tw_face = final_face_temp(snap, 'Tw');
    if isnan(Tw_face)
        if ~isempty(Tw)
            Tw_face = Tw(1);
        else
            Tw_face = params.Tw_inf;
        end
    end

    Ts_face = final_face_temp(snap, 'Ts');
    if isnan(Ts_face)
        Ts_face = params.Tf;
    end

    Tf_val = params.Tf;

    x_line = [xw; 0; NaN; 0; x_solid; snap.S; NaN; snap.S; x_liquid];
    T_line = [Tw; Tw_face; NaN; Ts_face; T_solid; Tf_val; NaN; Tf_val; T_liquid];
end

function val = final_face_temp(snap, which)
%FINAL_FACE_TEMP Extract the final wall face temperature for plotting.
    val = NaN;
    if isfield(snap,'faces') && isstruct(snap.faces) && isfield(snap.faces,'wall')
        wall_face = snap.faces.wall;
        if strcmp(which,'Tw') && isfield(wall_face,'Tw')
            val = wall_face.Tw;
            return;
        elseif strcmp(which,'Ts') && isfield(wall_face,'Ts')
            val = wall_face.Ts;
            return;
        end
    end

    if isfield(snap,'q') && isstruct(snap.q)
        if strcmp(which,'Tw') && isfield(snap.q,'Tw_face') && ~isempty(snap.q.Tw_face)
            val = snap.q.Tw_face(end);
            return;
        elseif strcmp(which,'Ts') && isfield(snap.q,'Ts_face') && ~isempty(snap.q.Ts_face)
            val = snap.q.Ts_face(end);
            return;
        end
    end
end
