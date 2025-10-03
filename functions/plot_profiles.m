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
        else
            if isfield(num_struct,'explicit')
                snap = num_struct.explicit;
                plot(snap.x, snap.T, '.', 'MarkerSize', 6, 'DisplayName','Explicit numeric');
                xline(snap.S, 'm--','LineWidth',1.2, 'DisplayName','S^{num}_{exp}');
                warn_if_unbounded(snap, 'explicit profile');
            end
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
