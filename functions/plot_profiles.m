function plot_profiles(caseX)
%PLOT_PROFILES Plot the explicit temperature profile for a case.

    if ~isfield(caseX, 'num') || ~isstruct(caseX.num)
        warning('plot_profiles:NoSnapshot', 'Case %s has no numerical snapshot.', caseX.label);
        return;
    end

    snap = caseX.num;
    if ~isfield(snap, 'x') || ~isfield(snap, 'T')
        warning('plot_profiles:MissingProfile', 'Snapshot lacks x/T arrays for case %s.', caseX.label);
        return;
    end

    x = snap.x(:);
    T = snap.T(:);
    params = caseX.params;
    if isfield(params,'Tf'), Tf = params.Tf; else, Tf = 0; end
    if isfield(params,'Tw_inf'), Tw = params.Tw_inf; else, Tw = min(T); end
    if isfield(params,'Tl_inf'), Tl_inf = params.Tl_inf; else, Tl_inf = max(T); end

    figure('Name',['Explicit profile @ t_{phys} — ', caseX.label]); hold on; box on; grid on;
    plot(x, T, '-', 'LineWidth', 1.8, 'Color', [0.2 0.4 0.8], 'DisplayName', 'Explicit numeric');
    xline(0, 'k:', 'LineWidth', 1.0, 'DisplayName', 'Wall/solid boundary');
    if isfield(snap, 'S')
        xline(snap.S, 'm--', 'LineWidth', 1.2, 'DisplayName', 'Interface S(t)');
    end

    xlabel('Physical coordinate  x  [m]');
    ylabel('Temperature  [^{\circ}C]');
    title(['Explicit profile @ t = t_{phys} — ', caseX.label]);
    legend('Location','Best');
    xlim([min(x), max(x)]);
    ylim([min([Tw Tf Tl_inf]) - 5, max([Tw Tf Tl_inf]) + 5]);
end
