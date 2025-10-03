function plot_diff_profile(caseX)
%PLOT_DIFF_PROFILE Plot explicit temperature relative to fusion temperature.

    if ~isfield(caseX, 'num') || ~isstruct(caseX.num)
        warning('plot_diff_profile:NoSnapshot', 'Case %s has no numerical snapshot.', caseX.label);
        return;
    end

    snap = caseX.num;
    if ~isfield(snap, 'x') || ~isfield(snap, 'T')
        warning('plot_diff_profile:MissingProfile', 'Snapshot lacks x/T arrays for case %s.', caseX.label);
        return;
    end

    params = caseX.params;
    if isfield(params,'Tf'), Tf = params.Tf; else, Tf = 0; end

    x = snap.x(:);
    T = snap.T(:);
    diff = T - Tf;

    figure('Name',['Explicit profile − T_f @ t_{phys} — ', caseX.label]); hold on; box on; grid on;
    plot(x, diff, 'm-', 'LineWidth', 1.8, 'DisplayName', 'Explicit (T − T_f)');
    yline(0, 'k:', 'LineWidth', 1.0);
    xline(0, 'k:', 'LineWidth', 1.0);
    if isfield(snap, 'S')
        xline(snap.S, 'm--', 'LineWidth', 1.2, 'DisplayName', 'Interface S(t)');
    end

    xlabel('Physical coordinate  x  [m]');
    ylabel('Temperature offset  [^{\circ}C]');
    title(['Explicit deviation from T_f — ', caseX.label]);
    legend('Location','Best');
    xlim([min(x), max(x)]);
end
