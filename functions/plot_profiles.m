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
        else
            if isfield(num_struct,'explicit')
                snap = num_struct.explicit;
                plot(snap.x, snap.T, '.', 'MarkerSize', 6, 'DisplayName','Explicit numeric');
                xline(snap.S, 'm--','LineWidth',1.2, 'DisplayName','S^{num}_{exp}');
            end
            if isfield(num_struct,'enthalpy')
                snapH = num_struct.enthalpy;
                plot(snapH.x, snapH.T, 'o', 'MarkerSize', 4, 'LineStyle','none', ...
                    'MarkerFaceColor','none', 'MarkerEdgeColor',[0.3 0.75 0.93], ...
                    'DisplayName','Enthalpy numeric');
                xline(snapH.S, 'c-.','LineWidth',1.2, 'DisplayName','S^{num}_{enth}');
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
