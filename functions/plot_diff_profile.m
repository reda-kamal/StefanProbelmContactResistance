function plot_diff_profile(caseX)
%PLOT_DIFF_PROFILE Plot the difference between late- and early-time VAM profiles.
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
