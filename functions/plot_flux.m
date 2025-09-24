function plot_flux(caseX, R_c, t_max)
%PLOT_FLUX Plot interface heat flux histories for VAM and explicit solutions.
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
