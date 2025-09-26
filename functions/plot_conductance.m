function plot_conductance(caseX, R_c, t_max)
%PLOT_CONDUCTANCE Plot effective contact conductance envelopes from VAM.
    if nargin < 3 || isempty(t_max), t_max = 0.1; end
    p  = caseX.params;   hc = 1./R_c;

    alpha_w = p.alpha_w;  alpha_s = p.alpha_s;
    S0_e = p.S0_e;  E0_e = p.E0_e;  t0_e = p.t0_e;
    S0_l = p.S0_l;  E0_l = p.E0_l;  t0_l = p.t0_l;
    kw = p.k_w; ks = p.k_s;
    gfun = @(chi) (2.*chi.*exp(-chi.^2)) ./ (sqrt(pi).*erf(chi));
    t  = linspace(0, t_max, 1000);

    tpe   = t + t0_e;
    phi_s = S0_e ./ (2*sqrt(alpha_s*tpe));
    phi_w = E0_e ./ (2*sqrt(alpha_w*tpe));
    hs_e  = (ks./S0_e) .* gfun(phi_s);
    hw_e  = (kw./E0_e) .* gfun(phi_w);
    he_e  = 1 ./ (1./hw_e + 1./hs_e); he_e(1) = hc;

    tpl    = t + t0_l;
    phi_sL = S0_l ./ (2*sqrt(alpha_s*tpl));
    phi_wL = E0_l ./ (2*sqrt(alpha_w*tpl));
    hs_l   = (ks./S0_l) .* gfun(phi_sL);
    hw_l   = (kw./E0_l) .* gfun(phi_wL);
    he_l   = 1 ./ (1./hw_l + 1./hs_l);

    figure('Name',['h_{eff}(t) — ', caseX.label]); hold on; grid on; box on;
    plot(t, he_e, 'k--','LineWidth',1.6, 'DisplayName','VAM (t=0 calibration)');
    plot(t, he_l, 'b-' ,'LineWidth',1.7, 'DisplayName','VAM (t=\infty calibration)');
    yline(hc, 'r-.','LineWidth',1.4, 'DisplayName','true h_c (=1/R_c)');
    plot(0, hc, 'ko', 'MarkerFaceColor','k', 'DisplayName','he^{(0)}(0)=h_c');
    xlabel('time t [s]');
    ylabel('effective conductance h_{eff}(t) [W m^{-2} K^{-1}]');
    title(['Effective contact conductance vs time — ', caseX.label]);
    legend('Location','SouthEast');
    xlim([0, t_max]);
end
