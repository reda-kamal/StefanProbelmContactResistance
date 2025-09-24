function [Tw0m,Ts0p,q_rc] = vam_face_temps_and_q(p, which, t, R_c)
%VAM_FACE_TEMPS_AND_Q Face temperatures from VAM and resulting flux by q=ΔT/Rc.
    alpha_w = p.alpha_w; alpha_s = p.alpha_s;
    lam = p.lam; Tf = p.Tf; Ti = p.Ti; Tw_inf = p.Tw_inf;
    if strcmp(which,'early')
        t0 = p.t0_e; S0 = p.S0_e; E0 = p.E0_e;
    else
        t0 = p.t0_l; S0 = p.S0_l; E0 = p.E0_l;
    end
    den_w = 2*sqrt(alpha_w*(t+t0));
    den_s = 2*sqrt(alpha_s*(t+t0));
    Tw0m = Ti + (Ti - Tw_inf) .* erf( (-E0)./den_w );
    Ts0p = Ti + (Tf - Ti)     .* erf( (+S0)./den_s ) ./ erf(lam);
    q_rc = (Tw0m - Ts0p) ./ R_c;
end
