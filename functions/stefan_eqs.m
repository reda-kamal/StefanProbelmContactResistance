function F = stefan_eqs(x,aw,as,al,kw,ks,kl,L,rhos,Tw_inf,Tl_inf,Tf)
%STEFAN_EQS Nonlinear system for interface parameters (lambda, Ti).
    lam = x(1); Ti  = x(2);
    mu  = lam*sqrt(as/al);
    F1 = kw*sqrt(as)*(Ti - Tw_inf) ...
       - ks*sqrt(aw)/erf(lam)*(Tf - Ti);
    termL = kl*(Tf - Tl_inf)/(lam*sqrt(pi)) * ( - mu / erfcx(mu) );
    termS = ks*(Tf - Ti)/sqrt(pi) * exp(-lam^2)/erf(lam);
    F2 = rhos*L*as*lam + termL - termS;
    F = [F1; F2];
end
