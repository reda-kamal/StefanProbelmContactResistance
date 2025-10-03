function out = run_vam_case(label, k_w, rho_w, c_w, M, R_c, t_phys, opts)
%RUN_VAM_CASE Assemble and execute an explicit snapshot for a material case.
%
% OUT = RUN_VAM_CASE(LABEL, k_w, rho_w, c_w, M, R_c, t_phys, OPTS) runs the
% explicit finite-difference solver for a three-domain Stefan problem with a
% contact resistance R_c.  Material properties are provided via struct M with
% fields k_s, rho_s, c_s, k_l, rho_l, c_l, L, Tf, Tw_inf, and Tl_inf.  OPTS may
% include an 'explicit' struct that is forwarded to explicit_stefan_snapshot.

    if nargin < 8 || isempty(opts)
        opts = struct();
    end

    explicit_opts = get_opt(opts, 'explicit', struct());

    k_s = M.k_s; rho_s = M.rho_s; c_s = M.c_s;
    k_l = M.k_l; rho_l = M.rho_l; c_l = M.c_l;
    L   = M.L;   Tf    = M.Tf;    Tw_inf = M.Tw_inf;  Tl_inf = M.Tl_inf;

    alpha_w = k_w/(rho_w*c_w);
    alpha_s = k_s/(rho_s*c_s);
    alpha_l = k_l/(rho_l*c_l);

    params_struct = struct('Tw_inf',Tw_inf,'Tl_inf',Tl_inf,'Tf',Tf, ...
        'alpha_w',alpha_w,'alpha_s',alpha_s,'alpha_l',alpha_l, ...
        'k_w',k_w,'k_s',k_s,'k_l',k_l,'L',L);

    snap_explicit = explicit_stefan_snapshot(k_w, rho_w, c_w, M, R_c, t_phys, params_struct, explicit_opts);

    out = struct();
    out.label  = label;
    out.params = params_struct;
    out.num    = snap_explicit;
end

function val = get_opt(opts, field, default)
%GET_OPT Fetch an option from a struct with a default fallback.
    if isstruct(opts) && isfield(opts, field) && ~isempty(opts.(field))
        val = opts.(field);
    else
        val = default;
    end
end
