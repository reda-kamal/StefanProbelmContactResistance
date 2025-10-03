function plot_flux(caseX, R_c, t_max)
%PLOT_FLUX Plot interface heat flux histories for VAM and numerical solutions.
    if nargin<3 || isempty(t_max)
        if isfield(caseX,'num') && ~isempty(caseX.num)
            num_struct = caseX.num;
            if isfield(num_struct,'t')
                t_max = max(0.1, num_struct.t);
            elseif isfield(num_struct,'explicit') && isfield(num_struct.explicit,'t')
                t_max = max(0.1, num_struct.explicit.t);
            else
                t_max = 0.1;
            end
        else
            t_max = 0.1;
        end
    end
    Nt = 600;  t = linspace(0,t_max,Nt);
    p = caseX.params;

    % VAM contact-law fluxes q = (Tw(0^-)-Ts(0^+))/Rc
    [~,~,q0_rc] = vam_face_temps_and_q(p,'early', t, R_c);
    [~,~,qI_rc] = vam_face_temps_and_q(p,'late',  t, R_c);

    figure('Name',['Interface flux vs time — ',caseX.label]); hold on; grid on; box on;
    plot(t, q0_rc, 'k--','LineWidth',1.6, 'DisplayName','VAM^{(0)}: q=\Delta T/R_c');
    plot(t, qI_rc, 'b-' ,'LineWidth',1.7, 'DisplayName','VAM^{(\infty)}: q=\Delta T/R_c');

    if isfield(caseX,'num') && ~isempty(caseX.num)
        num_struct = caseX.num;
        if isfield(num_struct, 'q')
            [th, qh, seed_t, label] = extract_flux(num_struct, 'Explicit (const R_c)');
            plot(th, qh, '-', 'LineWidth',1.6, 'Color',[0.95 0.65 0.2], 'DisplayName', label);
            if seed_t > 0
                xline(seed_t, 'Color',[0.4 0.4 0.4], 'LineStyle',':', 'LineWidth',1.0, ...
                    'DisplayName','Seed time (explicit)');
            end
            warn_if_flux_unbounded(num_struct, 'explicit flux');
        else
            if isfield(num_struct,'explicit')
                snap_exp = num_struct.explicit;
                [th, qh, seed_t, label] = extract_flux(snap_exp, 'Explicit (const R_c)');
                plot(th, qh, '-', 'LineWidth',1.6, 'Color',[0.95 0.65 0.2], 'DisplayName', label);
                if seed_t > 0
                    xline(seed_t, 'Color',[0.4 0.4 0.4], 'LineStyle',':', 'LineWidth',1.0, ...
                        'DisplayName','Seed time (explicit)');
                end
                warn_if_flux_unbounded(snap_exp, 'explicit flux');
            end
        end
    end
    xlabel('time t [s]');
    ylabel('interface heat flux q(0,t) [W m^{-2}]');
    title(['Interface flux vs time — ', caseX.label]);
    legend('Location','SouthEast');
    xlim([0, t_max]);
end

function warn_if_flux_unbounded(snap, label)
    if ~isstruct(snap)
        return;
    end
    if isfield(snap, 'meta') && isstruct(snap.meta) && isfield(snap.meta, 'bounds')
        b = snap.meta.bounds;
        if isstruct(b) && isfield(b,'ok') && ~b.ok
            flux_violation = NaN;
            if isfield(b,'flux') && isstruct(b.flux) && isfield(b.flux,'max_violation')
                flux_violation = b.flux.max_violation;
            end
            warning('plot_flux:NotBounded', ...
                'Numerical %s exceeds VAM flux envelope (Δq=%g W/m^2).', ...
                label, flux_violation);
        end
    end
end

function [th, qh, seed_t, label] = extract_flux(snap, base_label)
    if isfield(snap.q, 't_phys')
        th = snap.q.t_phys;
    else
        th = snap.q.t;
        if isfield(snap, 't_offset')
            th = th + snap.t_offset;
        end
    end
    qh = snap.q.val;
    if isfield(snap, 'seed') && isfield(snap.seed, 'time')
        seed_t = snap.seed.time;
    else
        seed_t = 0;
    end
    label = base_label;
    if isfield(snap, 'history') && isfield(snap.history, 'flux_window') && snap.history.flux_window > 1
        label = sprintf('%s (%d-pt mov. avg.)', base_label, snap.history.flux_window);
    end
end
