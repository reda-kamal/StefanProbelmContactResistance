function plot_flux(caseX, R_c, t_max)
%PLOT_FLUX Plot interface heat flux histories for VAM and numerical solutions.
    if nargin<3 || isempty(t_max)
        if isfield(caseX,'num') && ~isempty(caseX.num)
            num_struct = caseX.num;
            if isfield(num_struct,'t')
                t_max = max(0.1, num_struct.t);
            elseif isfield(num_struct,'explicit') && isfield(num_struct.explicit,'t')
                t_max = max(0.1, num_struct.explicit.t);
            elseif isfield(num_struct,'enthalpy') && isfield(num_struct.enthalpy,'t')
                t_max = max(0.1, num_struct.enthalpy.t);
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
        else
            if isfield(num_struct,'explicit')
                [th, qh, seed_t, label] = extract_flux(num_struct.explicit, 'Explicit (const R_c)');
                plot(th, qh, '-', 'LineWidth',1.6, 'Color',[0.95 0.65 0.2], 'DisplayName', label);
                if seed_t > 0
                    xline(seed_t, 'Color',[0.4 0.4 0.4], 'LineStyle',':', 'LineWidth',1.0, ...
                        'DisplayName','Seed time (explicit)');
                end
            end
            if isfield(num_struct,'enthalpy')
                [thH, qhH, seedH, labelH] = extract_flux(num_struct.enthalpy, 'Enthalpy (const R_c)');
                plot(thH, qhH, '--', 'LineWidth',1.5, 'Color',[0.3 0.75 0.93], 'DisplayName', labelH);
                if seedH > 0
                    xline(seedH, 'Color',[0.1 0.5 0.7], 'LineStyle',':', 'LineWidth',1.0, ...
                        'DisplayName','Seed time (enthalpy)');
                end
            end
        end
    end
    xlabel('time t [s]');
    ylabel('interface heat flux q(0,t) [W m^{-2}]');
    title(['Interface flux vs time — ', caseX.label]);
    legend('Location','SouthEast');
    xlim([0, t_max]);
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
