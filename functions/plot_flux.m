function plot_flux(caseX, ~, t_max)
%PLOT_FLUX Plot the explicit interface heat flux history for a case.

    if nargin < 3 || isempty(t_max)
        t_max = 0.1;
    end

    if ~isfield(caseX, 'num') || ~isstruct(caseX.num)
        warning('plot_flux:NoSnapshot', 'Case %s has no numerical snapshot.', caseX.label);
        return;
    end

    snap = caseX.num;
    if ~isfield(snap, 'q') || ~isstruct(snap.q)
        warning('plot_flux:MissingHistory', 'Snapshot lacks flux history for case %s.', caseX.label);
        return;
    end

    [times, flux, ~, ~, seed_time] = extract_history(snap);
    if isempty(flux)
        warning('plot_flux:EmptyHistory', 'Flux history is empty for case %s.', caseX.label);
        return;
    end

    if isempty(t_max)
        t_max = max(times);
    else
        t_max = max(t_max, max(times));
    end

    figure('Name',['Interface flux vs time — ', caseX.label]); hold on; box on; grid on;
    plot(times, flux, '-', 'LineWidth', 1.6, 'Color', [0.95 0.65 0.2], 'DisplayName', 'Explicit flux');
    if seed_time > 0
        xline(seed_time, 'k:', 'LineWidth', 1.0, 'DisplayName', 'Seed time');
    end
    xlabel('time  t  [s]');
    ylabel('interface heat flux  q  [W m^{-2}]');
    title(['Explicit interface flux — ', caseX.label]);
    legend('Location','Best');
    xlim([0, t_max]);
end

function [times, flux, Tw_face, Ts_face, seed_time] = extract_history(snap)
    q_struct = snap.q;
    if isfield(q_struct,'t_phys')
        times = q_struct.t_phys(:)';
    elseif isfield(q_struct,'t')
        if isfield(snap,'t_offset')
            times = q_struct.t(:)' + snap.t_offset;
        else
            times = q_struct.t(:)';
        end
    else
        times = linspace(0, snap.t, numel(q_struct.val));
    end
    flux = q_struct.val(:)';
    if isfield(q_struct,'Tw_face')
        Tw_face = q_struct.Tw_face(:)';
    else
        Tw_face = [];
    end
    if isfield(q_struct,'Ts_face')
        Ts_face = q_struct.Ts_face(:)';
    else
        Ts_face = [];
    end
    if isfield(snap,'seed') && isstruct(snap.seed) && isfield(snap.seed,'time')
        seed_time = snap.seed.time;
    else
        seed_time = 0;
    end
end
