function plot_conductance(caseX, R_c, t_max)
%PLOT_CONDUCTANCE Plot effective contact conductance from explicit flux.

    if nargin < 3 || isempty(t_max)
        t_max = 0.1;
    end

    if ~isfield(caseX, 'num') || ~isstruct(caseX.num)
        warning('plot_conductance:NoSnapshot', 'Case %s has no numerical snapshot.', caseX.label);
        return;
    end

    snap = caseX.num;
    if ~isfield(snap, 'q') || ~isstruct(snap.q)
        warning('plot_conductance:MissingHistory', 'Snapshot lacks flux history for case %s.', caseX.label);
        return;
    end

    [times, flux, Tw_face, Ts_face, seed_time] = extract_history(snap);
    if isempty(flux)
        warning('plot_conductance:EmptyHistory', 'Flux history is empty for case %s.', caseX.label);
        return;
    end

    if ~isempty(Tw_face) && ~isempty(Ts_face) && numel(Tw_face) == numel(flux)
        heff = flux ./ (Tw_face - Ts_face);
        heff(abs(Tw_face - Ts_face) < 1e-12) = NaN;
    else
        hc = 1/R_c;
        heff = hc * ones(size(flux));
    end

    hc_nom = 1/R_c;

    figure('Name',['h_{eff}(t) — ', caseX.label]); hold on; box on; grid on;
    plot(times, heff, '-', 'LineWidth', 1.6, 'Color', [0.2 0.6 0.2], 'DisplayName', 'Explicit h_{eff}');
    yline(hc_nom, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Nominal 1/R_c');
    if seed_time > 0
        xline(seed_time, 'k:', 'LineWidth', 1.0, 'DisplayName', 'Seed time');
    end

    xlabel('time  t  [s]');
    ylabel('effective conductance  h_{eff}  [W m^{-2} K^{-1}]');
    title(['Explicit contact conductance — ', caseX.label]);
    legend('Location','Best');
    xlim([0, max(t_max, times(end))]);
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
