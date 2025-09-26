function plot_front_history(caseX)
%PLOT_FRONT_HISTORY Plot freezing-front trajectory and interface temperature residuals.

    if ~isstruct(caseX) || ~isfield(caseX,'num') || isempty(caseX.num)
        warning('plot_front_history:NoNumerics', ...
            'Case structure does not contain numerical results.');
        return;
    end

    num_struct = caseX.num;
    if isfield(num_struct, 'explicit') && isstruct(num_struct.explicit)
        snap = num_struct.explicit;
    else
        snap = num_struct;
    end

    if ~isstruct(snap) || ~isfield(snap,'front') || ~isstruct(snap.front)
        warning('plot_front_history:NoFrontHistory', ...
            'Explicit snapshot does not include front history.');
        return;
    end

    front = snap.front;
    if isfield(front,'t_phys') && ~isempty(front.t_phys)
        time = front.t_phys(:);
    elseif isfield(front,'t') && ~isempty(front.t)
        time = front.t(:);
        if isfield(snap,'t_offset')
            time = time + snap.t_offset;
        end
    else
        warning('plot_front_history:MissingTime', ...
            'Front history is missing time stamps.');
        return;
    end

    if isfield(front,'S') && ~isempty(front.S)
        S_vals = front.S(:);
    else
        warning('plot_front_history:MissingFront', ...
            'Front history is missing S(t) samples.');
        return;
    end

    p = caseX.params;
    lam = p.lam; alpha_s = p.alpha_s;
    S0_e = p.S0_e; t0_e = p.t0_e;
    S0_l = p.S0_l; t0_l = p.t0_l;

    Se = 2*lam*sqrt(alpha_s*(time + t0_e)) - S0_e;
    Sl = 2*lam*sqrt(alpha_s*(time + t0_l)) - S0_l;

    figure('Name',['Freezing front — ', caseX.label]);
    hold on; grid on; box on;
    yyaxis left;
    h_lines = gobjects(0);
    labels = {};
    h_lines(end+1) = plot(time, Se, 'k--','LineWidth',1.4);
    labels{end+1} = 'VAM S^{(0)}(t)';
    h_lines(end+1) = plot(time, Sl, 'b-' ,'LineWidth',1.4);
    labels{end+1} = 'VAM S^{(\infty)}(t)';
    h_lines(end+1) = plot(time, S_vals, '-', 'Color',[0.95 0.65 0.2], 'LineWidth',1.6);
    labels{end+1} = 'Explicit S_{num}(t)';

    if isfield(snap,'seed') && isstruct(snap.seed) && isfield(snap.seed,'time')
        seed_t = snap.seed.time;
        if seed_t > 0
            h_seed = xline(seed_t, 'Color',[0.4 0.4 0.4], 'LineStyle',':', 'LineWidth',1.0);
            h_lines(end+1) = h_seed;
            labels{end+1} = 'Seed time';
        end
    end

    xlabel('time t [s]');
    ylabel('Front position S(t) [m]');
    title(['Freezing front — ', caseX.label]);

    solid_delta = [];
    liquid_delta = [];
    if isfield(front,'solid_delta'), solid_delta = front.solid_delta; end
    if isfield(front,'liquid_delta'), liquid_delta = front.liquid_delta; end
    have_delta = ~isempty(solid_delta) || ~isempty(liquid_delta);
    h_delta = gobjects(0);
    labels_delta = {};
    if have_delta
        solid_delta = solid_delta(:);
        liquid_delta = liquid_delta(:);
        if any(isfinite(solid_delta)) || any(isfinite(liquid_delta))
            yyaxis right;
            ylabel('Interface \DeltaT [^{\circ}C]');
            if any(isfinite(solid_delta))
                h_delta(end+1) = plot(time, solid_delta, 'm:', 'LineWidth',1.2);
                labels_delta{end+1} = 'T_s(S)-T_f';
            end
            if any(isfinite(liquid_delta))
                h_delta(end+1) = plot(time, liquid_delta, 'c-.', 'LineWidth',1.2);
                labels_delta{end+1} = 'T_l(S)-T_f';
            end
            yyaxis left;
        end
    end

    all_handles = [h_lines(:); h_delta(:)];
    all_labels = [labels, labels_delta];
    if ~isempty(all_handles)
        legend(all_handles, all_labels, 'Location','NorthWest');
    end
end
