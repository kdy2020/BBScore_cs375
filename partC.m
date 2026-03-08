%% BBScore Final Analysis (Person C)
% Generates 4 separate figures: Profile Curves, Rel Depth, Peak Mag, and AUC Area

clear; clc; close all;

%% 1. Setup & Dummy Data Generation
model_names = {'DeepSeek-R1-8B', 'Llama-3-8B', 'GPT-2-XL'};
num_layers = [32, 32, 48]; 
num_folds = 5;
noise_ceiling = 0.82; 

results = struct();
% Colors for each model (Blue, Orange, Yellow)
colors = [0 0.447 0.741; 0.85 0.325 0.098; 0.929 0.694 0.125]; 

for m = 1:length(model_names)
    L = num_layers(m);
    
    % --- DUMMY DATA ---
    x_raw = linspace(0, 1, L)';
    peak_loc = 0.3 + (m*0.15); % Different peak locations
    raw_scores = (exp(-(x_raw - peak_loc).^2 / 0.08) * 0.5) + randn(L, num_folds)*0.02;
    
    % (A) Normalization
    norm_scores = raw_scores / noise_ceiling;
    mean_curve = mean(norm_scores, 2);
    sem_curve = std(norm_scores, 0, 2) / sqrt(num_folds);
    
    % (B) Extract metrics
    [p_mag, p_idx] = max(mean_curve);
    rel_depth = (p_idx - 1) / (L - 1); 
    auc_val = trapz(linspace(0, 1, L), mean_curve); 
    
    % Store
    results(m).name = model_names{m};
    results(m).x = linspace(0, 1, L)';
    results(m).mean = mean_curve; 
    results(m).sem = sem_curve;
    results(m).rel_depth = rel_depth;
    results(m).peak_mag = p_mag; 
    results(m).auc = auc_val;
end

%% --- FIGURE 1: Layer-wise Profile Curves (with Peak Stars) ---
% Includes: Curves, SEM shading, AUC shaded areas, Peak Stars, and AUC values
figure; hold on;
set(gcf, 'Color', 'w', 'Position', [100, 100, 850, 600]);

for m = 1:length(model_names)
    x_vec = results(m).x(:)';
    mean_vec = results(m).mean(:)';
    sem_vec = results(m).sem(:)';
    
    % 1. AUC Shaded Area (Very translucent)
    fill([x_vec, fliplr(x_vec)], [mean_vec, zeros(size(mean_vec))], ...
         colors(m,:), 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % 2. SEM Shading (Uncertainty across folds)
    fill([x_vec, fliplr(x_vec)], [(mean_vec + sem_vec), fliplr(mean_vec - sem_vec)], ...
         colors(m,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
     
    % 3. Main Alignment Curve
    % Note: AUC is included in the Legend here
    plot(x_vec, mean_vec, 'Color', colors(m,:), 'LineWidth', 3, ...
         'DisplayName', sprintf('%s (AUC: %.3f)', results(m).name, results(m).auc));
    
    % 4. Peak Marker (Star at the maximum alignment)
    plot(results(m).rel_depth, results(m).peak_mag, 'p', 'MarkerSize', 15, ...
        'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', colors(m,:), 'HandleVisibility', 'off');

    % 5. Text Annotation for AUC (Directly on the plot)
    % This stacks the AUC values in the upper left corner in their respective colors
    text(0.02, 1 - (m * 0.05), sprintf('AUC (%s): %.3f', results(m).name, results(m).auc), ...
         'Color', colors(m,:), 'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized');
end

% Styling Figure

title('Integrated Layer-wise Brain Alignment Profile', 'FontSize', 14);
xlabel('Relative Layer Depth (Normalized 0 \rightarrow 1)', 'FontSize', 12);
ylabel('Normalized Predictivity (r / Noise Ceiling)', 'FontSize', 12);

% Reference line for human performance
yline(1.0, '--k', 'Noise Ceiling', 'LabelVerticalAlignment', 'bottom', 'Alpha', 0.5);

grid on; box on;
legend('Location', 'northeast', 'FontSize', 10);
ylim([0 1.25]); % Increased slightly to fit text annotations
hold off;
%% --- FIGURE 2: Relative Peak Depth ---
figure;
set(gcf, 'Color', 'w', 'Position', [150, 150, 500, 400]);
b2 = bar([results.rel_depth], 'FaceColor', 'flat');
b2.CData = colors;
set(gca, 'XTickLabel', model_names);
title('2. Relative Peak Depth');
ylabel('Relative Depth (0=Input, 1=Output)'); ylim([0 1]); grid on;

%% --- FIGURE 3: Peak Magnitude ---
figure
set(gcf, 'Color', 'w', 'Position', [200, 200, 500, 400]);
b3 = bar([results.peak_mag], 'FaceColor', 'flat');
b3.CData = colors;
set(gca, 'XTickLabel', model_names);
title('3. Peak Alignment Magnitude');
ylabel('Max Brain-Alignment Score'); ylim([0 1]); grid on;

%% --- FIGURE 4: AUC Visualization (Filled Area under Profile) ---
figure; hold on;
set(gcf, 'Color', 'w', 'Position', [250, 250, 700, 500]);

for m = 1:length(model_names)
    x_vec = results(m).x(:)';
    mean_vec = results(m).mean(:)';
    
    % Fill the area under the curve
    fill([x_vec, fliplr(x_vec)], ...
         [mean_vec, zeros(size(mean_vec))], ...
         colors(m,:), 'FaceAlpha', 0.2, 'EdgeColor', colors(m,:), 'LineWidth', 1.5);
     
    % AUC Text labels
    text_y_pos = 0.9 - (m * 0.06); % Adjusted for visibility
    text(0.05, text_y_pos, sprintf('%s AUC = %.3f', results(m).name, results(m).auc), ...
         'Color', colors(m,:), 'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized');
end
title('4. Overall Alignment Area (AUC)');
xlabel('Relative Layer Depth'); ylabel('Normalized Predictivity');
grid on; hold off;