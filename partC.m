%% BBScore Alignment Analysis Pipeline (Person C)
% Goal: Compute layer-wise alignment profiles, extract summary stats, and visualize.
% Inputs: Extracted hidden states/scores (DeepSeek-R1, Llama-3, GPT-2).
% Outputs: Relative Peak Depth, Peak Magnitude, AUC, and Comparison Plots.

clear; clc; close all;

%% 1. Parameters and Dummy Data Generation
% In the real scenario, load .mat or .csv files provided by Person A/B.
model_names = {'DeepSeek-R1-8B', 'Llama-3-8B', 'GPT-2-XL'};
num_layers = [32, 32, 48]; % Total layers for each model
num_folds = 5;             % Number of Cross-validation folds
noise_ceiling = 0.8;       % Hypothetical noise ceiling for normalization

% Generating dummy data for testing (simulating brain alignment scores)
% Structure: scores{model_idx} = (layers x folds)
scores = cell(1, length(model_names));

for m = 1:length(model_names)
    L = num_layers(m);
    % Simulate a brain-alignment curve using a Gaussian-like shape
    x = linspace(0, 1, L)';
    peak_loc = 0.35 + rand()*0.2; % Randomize peak location per model
    base_curve = exp(-(x - peak_loc).^2 / 0.04) * 0.65; 
    
    % Add Gaussian noise to each CV fold
    scores{m} = base_curve + randn(L, num_folds) * 0.04;
end

%% 2. Metrics Extraction and Normalization
results = struct();

for m = 1:length(model_names)
    raw_scores = scores{m};
    L = num_layers(m);
    
    % (1) Noise Ceiling Normalization (Predictivity / Noise Ceiling)
    norm_data = raw_scores / noise_ceiling;
    
    % (2) Compute Mean Profile and Standard Error (SEM) across folds
    mean_profile = mean(norm_data, 2);
    sem_profile = std(norm_data, 0, 2) / sqrt(num_folds);
    
    % (3) Extract Key Summary Statistics
    [peak_mag, peak_idx] = max(mean_profile);
    % Relative Peak Depth: Normalized position (0 = input, 1 = output)
    rel_peak_depth = (peak_idx - 1) / (L - 1); 
    % Area Under the Curve (AUC) using trapezoidal numerical integration
    area_under_curve = trapz(linspace(0, 1, L), mean_profile);
    
    % Store results in a structure
    results(m).name = model_names{m};
    results(m).layers = L;
    results(m).mean_curve = mean_profile;
    results(m).sem = sem_profile;
    results(m).peak_mag = peak_mag;
    results(m).rel_depth = rel_peak_depth;
    results(m).auc = area_under_curve;
end

%% 3. Visualization Pipeline
figure('Name', 'BBScore Layer-wise Analysis', 'Position', [100, 100, 1100, 500]);

% --- Subplot 1: Layer-wise Alignment Profiles ---
subplot(1, 2, 1); hold on;
colors = lines(length(model_names));

for m = 1:length(model_names)
    L = results(m).layers;
    % X-axis: Relative depth (0 to 1) to compare models of different depths
    x_axis = linspace(0, 1, L); 
    
    % Draw SEM Shading (Shaded Error Bars)
    fill([x_axis, fliplr(x_axis)], ...
         [results(m).mean_curve' + results(m).sem', fliplr(results(m).mean_curve' - results(m).sem')], ...
         colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Plot Main Average Alignment Curve
    plot(x_axis, results(m).mean_curve, 'Color', colors(m,:), 'LineWidth', 2.5, 'DisplayName', results(m).name);
    
    % Mark the Peak Location
    plot(results(m).rel_depth, results(m).peak_mag, 'p', 'MarkerSize', 10, ...
         'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', colors(m,:), 'HandleVisibility', 'off');
end

grid on; box on;
xlabel('Relative Layer Depth (Normalized 0-1)');
ylabel('Normalized Brain Predictivity (r / Noise Ceiling)');
title('Layer-wise Brain Alignment Profiles');
legend('Location', 'best', 'FontSize', 10);
ylim([0 1]);

% --- Subplot 2: Relative Peak Depth Comparison ---
subplot(1, 2, 2);
rel_depth_values = [results.rel_depth];
h_bar = bar(rel_depth_values, 'FaceColor', 'flat');
h_bar.CData = colors; % Set bar colors to match line plot

set(gca, 'XTickLabel', model_names, 'FontSize', 9);
ylabel('Relative Peak Depth (0=Early, 1=Late)');
title('Comparison of Peak Brain-Alignment Depth');
grid on;
ylim([0 1]); % Normalized scale

%% 4. Statistical Summary Output to Console
fprintf('\n===========================================================\n');
fprintf('                BBScore Analysis Summary\n');
fprintf('===========================================================\n');
fprintf('%-18s | %-12s | %-12s | %-8s\n', 'Model Name', 'Rel. Depth', 'Peak Mag.', 'AUC');
fprintf('-----------------------------------------------------------\n');
for m = 1:length(model_names)
    fprintf('%-18s | %-12.4f | %-12.4f | %-8.4f\n', ...
        results(m).name, results(m).rel_depth, results(m).peak_mag, results(m).auc);
end
fprintf('===========================================================\n');