%% 0. Setup Paths and Initial Configurations
base_dir = '/scratch/users/doyeonk9/bbscore_public';
model_files = {'gpt2xl_lebel2023_mean_content_pooled.npz', ...
               'deepseek8b_lebel2023_mean_content_pooled.npz', ...
               'deepseek14b_lebel2023_mean_content_pooled.npz'};
model_names = {'GPT-2-XL', 'DeepSeek-8B', 'DeepSeek-14B'};
num_models = length(model_names);
num_folds = 5;
num_subjects = 8;

% Define distinct colors for three models
colors = [0 0.447 0.741;      % Blue
          0.466 0.674 0.188;  % Green
          0.929 0.694 0.125]; % Orange

% Define range for lambda optimization (Log-spaced)
lambda_list = logspace(0, 5, 10); 

clear results;
results = struct();

% 1. Environment Setup
try
    np = py.importlib.import_module('numpy');
catch
    error('Ensure the correct Python environment with Numpy is active.');
end

% Load splithalf noise ceiling data
ceil_data = load(fullfile(base_dir, 'lebel2023_ceiling_splithalf.mat'));

% Parallel pool setup (12 workers as requested)
delete(gcp('nocreate'));
num_workers = 12; 
pc = parcluster('local');
pc.NumWorkers = num_workers;
parpool(pc, num_workers);

% 2. Regression Pipeline
for m = 1:num_models
    fprintf('\n=========================================\n');
    fprintf('[%d/%d] Processing Model: %s\n', m, num_models, model_names{m});
    
    % Load model features from NPZ (Shape: Stories x Layers x Hidden)
    np_data = np.load(fullfile(base_dir, model_files{m}), pyargs('allow_pickle', true));
    % Hidden states shape is (84, L, D) - permute to (L, 84, D) for easier layer-wise looping
    raw_features = double(np_data.get('hidden_states')); 
    [num_stories, L, hidden_dim] = size(raw_features);
    model_features = permute(raw_features, [2, 1, 3]);
    
    subject_layer_scores = zeros(num_subjects, L); 
    
    for s = 1:num_subjects
        uts_name = sprintf('UTS%02d', s);
        brain_file = load(fullfile(base_dir, sprintf('brain_data_UTS%02d.mat', s)));
        brain_raw = brain_file.brain_responses{1}; 
        
        % Filter voxels based on Noise Ceiling (threshold > 0.15)
        raw_ceil = ceil_data.(uts_name);
        valid_mask = raw_ceil > 0.15; 
        
        % Clean brain data and normalize (Voxel-wise Z-score)
        brain_clean = zscore(brain_raw(:, valid_mask));    
        ceil_clean = raw_ceil(valid_mask);
        
        fprintf('    -> %s: Median Ceiling = %.4f | Analyzed Voxels = %d\n', ...
                uts_name, median(ceil_clean), sum(valid_mask));
        
        % Cross-validation indices based on story count
        cv_indices = crossvalind('Kfold', num_stories, num_folds);
        layer_scores_for_subj = zeros(L, 1);
        
        % Parallel processing of layers for speed
        parfor l = 1:L
            % Extract layer features and apply Z-scoring and L2 normalization
            X_full = squeeze(model_features(l, :, :)); 
            X = zscore(X_full); 
            row_norms = sqrt(sum(X.^2, 2));
            row_norms(row_norms == 0) = 1;
            X = X ./ row_norms;
            
            Y = brain_clean;
            Y_pred_all = zeros(size(Y)); 
            
            for f = 1:num_folds
                test_idx = (cv_indices == f);
                train_idx = ~test_idx;
                
                X_tr = X(train_idx, :); Y_tr = Y(train_idx, :);
                X_te = X(test_idx, :);  Y_te = Y(test_idx, :);
                
                [U, S, V] = svd(X_tr, 'econ');
                s_vals = diag(S);
                
                best_lam_r = -inf;
                best_Y_te_pred = zeros(size(Y_te));
                
                % Pre-normalize validation data (Once outside the lambda loop for efficiency)
                Y_te_std = zscore(Y_te); 
                
                for lam = lambda_list
                    d = s_vals ./ (s_vals.^2 + lam);
                    curr_Y_pred = X_te * (V * (diag(d) * (U' * Y_tr)));
                    
                    % [Optimization] Fast correlation calculation using
                    % vector operations directly instead of corr(A,B).
                    %Multiplying z-scored matrices is mathematically
                    %equivalent to diag(corr()) but hundreds of times faster and memory-efficient.
                    curr_Y_pred_std = zscore(curr_Y_pred);
                    r_val = mean(mean(curr_Y_pred_std .* Y_te_std, 1), 'omitnan');
                    
                    if r_val > best_lam_r
                        best_lam_r = r_val;
                        best_Y_te_pred = curr_Y_pred;
                    end
                end
                Y_pred_all(test_idx, :) = best_Y_te_pred;
            end
            
            % vectorized calculation for final results
            Y_std = zscore(Y);
            Y_pred_all_std = zscore(Y_pred_all);
            
            % Compute correlation per voxel (1 x NumVoxels)
            r_voxels = mean(Y_pred_all_std .* Y_std, 1); 
            
            % Force both to be column vectors using (:) to avoid dimension mismatch
            r_voxels = max(r_voxels(:), 0); 
            curr_ceil = ceil_clean(:);
            
            % Normalize by noise ceiling and average across voxels
            layer_scores_for_subj(l) = mean(r_voxels ./ curr_ceil, 'omitnan');
        end 
        subject_layer_scores(s, :) = layer_scores_for_subj;
    end
    
    % Store aggregate results
    results(m).mean = mean(subject_layer_scores, 1)';
    results(m).sem = std(subject_layer_scores, 0, 1)' / sqrt(num_subjects); 
    results(m).x = linspace(0, 1, L)';
    [results(m).peak_mag, p_idx] = max(results(m).mean);
    results(m).rel_depth = (p_idx - 1) / (L - 1);
    results(m).name = model_names{m};
    results(m).auc = trapz(results(m).x, results(m).mean);
end

% 3. Save Results and Figures
save(fullfile(base_dir, 'final_analysis_results.mat'), 'results');

% Plotting Configurations
set(0, 'DefaultFigureRenderer', 'painters');

% Figure 1: Layer-wise Profile Comparison
fig1 = figure('Visible', 'off'); hold on;
set(fig1, 'Color', 'w', 'Position', [100, 100, 900, 600]);
for m = 1:num_models
    x_vec = results(m).x(:)'; mean_vec = results(m).mean(:)'; sem_vec = results(m).sem(:)';
    % Shaded error area
    fill([x_vec, fliplr(x_vec)], [(mean_vec + sem_vec), fliplr(mean_vec - sem_vec)], ...
        colors(m,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % Main profile curve
    plot(x_vec, mean_vec, 'Color', colors(m,:), 'LineWidth', 3, ...
        'DisplayName', sprintf('%s (AUC: %.3f)', results(m).name, results(m).auc));
    % Highlight Peak
    plot(results(m).rel_depth, results(m).peak_mag, 'p', 'MarkerSize', 15, ...
        'MarkerEdgeColor', colors(m,:), 'MarkerFaceColor', colors(m,:), 'HandleVisibility', 'off');
end
title('Layer-wise Brain Alignment Comparison', 'FontSize', 14);
xlabel('Relative Layer Depth', 'FontSize', 12); 
ylabel('Normalized Predictivity (r / Ceiling)', 'FontSize', 12);
yline(1.0, '--k', 'Noise Ceiling', 'Alpha', 0.5, 'LineWidth', 1.5); 
grid on; box on; legend('Location', 'northeast'); ylim([0 1.1]);
hold off;
saveas(fig1, fullfile(base_dir, 'Figure1_Final_Profile.png'));

% Figure 2: Relative Peak Depth Summary
fig2 = figure('Visible', 'off'); set(fig2, 'Color', 'w');
b2 = bar([results.rel_depth], 'FaceColor', 'flat'); 
b2.CData = colors; set(gca, 'XTickLabel', model_names);
title('Relative Peak Alignment Depth'); ylabel('Relative Depth (0-1)'); ylim([0 1]); grid on;
saveas(fig2, fullfile(base_dir, 'Figure2_Final_PeakDepth.png'));

% Figure 3: Peak Magnitude Comparison
fig3 = figure('Visible', 'off'); set(fig3, 'Color', 'w');
b3 = bar([results.peak_mag], 'FaceColor', 'flat'); 
b3.CData = colors; set(gca, 'XTickLabel', model_names);
title('Peak Alignment Magnitude'); ylabel('Max Predictivity (r/Ceiling)'); ylim([0 1.1]); grid on;
saveas(fig3, fullfile(base_dir, 'Figure3_Final_PeakMag.png'));

fprintf('\nAnalysis complete. Check Figure1, 2, 3 and final_analysis_results.mat in the base directory.\n');
