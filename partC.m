%% 0. Setup Paths and Initial Configurations
base_dir = '/scratch/users/doyeonk9/bbscore_public';
model_files = {'gpt2xl_lebel2023_mean_content_pooled.npz', ...
    'deepseek8b_lebel2023_mean_content_pooled.npz', ...
    'deepseek14b_lebel2023_mean_content_pooled.npz'};
model_names = {'GPT-2-XL', 'DeepSeek-8B', 'DeepSeek-14B'};
num_models = length(model_names);
num_folds = 5;
num_subjects = 8;
lambda_list = logspace(0, 5, 10);
colors = [0.00, 0.45, 0.74; 0.85, 0.33, 0.10; 0.93, 0.69, 0.13];

% 1. Environment and Parallel Setup
try
    np = py.importlib.import_module('numpy');
catch
    error('Ensure the correct Python environment with Numpy is active.');
end

% Load splithalf noise ceiling data (Contains UTS_xx and UTS_xx_valid)
ceil_data = load(fullfile(base_dir, 'lebel2023_ceiling_splithalf.mat'));

delete(gcp('nocreate'));
num_workers = 12;
pc = parcluster('local');
pc.NumWorkers = num_workers;
parpool(pc, num_workers);

% 2. Define Subject-Specific Story Lists (Manual Map)
stories_map = containers.Map();
uts_common_84 = {'adollshouse', 'adventuresinsayingyes', 'afatherscover', 'againstthewind', ...
    'alternateithicatom', 'avatar', 'backsideofthestorm', 'becomingindian', ...
    'beneaththemushroomcloud', 'birthofanation', 'bluehope', 'breakingupintheageofgoogle', ...
    'buck', 'catfishingstrangerstofindmyself', 'cautioneating', 'christmas1940', ...
    'cocoonoflove', 'comingofageondeathrow', 'exorcism', 'eyespy', 'firetestforlove', ...
    'food', 'forgettingfear', 'fromboyhoodtofatherhood', 'gangstersandcookies', ...
    'goingthelibertyway', 'goldiethegoldfish', 'golfclubbing', 'gpsformylostidentity', ...
    'hangtime', 'haveyoumethimyet', 'howtodraw', 'ifthishaircouldtalk', 'inamoment', ...
    'itsabox', 'jugglingandjesus', 'kiksuya', 'leavingbaghdad', 'legacy', 'life', ...
    'lifeanddeathontheoregontrail', 'lifereimagined', 'listo', 'mayorofthefreaks', ...
    'metsmagic', 'mybackseatviewofagreatromance', 'myfathershands', 'myfirstdaywiththeyankees', ...
    'naked', 'notontheusualtour', 'odetostepfather', 'onlyonewaytofindout', 'penpal', ...
    'quietfire', 'reachingoutbetweenthebars', 'shoppinginchina', 'singlewomanseekingmanwich', ...
    'sloth', 'souls', 'stagefright', 'stumblinginthedark', 'superheroesjustforeachother', ...
    'sweetaspie', 'swimmingwithastronauts', 'thatthingonmyarm', 'theadvancedbeginner', ...
    'theclosetthatateeverything', 'thecurse', 'thefreedomridersandme', 'theinterview', ...
    'thepostmanalwayscalls', 'theshower', 'thetiniestbouquet', 'thetriangleshirtwaistconnection', ...
    'threemonths', 'thumbsup', 'tildeath', 'treasureisland', 'undertheinfluence', ...
    'vixenandtheussr', 'waitingtogo', 'whenmothersbullyback', 'wheretheressmoke', ...
    'wildwomenanddancingqueens'};
stories_map('UTS01') = uts_common_84; stories_map('UTS02') = uts_common_84; stories_map('UTS03') = uts_common_84;
uts_04_list = {'adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', ...
    'buck', 'exorcism', 'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet', ...
    'howtodraw', 'inamoment', 'itsabox', 'legacy', 'myfirstdaywiththeyankees', 'naked', ...
    'odetostepfather', 'sloth', 'souls', 'stagefright', 'swimmingwithastronauts', ...
    'thatthingonmyarm', 'theclosetthatateeverything', 'tildeath', 'undertheinfluence', ...
    'wheretheressmoke'};
stories_map('UTS04') = uts_04_list;
uts_05_08 = [uts_04_list, {'life'}];
stories_map('UTS05') = uts_05_08; stories_map('UTS06') = uts_05_08;
stories_map('UTS07') = uts_05_08; stories_map('UTS08') = uts_05_08;

% 3. Regression Pipeline
results = struct();
for m = 1:num_models
    fprintf('\n=========================================\n');
    fprintf('[%d/%d] Processing Model: %s\n', m, num_models, model_names{m});
    
    np_data = np.load(fullfile(base_dir, model_files{m}), pyargs('allow_pickle', true));
    model_story_ids = string(cell(np_data.get('story_ids').tolist()));
    model_story_ids = lower(erase(model_story_ids, ".hf5")); 
    
    raw_features = double(np_data.get('hidden_states'));
    [~, L, ~] = size(raw_features);
    model_features_all = permute(raw_features, [2, 1, 3]);
    
    subject_layer_scores = zeros(num_subjects, L);
    
    for s = 1:num_subjects
        uts_name = sprintf('UTS%02d', s);
        brain_file = load(fullfile(base_dir, sprintf('brain_data_UTS%02d.mat', s)));
        brain_raw = brain_file.brain_responses{1}; 
        
        subj_story_names = string(stories_map(uts_name));
        [common_stories, model_idx, brain_idx] = intersect(model_story_ids, subj_story_names, 'stable');
        
        if isempty(common_stories), continue; end
        
        % --- Subject-specific Masking (UTS_xx_valid) ---
        raw_ceil = ceil_data.(uts_name);
        valid_field = sprintf('%s_valid', uts_name);
        
        % Load the mask from ceil_data if it exists
        if isfield(ceil_data, valid_field)
            subj_mask = ceil_data.(valid_field);
        else
            subj_mask = true(size(raw_ceil));
        end
        
        % Final mask: Ceiling threshold AND the valid voxel mask
        valid_mask = (raw_ceil > 0.15) & (subj_mask > 0);
        
        brain_clean = zscore(brain_raw(brain_idx, valid_mask));
        ceil_clean = raw_ceil(valid_mask);
        current_features = model_features_all(:, model_idx, :);
        [num_samples, num_voxels] = size(brain_clean);
        
        fprintf('    -> %s: Matched = %d | Voxels = %d | Median Ceiling = %.3f\n', ...
                uts_name, length(common_stories), num_voxels, median(ceil_clean));
        
        cv_indices = crossvalind('Kfold', num_samples, num_folds);
        layer_scores_for_subj = zeros(L, 1);
        lambda_list_local = lambda_list; 
        
        parfor l = 1:L
            X_full = squeeze(current_features(l, :, :));
            X = zscore(X_full);
            row_norms = sqrt(sum(X.^2, 2));
            row_norms(row_norms == 0) = 1;
            X = X ./ row_norms;
            
            Y_pred_all = zeros(num_samples, num_voxels);
            Y_full_std = zscore(brain_clean);
            
            for f = 1:num_folds
                test_idx = (cv_indices == f);
                train_idx = ~test_idx;
                
                X_tr = X(train_idx, :); Y_tr = brain_clean(train_idx, :);
                X_te = X(test_idx, :);  Y_te = brain_clean(test_idx, :);
                
                [U, S, V] = svd(X_tr, 'econ');
                s_vals = diag(S);
                Y_te_std = zscore(Y_te);
                
                best_lam_r = -inf;
                best_Y_te_pred = zeros(size(Y_te));
                
                for lam = lambda_list_local
                    d = s_vals ./ (s_vals.^2 + lam);
                    curr_Y_pred = X_te * (V * (diag(d) * (U' * Y_tr)));
                    curr_Y_pred_std = zscore(curr_Y_pred);
                    r_val = mean(mean(curr_Y_pred_std .* Y_te_std, 1, 'omitnan'), 'omitnan');
                    
                    if r_val > best_lam_r
                        best_lam_r = r_val;
                        best_Y_te_pred = curr_Y_pred;
                    end
                end
                Y_pred_all(test_idx, :) = best_Y_te_pred;
            end
            
            Y_pred_all_std = zscore(Y_pred_all);
            r_voxels = mean(Y_pred_all_std .* Y_full_std, 1, 'omitnan');
            layer_scores_for_subj(l) = mean(r_voxels ./ ceil_clean(:)', 'omitnan');
        end
        subject_layer_scores(s, :) = layer_scores_for_subj;
    end
    
    results(m).mean = mean(subject_layer_scores, 1)';
    results(m).sem = std(subject_layer_scores, 0, 1)' / sqrt(num_subjects);
    results(m).x = linspace(0, 1, L)';
    [results(m).peak_mag, p_idx] = max(results(m).mean);
    results(m).rel_depth = (p_idx - 1) / (L - 1);
    results(m).name = model_names{m};
    results(m).auc = trapz(results(m).x, results(m).mean);
end
save(fullfile(base_dir, 'final_analysis_results.mat'), 'results');

fprintf('\nSuccess! Results saved in %s\n', base_dir);

%% 4. Save and Finalize

set(0, 'DefaultFigureRenderer', 'painters');

% --- Figure 1: Profile (Legend with AUC) ---
fig1 = figure('Visible', 'off','Color', 'w', 'Position', [100, 100, 900, 600]); hold on;

for m = 1:num_models
    x_v = results(m).x'; 
    m_v = results(m).mean'; 
    s_v = results(m).sem';
    
    % 레전드에 표시될 텍스트 생성 (예: GPT-2-XL (AUC: 0.1234))
    legend_text = sprintf('%s (AUC: %.4f)', results(m).name, results(m).auc);
    
    % 오차 범위 (Shadow) - HandleVisibility off로 해서 레전드에서 제외
    fill([x_v, fliplr(x_v)], [(m_v+s_v), fliplr(m_v-s_v)], colors(m,:), ...
        'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % 메인 선 - DisplayName에 AUC 포함된 텍스트 적용
    plot(x_v, m_v, 'Color', colors(m,:), 'LineWidth', 2.5, 'DisplayName', legend_text);
    
    % 피크 지점 (별표) - HandleVisibility off
    plot(results(m).rel_depth, results(m).peak_mag, 'p', 'MarkerSize', 12, ...
        'MarkerFaceColor', colors(m,:), 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

title('Layer-wise Brain Alignment Profile'); 
xlabel('Relative Layer Depth'); 
ylabel('Normalized Predictivity (r/ceiling)');
ylim([0 1.0]); 
grid on; 
legend('Location', 'northeast', 'FontSize', 10); % 레전드 위치 및 폰트 조절

saveas(fig1, fullfile(base_dir, 'Figure1_Final_Profile_with_AUC.png'));

%% --- Figure 2: Relative Peak Depth  ---
fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 600, 500]);
b2 = bar([results.rel_depth], 'FaceColor', 'flat', 'EdgeColor', 'k');
for k = 1:num_models, b2.CData(k,:) = colors(k,:); end
set(gca, 'XTickLabel', model_names, 'TickLabelInterpreter', 'none');
ylabel('Relative Depth (0-1)'); title('Relative Peak Alignment Depth');
ylim([0 1.0]); grid on;
saveas(fig2, fullfile(base_dir, 'Figure2_Peak_Depth.png'));

% --- Figure 3: Peak Magnitude  ---
fig3 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 600, 500]);
b3 = bar([results.peak_mag], 'FaceColor', 'flat', 'EdgeColor', 'k');
for k = 1:num_models, b3.CData(k,:) = colors(k,:); end
set(gca, 'XTickLabel', model_names, 'TickLabelInterpreter', 'none');
ylabel('Max Predictivity (r/Ceiling)'); title('Peak Alignment Magnitude');
ylim([0 1.0]); grid on;
saveas(fig3, fullfile(base_dir, 'Figure3_Peak_Magnitude.png'));

fprintf('\nSuccess! All Figures (1, 2, 3) saved with AUC legends.\n');
