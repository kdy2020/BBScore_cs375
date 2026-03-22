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
    results(m).sbj = subject_layer_scores;
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

font_size_axis  = 14; % 
font_size_label = 18; % x, y
font_size_title = 20;
font_size_legend = 13;

% --- Figure 1: Profile ---
fig1 = figure('Color', 'w', 'Position', [100, 100, 900, 600]); hold on;

for m = 1:num_models
    x_v = results(m).x'; 
    m_v = results(m).mean'; 
    s_v = results(m).sem';
    
    legend_text = sprintf('%s (AUC: %.4f)', results(m).name, results(m).auc);
    
    fill([x_v, fliplr(x_v)], [(m_v+s_v), fliplr(m_v-s_v)], colors(m,:), ...
        'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    plot(x_v, m_v, 'Color', colors(m,:), 'LineWidth', 3, 'DisplayName', legend_text);

    plot(results(m).rel_depth, results(m).peak_mag, 'p', 'MarkerSize', 15, ...
        'MarkerFaceColor', colors(m,:), 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
end


set(gca, 'FontSize', font_size_axis, ...  
         'Box', 'off', ...              
         'XGrid', 'off', 'YGrid', 'off', ...
         'LineWidth', 1.2);          

title('Layer-wise Brain Alignment Profile', 'FontSize', font_size_title); 
xlabel('Relative Layer Depth', 'FontSize', font_size_label); 
ylabel('Normalized Predictivity (r/ceiling)', 'FontSize', font_size_label);

ylim([0 1.0]); 
legend('Location', 'northeast', 'FontSize', font_size_legend, 'Box', 'off');

saveas(fig1, fullfile(base_dir, 'Figure1_Final_Profile_with_AUC.png'));

%% --- Figure 2: Relative Peak Depth with SE ---
fig2 = figure('Color', 'w', 'Position', [100, 100, 600, 500]); hold on;

all_depths = zeros(num_subjects, num_models);
for m = 1:num_models
    [~, p_idx] = max(results(m).sbj, [], 2); % peak index
    all_depths(:, m) = (p_idx - 1) / (size(results(m).sbj, 2) - 1); % relative depth 
end

mean_depths = mean(all_depths, 1);
se_depths = std(all_depths, 0, 1) / sqrt(num_subjects); % Standard Error

b2 = bar(mean_depths, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6);
for k = 1:num_models, b2.CData(k,:) = colors(k,:); end

errorbar(1:num_models, mean_depths, se_depths, 'k', 'linestyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);

set(gca, 'XTick', 1:num_models, ... 
         'XTickLabel', model_names,'XTickLabel', model_names, 'TickLabelInterpreter', 'none', ...
         'FontSize', font_size_axis, 'Box', 'off', ...
         'XGrid', 'off', 'YGrid', 'off');
ylabel('Relative Depth (0-1)', 'FontSize', font_size_label); 
title('Relative Peak Alignment Depth', 'FontSize', font_size_title);
ylim([0 1.0]); 

saveas(fig2, fullfile(base_dir, 'Figure2_Peak_Depth.png'));

%% --- Figure 3: Peak Magnitude with SE ---
fig3 = figure('Color', 'w', 'Position', [100, 100, 600, 500]); hold on;

all_mags = zeros(num_subjects, num_models);
for m = 1:num_models
    all_mags(:, m) = max(results(m).sbj, [], 2);
end

mean_mags = mean(all_mags, 1);
se_mags = std(all_mags, 0, 1) / sqrt(num_subjects); % Standard Error

b3 = bar(mean_mags, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6);
for k = 1:num_models, b3.CData(k,:) = colors(k,:); end

errorbar(1:num_models, mean_mags, se_mags, 'k', 'linestyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);

set(gca,'XTick', 1:num_models, ... 
         'XTickLabel', model_names, 'XTickLabel', model_names, 'TickLabelInterpreter', 'none', ...
         'FontSize', font_size_axis, 'Box', 'off', ...
         'XGrid', 'off', 'YGrid', 'off');
ylabel('Max Predictivity (r/Ceiling)', 'FontSize', font_size_label); 
title('Peak Alignment Magnitude', 'FontSize', font_size_title);
ylim([0 1.0]); 

saveas(fig3, fullfile(base_dir, 'Figure3_Peak_Magnitude.png'));
%%
fprintf('\n--- T-test Analysis for Figure 2 & 3 (N=8) ---\n');

% 1.(Peak Depth & Magnitude)
sbj_peak_depths = zeros(num_subjects, num_models);
sbj_peak_mags = zeros(num_subjects, num_models);

for m = 1:num_models
    [mags, idxs] = max(results(m).sbj, [], 2);
    sbj_peak_mags(:, m) = mags;
    % Relative Depth
    sbj_peak_depths(:, m) = (idxs - 1) / (size(results(m).sbj, 2) - 1);
end

% 2. T-test 
pairs = [1 2; 1 3; 2 3]; % GPT-8B, GPT-14B, 8B-14B
pair_names = {'GPT vs 8B', 'GPT vs 14B', '8B vs 14B'};

fprintf('\n[Figure 3: Peak Magnitude T-test]\n');
for i = 1:3
    [h, p] = ttest(sbj_peak_mags(:, pairs(i,1)), sbj_peak_mags(:, pairs(i,2)));
    fprintf('%s: p = %.4f %s\n', pair_names{i}, p, char(repmat('*',1,p<0.05)));
end

fprintf('\n[Figure 2: Peak Depth T-test]\n');
for i = 1:3
    [h, p] = ttest(sbj_peak_depths(:, pairs(i,1)), sbj_peak_depths(:, pairs(i,2)));
    fprintf('%s: p = %.4f %s\n', pair_names{i}, p, char(repmat('*',1,p<0.05)));
end
%% 5. Statistical Analysis: Participant-wise AUC & ANOVA (No Correction)
fprintf('\n--- Running Statistical Analysis (N=8) ---\n');
for m = 1:num_models
    results(m).auc_sbj = trapz(results(m).x, results(m).sbj, 2); 
    results(m).peak_sbj = max(results(m).sbj, [], 2);
end

% ANOVA
VarNames = {'GPT2', 'DSeek8B', 'DSeek14B'};
auc_table = array2table([results(1).auc_sbj, results(2).auc_sbj, results(3).auc_sbj], ...
    'VariableNames', VarNames);
Meas = table([1 2 3]', 'VariableNames', {'Models'});
rm_auc = fitrm(auc_table, 'GPT2-DSeek14B ~ 1', 'WithinDesign', Meas);
ranovatbl = ranova(rm_auc);
p_anova = ranovatbl.pValue(1); 

fprintf('Overall ANOVA p-value: %.4f\n', p_anova);

if p_anova < 0.05
    fprintf('Significant difference found. Running Post-hoc (Fisher''s LSD - No Correction)...\n');

    try
        posthoc = multcompare(rm_auc, 'Models', 'ComparisonType', 'lsd'); 
    catch
       
        posthoc = multcompare(rm_auc, 'Models'); 
        fprintf('Note: Default comparison used (Tukey-Kramer) as ''lsd'' failed.\n');
    end
    disp(posthoc);
else
    fprintf('No significant difference found at alpha=0.05.\n');
end

%% 6. Post-hoc tests and save it as Excel file
report_name = fullfile(base_dir, 'Statistical_Analysis_Results_LSD.xlsx');

if exist('posthoc', 'var')
    statsTable = posthoc;
  
    model_A_names = model_names(statsTable.Models_1)';
    model_B_names = model_names(statsTable.Models_2)';
    
    diff_rounded   = round(statsTable.Difference, 4);
    stderr_rounded = round(statsTable.StdErr, 4);
    p_rounded      = round(statsTable.pValue, 4);
    lower_rounded  = round(statsTable.Lower, 4);
    upper_rounded  = round(statsTable.Upper, 4);
    
    finalExcelTable = table(model_A_names, model_B_names, ...
        diff_rounded, stderr_rounded, p_rounded, ...
        lower_rounded, upper_rounded, ...
        'VariableNames', {'Model_A', 'Model_B', 'Difference', 'StdErr', 'pValue_Raw', 'Lower_95CI', 'Upper_95CI'});

    writetable(finalExcelTable, report_name, 'Sheet', 'PostHoc_AUC_Raw');
    
    try
        excelApp = actxserver('Excel.Application');
        workbook = excelApp.Workbooks.Open(report_name);
        sheet = workbook.Sheets.Item('PostHoc_AUC_Raw');
 
        sheet.Cells.Font.Name = 'Times New Roman';
        sheet.Cells.Font.Size = 11;

        dataRange = sheet.Range('C2:G7');
        dataRange.NumberFormat = '0.0000';
        
        sheet.Columns.AutoFit;
        sheet.Rows.Item(1).Font.Bold = true;
        
        workbook.Save;
        workbook.Close;
        excelApp.Quit;
        fprintf('Excel: "%s" saved!\n', report_name);
    catch
        fprintf('Excel sheet not saved.\n');
    end
end
