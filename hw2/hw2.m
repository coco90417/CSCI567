sum_tp = sum(train_data_TP);
sum_tn = sum(train_data_TN);
sum_all = sum_tp + sum_tn;
[sort_sum_all, sort_sum_all_index] = sort(sum_all);

