function plotcluster(train_data, output_label, name, filename)

h=figure;
gscatter(train_data(:,1), train_data(:,2), output_label);
xlabel('x1');
ylabel('x2');
title(name);
saveas(h, filename);



