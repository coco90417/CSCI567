function plotcluster(train_data, name, filename)

h=figure;
gscatter(train_data(:,2), train_data(:,3), train_data(:,1));
xlabel('x1');
ylabel('x2');
title(name);
saveas(h, filename);


