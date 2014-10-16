B=mnrfit(train_data,train_label);
tpihat=mnrval(B,train_data);
train_pred=zeros(size((train_label),1),1);
taccuracy=0;
for i=1:size((train_label),1)
train_pred(i)=find(tpihat(i,:)==max(tpihat(i,:)));
if train_pred(i)==train_label(i);
taccuracy=taccuracy+1;
end
end

train_accu=taccuracy/size((train_label),1);

npihat=mnrval(B,new_data);
new_pred=zeros(size((new_label),1),1);
naccuracy=0;
for i=1:size((new_label),1)
new_pred(i)=find(npihat(i,:)==max(npihat(i,:)));
if new_pred(i)==new_label(i);
naccuracy=naccuracy+1;
end
end

new_accu=naccuracy/size((new_label),1);

vpihat=mnrval(B,valid_data);
valid_pred=zeros(size((valid_label),1),1);
vaccuracy=0;
for i=1:size((valid_label),1)
valid_pred(i)=find(vpihat(i,:)==max(vpihat(i,:)));
if valid_pred(i)==valid_label(i);
vaccuracy=vaccuracy+1;
end
end

valid_accu=vaccuracy/size((valid_label),1);