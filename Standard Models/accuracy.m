function percentageAccuracy = accuracy(out,test_label)
r=out==test_label;
count=0;
len=length(test_label);
for l=1:len
    if (r(l)==1)
        count=count+1;
    end
end
percentageAccuracy=count/len*100;