function plotPrediction(m,y,pred)


close
plot([1:1:m],y,'ko', 'MarkerFaceColor', 'b','LineWidth', 1, 'MarkerSize', 4,'linestyle','--');
hold on;
plot([1:1:m],pred,'ko', 'MarkerFaceColor', 'r','LineWidth', 1, 'MarkerSize', 4,'linestyle','--');
hold on;

xlabel('time');
ylabel('bitcoin price');
legend('real price','predicted');
end