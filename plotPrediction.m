function plotPrediction(m,y,pred)


close
plot([1:1:m],y,'ko', 'MarkerFaceColor', 'b','LineWidth', 0, 'MarkerSize', 4);
hold on;
plot([1:1:m],pred,'ko', 'MarkerFaceColor', 'r','LineWidth', 0, 'MarkerSize', 4);

xlabel('time');
ylabel('bitcoin price');
legend('real price','prediction');
end