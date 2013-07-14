% read the first input into a feature matrix X
fileId = fopen('q2x.dat');
fileX = fscanf(fileId, '%f');
fclose(fileId);

% number of training inputs
m = size(fileX,1);
X = [ones(m,1) fileX];

% read the second input into the output vector y
fileId = fopen('q2y.dat');
y = fscanf(fileId, '%f');
fclose(fileId);

% standard linear regression
theta = (X'*X)\(X'*y);

scatter(X(:,2),y,'o');

hold all;

t=-6:0.1:12;
line = theta(1)+theta(2)*t;
plot(t,line);

% locally weighted linear regression

bandwidth = 0.8;
line2 = zeros(1,size(t,2));
index = 1;

for x=t,
    
    weights = zeros(1,m);
    for i=1:m,
       weights(i) = exp(-((x-X(i,2))^2)/(2*bandwidth^2));
    end
    
    W = diag(weights);
    theta = (X'*W*X)\(X'*W*y);
    line2(index) = theta'*[1;x];
    
    index = index + 1;
end

plot(t,line2);

hold off;
