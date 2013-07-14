function theta = q1

    % read the first input and reshape it into a feature matrix X
    fileId = fopen('q1x.dat');
    fileX = fscanf(fileId, '%f %f');
    fclose(fileId);

    % number of training inputs
    m = size(fileX,1)/2;
    X = [ones(m,1) reshape(fileX',2,[])'];

    % read the second input into the output vector y
    fileId = fopen('q1y.dat');
    y = fscanf(fileId, '%f');
    fclose(fileId);

    % initialize the solution
    theta = zeros(3,1);
    precision = 1e-10;
    converged = 0;

    while ~converged

        % calculate the gradient vector
        grad = zeros(3, 1);
        for j=1:3,
            for i=1:m,
               grad(j) = grad(j) + (y(i) - h(theta, X(i,:)')) * X(i,j);
            end
        end

        % calculate the Hessian matrix
        H = zeros(3, 3);
        for j=1:3,
            for i=1:3,
                for k=1:m,
                    hyp = h(theta, X(k,:)');
                    H(i,j) = H(i,j) - X(k,i) * X(k,j) * hyp * (1 - hyp);
                end
            end
        end
        
        diff = H\grad;
        theta = theta - diff;
        
        converged = all(abs(diff) < precision);
    end
    
    % plot zero-valued training inputs as x's
    zeroPoints = X((y==0),[2 3]);
    scatter(zeroPoints(:,1), zeroPoints(:,2), 'x');
    
    hold all;
    
    % plot one-valued training inputs as o's
    onePoints = X((y==1),[2 3]);
    scatter(onePoints(:,1), onePoints(:,2), 'o');
    
    % plot the dividing line between the groups
    t=0:0.1:8;
    line = -theta(1)/theta(3)-theta(2)/theta(3)*t;
    plot(t,line);
    
    hold off;
end

% hypothesis function for logistic regression
function y = h(th, x)
    y = 1 / (1 + exp(-th'*x));
end


