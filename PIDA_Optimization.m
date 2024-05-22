% Main script to optimize and benchmark using various algorithms
function optimization
    % List of benchmark functions
    benchmark_functions = {@sphere_function, @ackley_function, @rastrigin_function, @rosenbrock_function, @schwefel_function, @griewank_function};
    benchmark_names = {'Sphere', 'Ackley', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Griewank'};
    optimization_algorithms = {@ga_algorithm, @gwo_algorithm, @woa_algorithm, @hho_algorithm, @geo_algorithm};
    optimization_names = {'GA', 'GWO', 'WOA', 'HHO', 'GEO'};

    % Define the parameters for the system
    Ka = 10; Ta = 0.1; Ke = 1; Te = 0.4; Kg = 1; Tg = 1; Ks = 1; Ts = 0.01;

    % Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
    lb = [0, 0, 0, 0, 0, 0];
    ub = [2, 2, 2, 2, 2, 2];
    bounds = [lb; ub];

    % Store results for plotting
    results = struct();
    convergence_curves = struct();

    for i = 1:length(benchmark_functions)
        benchmark_func = benchmark_functions{i};
        benchmark_name = benchmark_names{i};

        for j = 1:length(optimization_algorithms)
            optimization_algorithm = optimization_algorithms{j};
            optimization_name = optimization_names{j};

            fprintf('Optimizing with %s using %s Function\n', optimization_name, benchmark_name);

            [best_params, best_score, convergence_curve] = optimization_algorithm(@(params) objective_function(params, benchmark_func, Ka, Ta, Ke, Te, Kg, Tg, Ks, Ts), bounds);

            fprintf('Optimized Parameters for %s with %s: [%s]\n', benchmark_name, optimization_name, num2str(best_params));
            fprintf('Best Score for %s with %s: %f\n', benchmark_name, optimization_name, best_score);

            results.(benchmark_name).(optimization_name) = best_params;
            convergence_curves.(benchmark_name).(optimization_name) = convergence_curve;
        end
    end

    % Plot results for each benchmark function
    for i = 1:length(benchmark_names)
        benchmark_name = benchmark_names{i};
        figure;
        hold on;
        for j = 1:length(optimization_names)
            optimization_name = optimization_names{j};
            best_params = results.(benchmark_name).(optimization_name);
            Ka_pida = best_params(1); Kp_pida = best_params(2); Ki_pida = best_params(3); Kd_pida = best_params(4); alpha = best_params(5); beta = best_params(6);

            PIDA = tf([Ka_pida Kd_pida Kp_pida Ki_pida], [1 alpha beta 0]);
            Amplifier = tf([Ka], [Ta 1]);
            Exciter = tf([Ke], [Te 1]);
            Generator = tf([Kg], [Tg 1]);
            Sensor = tf([Ks], [Ts 1]);

            OpenLoop = PIDA * Amplifier * Exciter * Generator;
            ClosedLoop = feedback(OpenLoop, Sensor);

            [response, time] = step(ClosedLoop, 0:0.01:10);
            plot(time, response, 'DisplayName', optimization_name);
        end
        xlabel('Time (s)');
        ylabel('Response');
        title(['Closed-Loop Step Response with ', benchmark_name, ' Function']);
        legend('show');
        grid on;
        hold off;
    end

    % Plot convergence curves for each benchmark function
    for i = 1:length(benchmark_names)
        benchmark_name = benchmark_names{i};
        figure;
        hold on;
        for j = 1:length(optimization_names)
            optimization_name = optimization_names{j};
            convergence_curve = convergence_curves.(benchmark_name).(optimization_name);
            plot(convergence_curve, 'DisplayName', optimization_name);
        end
        xlabel('Iterations');
        ylabel('Best Score');
        title(['Convergence Curve with ', benchmark_name, ' Function']);
        legend('show');
        grid on;
        hold off;
    end
end

% Benchmark Functions
function f = sphere_function(x)
    f = sum(x.^2);
end

function f = ackley_function(x)
    f = -20 * exp(-0.2 * sqrt(0.5 * sum(x.^2))) - exp(0.5 * sum(cos(2 * pi * x))) + 20 + exp(1);
end

function f = rastrigin_function(x)
    f = 10 * length(x) + sum(x.^2 - 10 * cos(2 * pi * x));
end

function f = rosenbrock_function(x)
    f = sum(100 * (x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2);
end

function f = schwefel_function(x)
    f = 418.9829 * length(x) - sum(x .* sin(sqrt(abs(x))));
end

function f = griewank_function(x)
    f = 1 + sum(x.^2 / 4000) - prod(cos(x ./ sqrt(1:length(x))));
end

% Objective function to minimize the integrated absolute error of the step response
function error = objective_function(params, benchmark_func, Ka, Ta, Ke, Te, Kg, Tg, Ks, Ts)
    Ka_pida = params(1); Kp_pida = params(2); Ki_pida = params(3); Kd_pida = params(4); alpha = params(5); beta = params(6);

    PIDA = tf([Ka_pida Kd_pida Kp_pida Ki_pida], [1 alpha beta 0]);
    Amplifier = tf([Ka], [Ta 1]);
    Exciter = tf([Ke], [Te 1]);
    Generator = tf([Kg], [Tg 1]);
    Sensor = tf([Ks], [Ts 1]);

    OpenLoop = PIDA * Amplifier * Exciter * Generator;
    ClosedLoop = feedback(OpenLoop, Sensor);

    [response, time] = step(ClosedLoop, 0:0.01:10);
    control_error = sum(abs(1 - response));
    benchmark_error = benchmark_func(params);

    error = control_error + benchmark_error;
end

% Grey Wolf Optimizer (GWO) Algorithm
function [best_position, best_score, convergence_curve] = gwo_algorithm(obj_function, bounds)
    num_agents = 5;
    max_iter = 100;
    dim = length(bounds(1,:));
    alpha_pos = zeros(1, dim);
    alpha_score = inf;
    beta_pos = zeros(1, dim);
    beta_score = inf;
    delta_pos = zeros(1, dim);
    delta_score = inf;

    positions = rand(num_agents, dim) .* (bounds(2,:) - bounds(1,:)) + bounds(1,:);
    convergence_curve = zeros(1, max_iter);

    for iter = 1:max_iter
        for i = 1:num_agents
            fitness = obj_function(positions(i, :));
            if fitness < alpha_score
                alpha_score = fitness;
                alpha_pos = positions(i, :);
            elseif fitness < beta_score
                beta_score = fitness;
                beta_pos = positions(i, :);
            elseif fitness < delta_score
                delta_score = fitness;
                delta_pos = positions(i, :);
            end
        end

        a = 2 - iter * (2 / max_iter);

        for i = 1:num_agents
            for j = 1:dim
                r1 = rand();
                r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * alpha_pos(j) - positions(i, j));
                X1 = alpha_pos(j) - A1 * D_alpha;

                r1 = rand();
                r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * beta_pos(j) - positions(i, j));
                X2 = beta_pos(j) - A2 * D_beta;

                r1 = rand();
                r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * delta_pos(j) - positions(i, j));
                X3 = delta_pos(j) - A3 * D_delta;

                positions(i, j) = (X1 + X2 + X3) / 3;
            end
        end

        convergence_curve(iter) = alpha_score;
    end

    best_position = alpha_pos;
    best_score = alpha_score;
end

% Genetic Algorithm (GA)
function [best_position, best_score, convergence_curve] = ga_algorithm(obj_function, bounds)
    convergence = [];
    options = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 100, 'PopulationSize', 50, ...
        'OutputFcn', @ga_outfun);
    [best_position, best_score] = ga(obj_function, 6, [], [], [], [], bounds(1,:), bounds(2,:), [], options);

    function [state, options, optchanged] = ga_outfun(options, state, flag)
        optchanged = false;
        if strcmp(flag, 'init')
            convergence = [];
        end
        if ~isempty(state.Best)
            convergence = [convergence; state.Best(end)];
        end
    end

    convergence_curve = convergence;
end

% Whale Optimization Algorithm (WOA)
function [best_position, best_score, convergence_curve] = woa_algorithm(obj_function, bounds)
    num_agents = 5;
    max_iter = 100;
    dim = length(bounds(1,:));
    positions = rand(num_agents, dim) .* (bounds(2,:) - bounds(1,:)) + bounds(1,:);
    leader_pos = zeros(1, dim);
    leader_score = inf;

    convergence_curve = zeros(1, max_iter);

    for iter = 1:max_iter
        for i = 1:num_agents
            fitness = obj_function(positions(i, :));
            if fitness < leader_score
                leader_score = fitness;
                leader_pos = positions(i, :);
            end
        end

        a = 2 - iter * (2 / max_iter);
        b = 1;
        l = rand() * 2 - 1;

        for i = 1:num_agents
            r1 = rand();
            r2 = rand();
            A = 2 * a * r1 - a;
            C = 2 * r2;

            p = rand();
            if p < 0.5
                if abs(A) < 1
                    D = abs(C * leader_pos - positions(i, :));
                    positions(i, :) = leader_pos - A * D;
                else
                    rand_pos = positions(randi([1 num_agents]), :);
                    D = abs(C * rand_pos - positions(i, :));
                    positions(i, :) = rand_pos - A * D;
                end
            else
                D_leader = abs(leader_pos - positions(i, :));
                positions(i, :) = D_leader * exp(b * l) * cos(2 * pi * l) + leader_pos;
            end
        end

        convergence_curve(iter) = leader_score;
    end

    best_position = leader_pos;
    best_score = leader_score;
end

% Harris Hawks Optimization (HHO)
function [best_position, best_score, convergence_curve] = hho_algorithm(obj_function, bounds)
    num_agents = 5;
    max_iter = 100;
    dim = length(bounds(1,:));
    positions = rand(num_agents, dim) .* (bounds(2,:) - bounds(1,:)) + bounds(1,:);
    best_position = zeros(1, dim);
    best_score = inf;

    convergence_curve = zeros(1, max_iter);

    for iter = 1:max_iter
        for i = 1:num_agents
            fitness = obj_function(positions(i, :));
            if fitness < best_score
                best_score = fitness;
                best_position = positions(i, :);
            end
        end

        E1 = 2 * (1 - iter / max_iter);

        for i = 1:num_agents
            E0 = 2 * rand() - 1;
            E = E1 * E0;
            q = rand();
            r = rand();

            if abs(E) >= 1
                rand_index = randi([1 num_agents]);
                X_rand = positions(rand_index, :);
                positions(i, :) = X_rand - r * abs(X_rand - 2 * r * positions(i, :));
            elseif q < 0.5
                if abs(E) < 0.5
                    positions(i, :) = best_position - E * abs(best_position - positions(i, :));
                else
                    positions(i, :) = best_position - E * abs(best_position - positions(i, :)) + randn(1, dim) * 0.01;
                end
            else
                D_best = abs(best_position - positions(i, :));
                positions(i, :) = D_best * exp(E * (rand() - 1)) * cos(E * 2 * pi) + best_position;
            end
        end

        convergence_curve(iter) = best_score;
    end

    best_position = best_position;
    best_score = best_score;
end

% Golden Eagle Optimizer (GEO)
function [best_position, best_score, convergence_curve] = geo_algorithm(obj_function, bounds)
    num_agents = 5;
    max_iter = 100;
    dim = length(bounds(1,:));
    positions = rand(num_agents, dim) .* (bounds(2,:) - bounds(1,:)) + bounds(1,:);
    best_position = zeros(1, dim);
    best_score = inf;

    convergence_curve = zeros(1, max_iter);

    for iter = 1:max_iter
        for i = 1:num_agents
            fitness = obj_function(positions(i, :));
            if fitness < best_score
                best_score = fitness;
                best_position = positions(i, :);
            end
        end

        for i = 1:num_agents
            r1 = rand();
            r2 = rand();
            r3 = rand();
            r4 = rand();
            if r1 < 0.5
                if r2 < 0.5
                    positions(i, :) = positions(i, :) + r3 * (best_position - abs(positions(i, :)));
                else
                    positions(i, :) = best_position - r3 * abs(positions(i, :) - best_position);
                end
            else
                if r2 < 0.5
                    positions(i, :) = positions(i, :) + r3 * (best_position - abs(positions(i, :))) + r4 * (positions(i, :) - best_position);
                else
                    positions(i, :) = positions(i, :) - r3 * (best_position - abs(positions(i, :))) - r4 * (positions(i, :) - best_position);
                end
            end
        end

        convergence_curve(iter) = best_score;
    end

    best_position = best_position;
    best_score = best_score;
end
