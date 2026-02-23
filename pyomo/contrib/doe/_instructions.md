In `optimize_experiments()` method, we want to add an option to initialize the experiments.
1. Option 1: The users can pass the initialized experiments
2. Option 2: The users can pass the experiemtns and we will use a initialization scheme.
            - We will use Latin Hypercube sampling (LHS) to sample from each dimension. We will
            need how many samples the user want to use in each dimension. For example, 
            if the user want to design 2 experiments, and each experiments has 2 decision variables (x, y) defined
            by the Pyomo `experiment_inputs` suffix and the user wants to have 10 samples for each 
            dimension of x and y. 
           
            - then we will create 10 samples of x and 10 for y using LHS.
            Then we will compute the FIM at those points using `compute_FIM()` method.

            - After that, we will choose those points without replacement and and add the FIM.
            e.g., since we want to design 2 experiments, let's say we have sampled, x = 1, 2, y = -1, 1.
            then we will compute FIM for (1, -1), (1, 1), (2, -1) and (2, 1) points. Now, from these 4 points, 
            we need to choose 2 points (since we have 2 experiments) without replacement and add their FIM. 
            If prior_FIM is not None, we also need to add prior_FIM here.
            For example,
            FIM_total = FIM at (1, -1) + FIM at (1, 1) + prior_FIM
            The points should not repeat
            
            If the number of points is more than 10,000 we will warn the user. It would be a
            good idea if we can tell the user how long it may take for the initialization / evaluate all these points.

            - then we will compute the value of the objective using the `objective_option` in `DesignOfExperiments`
            class.  the combination of points, for which the objective is maximum (if objective option is determinat or pseudo_trace),
            or minimum (if objective option is trace), we will chose that point as the initial point.

            - Using these points as the initial point, we will update the `experiment_inputs()`
            suffix and solve the optimization problem. 

3. Add a test file using #rooney_biegler_multiexperiment.py
4. If you have any clarifying questions, go ahead and ask. 
5. Discuss edge cases and concerns.
            

