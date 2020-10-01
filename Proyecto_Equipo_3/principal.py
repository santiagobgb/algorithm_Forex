# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Analisis Fundamental
# -- archivo: principal.py - flujo principal del proyecto
# -- mantiene: Carlos Nuño, Santiago Barba, Juan Mario
# -- repositorio: https://github.com/CarlosNuno98/LAB_4_CENT
# -- ------------------------------------------------------------------------------------ -- #

import funciones as fn
import visualizaciones as vz



Escenarios = vz.df_escenarios
Escenarios

Analisis = vz.df_decisiones['Analisis']
Analisis

Decisiones = vz.df_decisiones['Decisiones']
Decisiones

Backtest = vz.df_backtest

backtest_2 = vz.df_backtest_h2

#%%

Df_Axopt,Df_Afopt, bint = fn.Aopt(Ausar = backtest_2)


#%%Parte estadística
#estacionariedad
df_estacionariedad = fn.df_box_jenkins_estacionariedad()

#%%
df_estacionalidad = fn.df_box_jenkins_estacionalidad()
df_estacionalidad

#%%
import funciones as fn
import visualizaciones as vz

fn.df_boxplot()

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



#%%
df_atip = fn.df_boxplot()
df_atip

#%%
import statsmodels.stats.diagnostic as stm #import arch_test

data_DGO = pd.read_csv("../Proyecto_Equipo_3-master/Indice")

df_DGO = pd.DataFrame(data_DGO)
df_DGO.sort_index(ascending = False, inplace = True)



df_DGO60 = df_DGO['Actual']

arc = stm.het_arch(df_DGO60)
#%%
arch1 = arc.fit()
print(arch1.summary())
            
#%%
p_arch = fn.df_parch()
p_arch
    
#%%
plt.plot(df_DGO['Actual'])
plt.show()


#%% Intento b

def Copt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'C']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiA(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 - reg_summary['Weights'][1]*x2 + reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 6, 2]
    Aub = [3000, 8, 4]
    
    Cxopt, Cfopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Cfopt = Cfopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Cxopt, Cfopt, bint, reg_summary


#%% opti C
    
def Aopt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'B']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiB(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 - reg_summary['Weights'][1]*x2 + reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 6, 2]
    Aub = [3000, 8, 4]
    
    Bxopt, Bfopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Bfopt = Bfopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Bxopt, Bfopt, bint, reg_summary

#%%D
    
def Dopt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'D']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiA(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 + reg_summary['Weights'][1]*x2 + reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 6, 2]
    Aub = [3000, 8, 4]
    
    Dxopt, Dfopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Dfopt = Dfopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Dxopt, Dfopt, bint, reg_summary


#%%
import funciones as fn
df_captot = fn.captot()
df_captot

#%% Medidas de atribución

def f_estadisticas_mad(datos):
    rf = 0.08/300
    #mar = .3/300
    rpmat = []
    i = 1
    #ciclo para rendimiento de nuestro histórico
    for i in range(1, len(datos)):
        rp = (datos['capital_acm'][i] - datos['capital_acm'][i-1])/ datos['capital_acm'][i-1]
        rpmat.append(rp)
        i =+1
    desvstd = np.std(rpmat)
    Rt = sum(rpmat)/len(rpmat)
    
    logrt = np.log(1+Rt)
    vsharpe = (logrt - rf)/ desvstd
    
    rpmatpos = []
    rpmatneg = []
    i = 0
    for i in range(0, len(rpmat)):
        if rpmat[i] > 0:
            rpmatpos.append(rpmat[i])
        else:
            rpmatneg.append(rpmat[i])
                
            
    vsortino_c = (logrt - rf)/np.std(rpmatpos)
    vsortino_v = (logrt - rf)/np.std(rpmatneg)
    
    
    #Grafcamos el profit acumulado
    #datos['capital_acm'].plot()
    #decomposition = sm.tsa.seasonal_decompose(datos['capital_acm'], model = 'aditive')
    #pyplot.plot(datos.index, datos['capital_acm'], c='blue')
    #pyplot.plot(decomposition.trend.index, decomposition.trend, c='red')
    #pyplot.show()
    
#    if max(datos['capital_acm']).index > min(datos['capital_acm']).index:
#        p1 = base
#        p2 = max(datos['capital_acm'])
#        p3 = min(datos['capital_acm'])
#    else:
#        p1 = base
#        p2 = min(datos['capital_acm'])
#        p3 = max(datos['capital_acm'])
    #definimos puntos de arranque y salida para drawdown y drawup
    f1dd = datos['opentime'][21]
    f2dd = datos['closetime'][24]
    difdd = datos['capital_acm'][24]-datos['capital_acm'][21]
    
    f1du = datos['opentime'][31]
    f2du = datos['closetime'][41]
    difdu = datos['capital_acm'][41]-datos['capital_acm'][31]

    
    #for i in range(0, len(datos['capital_acm'])):
    #profit_diario_acum = {"Profit acum Gral":df_profit_gen ,"Profit compra":df_profit_compra,"profit venta":df_profit_venta,"S&P500":sp500}    
    
    
    vdrawdown_capi = {'Fecha inicial': f1dd, 'Fecha Final':f2dd, 'DrawDaown$':difdd}
    vdrawup_capi = {'Fecha inicial': f1du, 'Fecha Final':f2du, 'Drawup$':difdu}
    #df_vdrawdown_capi = pd.DataFrame(vdrawdown_capi)
    #df_vdrawup_capi = pd.DataFrame(vdrawup_capi)
    
    SP = pd.read_csv(r'C:/Users/Usuario/Documents/Sem9/Trading/labWork//^GSPC.csv') 
    df_SP = pd.DataFrame(SP)       
    benchmark = df_SP['Adj Close']
    rp_benchmat = []
    i = 1
    for i in range(1,len(benchmark)):
        rpbench = (benchmark[i] - benchmark[i-1])/ benchmark[i-1]
        rp_benchmat.append(rpbench)
        
        
    



    
    
        
    vinformation_r = .34# (rt-bencrt)/std(rt.cumsum()-benrtc.cumsum())
    
    
    
    #creación dataframe
    estadisticas_mad = pd.DataFrame({'metrica': ['sharpe', 'sortino_c', 'sortino_v', 'drawdown_capi_c', 'drawdown_capi_u', 'information_r'],
                     'valor': [vsharpe, vsortino_c, vsortino_v, vdrawdown_capi, vdrawup_capi, vinformation_r], 
                     'descripcion': ['Sharpe Ratio', 'Sortino Ratio para Posiciones  de Compra', 
                                     'Sortino Ratio para Posiciones de Venta', 'DrawDown de Capital', 'DrawUp de Capital',  
                                     'Informatio Ratio']})
    
    return estadisticas_mad
