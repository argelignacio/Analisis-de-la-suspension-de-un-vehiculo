import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, stem
from sympy import *

'Euler Explicito'

def metodoDeEulerExplicitoDerivada2(f,intervalo, paso):
    cantidad = intervalo/paso +1 
    u=np.zeros(int(cantidad),dtype=float)
    v=np.zeros(int(cantidad),dtype=float)
    u[0]=0
    v[0]=0
    i=0
    n=0
    while n<cantidad-1:
        u_actual = (u[(n)] + paso * v[n])
        u[(n+1)]= u_actual
        v_actual = v[n] + paso * f(u[n])
        v[n+1] = v_actual
        n+=1
        i+=paso
    return u

'Euler Implicito'

def metodoDeEulerImplicitoDerivada2(A_menos1,termino_indep,intervalo, paso):
    cantidad = intervalo/paso +1
    res = np.zeros(int(cantidad*2))
    res.shape=(2,int(cantidad))
    i=0
    n=0
    while n<cantidad-1:
        aux = np.array(A_menos1 @ res[:,n] + termino_indep)
        res[:,(n+1)]= aux
        n+=1
        i+=paso
    return res[1]

'Runge Kutta'

def rungeKuttaO2(f,intervalo, paso):
    cantidad = intervalo/paso +1 
    u=np.zeros(int(cantidad),dtype=float)
    v=np.zeros(int(cantidad),dtype=float)
    u[0]=0
    v[0]=0
    i=0
    n=0
    while n<cantidad-1:
        u_actual_predictor = (u[(n)] + paso * v[n])
        v_actual_predictor = (v[n] + paso * f(u[n]))
        u[(n+1)]= (u[n] + (paso/2) *(v[n] + v_actual_predictor) )
        v[n+1] = (v[n] + (paso/2) * (f(u[n]) + f(u_actual_predictor)) )
        n+=1
        i+=paso
    return u

def rungeKuttaO2ConExtras(f,intervalo, paso,extras):
    cantidad = intervalo/paso +1 
    u=np.zeros(int(cantidad),dtype=float)
    v=np.zeros(int(cantidad),dtype=float)
    u[0]=0
    v[0]=0
    i=0
    n=0
    while n<cantidad-1:
        u_actual_predictor = (u[(n)] + paso * v[n])
        v_actual_predictor = (v[n] + paso * f(u[n],extras[0],i,v[n],extras[1]))
        u[(n+1)]= (u[n] + (paso/2) *(v[n] + v_actual_predictor) )
        v[n+1] = (v[n] + (paso/2) * (f(u[n],extras[0],i,v[n],extras[1]) + f(u_actual_predictor,extras[0],i+paso,v_actual_predictor,extras[1])))
        n+=1
        i+=paso
    return u,v

def maximaCompresion(intervalo, h,extras,c,t,k,lam):
    def f2(y,c,t,y_prim,c_prim):
        m = 104351/200
        res =  (k/m)*(c(t)-y) + (lam/m) * (c_prim(t) - y_prim)
        return res
    
    paso = 0
    aproximada,v = rungeKuttaO2ConExtras(f2,intervalo,h,extras)
    minimo = 0
    for j in range (int((intervalo/h)+1)):
        aproximada[j] = (aproximada[j] - float(c(paso)))
        if aproximada[j] < minimo :
            minimo = aproximada[j]
        paso += h
    return minimo

def imprimirRungeKuttaO2(vfun, intervalo, h,extras,t,c,k,lam):
    x = np.arange(0,intervalo+h,h)
    y = vfun(x)
    plt.plot(x,y,'r')

    def f2(y,c,t,y_prim,c_prim):
        m = 104351/200
        res =  (k/m)*(c(t)-y) + (lam/m) * (c_prim(t) - y_prim)
        return res
    
    paso = 0
    aproximada,v = rungeKuttaO2ConExtras(f2,intervalo,h,extras)
    u = np.copy(aproximada)
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('y(t) RK2 k = '+str(k))
    name = 'RK2Amortiguado'+str(h)+'.png'
    #plt.savefig(name)
    #plt.show()
    minimo = 0
    for j in range (int((intervalo/h)+1)):
        aproximada[j] = (aproximada[j] - float(c(paso)))
        if aproximada[j] < minimo :
            minimo = aproximada[j]
        paso += h
    print('maxima compresion k = '+str(k)+ ' lambda = '+str(lam)+ '  :  '+ str(minimo))
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('compresion')
    plt.title('compresion(t) del amortiguador ')
    name = 'Compresion'+str(h)+'.png'
    #plt.savefig(name)
    #plt.show()
    return u,v

def imprimirEulerexplicito(f,intervalo,h,analitica,t):
    aproximada = metodoDeEulerExplicitoDerivada2(f,intervalo,h)
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('y(t) euler explicito con paso = ' + str(h))
    plt.plot(t,analitica(t) , 'r--')
    name = 'EulerExplicito'+str(h)+'.png'
    plt.savefig(name)
    plt.show()
    
    paso=0
    for j in range (int((intervalo/h)+1)):
        aproximada[j] = (analitica(paso) - aproximada[j])
        paso += h
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('e(t) euler explicito con paso = ' + str(h))
    name = 'error-EulerExplicito'+str(h)+'.png'
    plt.savefig(name)
    plt.show()

def imprimirEulerImplicito(A_menos1, termino_indep,intervalo,h,analitica,t):
    aproximada = metodoDeEulerImplicitoDerivada2(A_menos1,termino_indep,intervalo,h)
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('y(t) euler implicito con paso = ' + str(h))
    plt.plot(t,analitica(t) , 'r--')
    name = 'EulerImplicito'+str(h)+'.png'
    plt.savefig(name)
    plt.show()
    paso=0
    for j in range (int((intervalo/h)+1)):
        aproximada[j] = (analitica(paso) - aproximada[j])
        paso += h
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('e(t) euler implicito con paso = ' + str(h))
    name = 'error-EulerImplicito'+str(h)+'.png'
    plt.savefig(name)
    plt.show()

def RungeKuttaO2sinAmortiguar(f,intervalo, h,analitica,t):
    aproximada = rungeKuttaO2(f,intervalo, h)
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('y(t) RK2 con paso = ' + str(h))
    plt.plot(t,analitica(t) , 'r--')
    name = 'RK2'+str(h)+'.png'
    plt.savefig(name)
    plt.show()
    paso=0
    for j in range (int((intervalo/h)+1)):
        aproximada[j] = (analitica(paso) - aproximada[j])
        paso += h
    plt.plot(t,aproximada)
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('e(t) RK2 con paso = ' + str(h))
    name = 'error-RK2'+str(h)+'.png'
    plt.savefig(name)
    plt.show()

def sinAmortiguar():

    'Funcion sin amortiguacion'
    def f(x):
        k= 25000
        m = 104351/200
        c = 0.1
        return (k/m)*(c-x)

    'Solucion analitica sin amortiguacion'
    def analitica(t):
        k= 25000
        m = 104351/200
        c = 0.1
        return 0.1 - 0.1 * np.cos(((k/m)**0.5)*t)

    'Parametros para el problema sin amortiguacion'
    intervalo = 5
    h = 0.005
    k = 25000
    m = 104351/200
    lam = 0
    c= 0.1
    c_prim = 0

    'Matriz inversa para la solucion de euler implicito sin amortiguacion'
    divisor = (h**2)*k+h*lam+m
    A_menos1 = np.array([[m/divisor , -h*k/divisor],
                        [h*m/divisor , (h*lam+m)/divisor]])

    termino_indep = np.array([h*((k*c/m)+(lam*c_prim/m)),0])

    t=np.arange(0,intervalo+h,h)


    'Metodo de euler explicito sin amortiguar'
    h=0.005
    t=np.arange(0,intervalo+h,h)
    imprimirEulerexplicito(f,intervalo,h,analitica,t)
    
    h=0.01
    t=np.arange(0,intervalo+h,h)
    imprimirEulerexplicito(f,intervalo,h,analitica,t)

    h=0.001
    t=np.arange(0,intervalo+h,h)
    imprimirEulerexplicito(f,intervalo,h,analitica,t)

    h=0.0005
    t=np.arange(0,intervalo+h,h)
    imprimirEulerexplicito(f,intervalo,h,analitica,t)

    'Metodo de euler implicito sin amortiguar'

    h=0.005
    divisor = (h**2)*k+h*lam+m
    t=np.arange(0,intervalo+h,h)
    A_menos1 = np.array([[m/divisor , -h*k/divisor],
                        [h*m/divisor , (h*lam+m)/divisor]])

    termino_indep = np.array([h*((k*c/m)+(lam*c_prim/m)),0])
    imprimirEulerImplicito(A_menos1, termino_indep,intervalo,h,analitica,t)
    
    h=0.01
    divisor = (h**2)*k+h*lam+m
    A_menos1 = np.array([[m/divisor , -h*k/divisor],
                        [h*m/divisor , (h*lam+m)/divisor]])

    termino_indep = np.array([h*((k*c/m)+(lam*c_prim/m)),0])
    t=np.arange(0,intervalo+h,h)
    imprimirEulerImplicito(A_menos1, termino_indep,intervalo,h,analitica,t)

    h=0.001
    divisor = (h**2)*k+h*lam+m
    A_menos1 = np.array([[m/divisor , -h*k/divisor],
                        [h*m/divisor , (h*lam+m)/divisor]])

    termino_indep = np.array([h*((k*c/m)+(lam*c_prim/m)),0])
    t=np.arange(0,intervalo+h,h)
    imprimirEulerImplicito(A_menos1, termino_indep,intervalo,h,analitica,t)

    h=0.0005
    divisor = (h**2)*k+h*lam+m
    A_menos1 = np.array([[m/divisor , -h*k/divisor],
                        [h*m/divisor , (h*lam+m)/divisor]])

    termino_indep = np.array([h*((k*c/m)+(lam*c_prim/m)),0])
    t=np.arange(0,intervalo+h,h)
    imprimirEulerImplicito(A_menos1, termino_indep,intervalo,h,analitica,t)


    'Metodo de Runge Kutta sin amortiguar'

    h=0.005
    t=np.arange(0,intervalo+h,h)
    RungeKuttaO2sinAmortiguar(f,intervalo, h,analitica,t)
    
    h=0.01
    t=np.arange(0,intervalo+h,h)
    RungeKuttaO2sinAmortiguar(f,intervalo, h,analitica,t)

    h=0.001
    t=np.arange(0,intervalo+h,h)
    RungeKuttaO2sinAmortiguar(f,intervalo, h,analitica,t)

    h=0.0005
    t=np.arange(0,intervalo+h,h)
    RungeKuttaO2sinAmortiguar(f,intervalo, h,analitica,t)


def amortiguado():
    h = 0.005
    intervalo = 5
    lam = 750
    'c'
    def c(t):
        if t>=1 and t<1.1:
            return float(t - 1.0)
        elif (t>=1.1 and t<1.3):
            return float(0.1)
        elif (t>=1.3 and t<1.4 ):
            return float(1.4 - t )
        return float(0)

    vfun = np.vectorize(c)

    x = np.arange(0,intervalo+h,h)
    y = vfun(x)
    plt.plot(x,y,'r')

    t=np.arange(0,intervalo+h,h)

    'Derivada de c'
    def c_prim(t):
        if (t<1):
            return 0
        if (t<1.1 and t>1):
            return 1
        if (t<1.3 and t>1.1):
            return 0
        if (t<1.4 and t>1.3):
            return -1
        return 0

    'Funcion con amortiguacion'
    def f1(y,c,t,y_prim,c_prim):
        k= 25000
        m = 104351/200
        res =  (k/m)*(c(t)-y) + (lam/m) * (c_prim(t) - y_prim)
        return res

    extras=[c,c_prim]

    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,500)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,750)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,1000)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,1250)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,1500)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,1750)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,15000,2000)
    
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,500)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,750)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,1000)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,1250)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,1500)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,1750)
    imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,25000,2000)
    '''
    maximaComp = -0.05
    compFinal = -10000000
    minimaPonderacion = 10000000000
    lamElecto = 150
    kElecto = 25000
    lamV = np.arange(150,12000,50)
    kV = np.arange(2500,100000,1000)
    for i in range(int(lamV.shape[0])-1):
        for j in range(int(kV.shape[0])-1):
            act = 0
            act = maximaCompresion(intervalo, h,extras,c,t,int(kV[j]),int(lamV[i]))
            if act >= maximaComp:
                ponderacionActual = lamV[i]/750 + kV[j]/25000
                if ponderacionActual < minimaPonderacion :
                    minimaPonderacion = ponderacionActual
                    lamElecto= lamV[i]
                    
                    kElecto = kV[j]
                    compFinal = act
    print('K elegido = '+ str(kElecto) +' y lambda electo = ' + str(lamElecto) + 'con compresion = ' + str(compFinal))
    kElecto = 80500
    lamElecto = 4550
    t=np.arange(0,intervalo+h,h)
    def aceleracion(c,t,c_prim,h,u,v):
        k= 25000
        m = 104351/200
        res =  (k/m)*(c(t)-u) + (lam/m) * (c_prim(t) - v)
        return res
    u,v = imprimirRungeKuttaO2(vfun,intervalo,h,extras,t,c,kElecto,lamElecto)
    
    afun = np.vectorize(aceleracion)
    x = np.arange(0,intervalo+h,h)
    y = afun(c,x,c_prim,h,u,v)
    plt.xlabel('t')
    plt.ylabel('y\'\'(t)')
    plt.title('Aceleracion')
    plt.plot(x,y,'r')
    plt.show()
    '''

                

#sinAmortiguar()
amortiguado()




