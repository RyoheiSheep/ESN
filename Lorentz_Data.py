#################
###Lorentz Equation##
###################




import numpy as np
# from numba import jit 
# import signalz


class InputGenerator:

    def __init__(self, start_time, end_time, num_time_steps):
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps 

    def generate_sin(self, amplitude=1.0):
        return np.sin( np.linspace(self.start_time, self.end_time, self.num_time_steps) ) * amplitude

    # def generate_mackey_glass(self, a=0.2, b=1, c=0.9, d=17, e=10, initial=0.1):
    #     return (signalz.mackey_glass(self.num_time_steps+200, a=a, b=b, c=c, d=d, e=e, initial=initial) - 0.8)[200:]

    def generate_logistic(self,a=3.6, x0=0.52):
        s=np.linspace(self.start_time, self.end_time, self.num_time_steps)
        xn=np.array([x0])
        for t in range(len(s)-1):
            xn = np.append(xn, np.array([a*xn[-1]*(1-xn[-1])]))
        return xn
    
    # @jit
    def generate_lorentz(self):
        
        def lorenz(x, y, z, p=10, r=28, b=8/3):
            x_dot = -p*x +p*y
            y_dot = -x*z +r*x-y
            z_dot = x*y -b*z
            return np.array([x_dot, y_dot, z_dot])
    
        t=0             ##### 初期時間######
        dt = 0.01       ##### 微分間隔######
        stepCnt = int((self.end_time-self.start_time)/dt) ######時間ステップ##
        xs = np.zeros(stepCnt +1) 
        ys = np.zeros(stepCnt +1)
        zs = np.zeros(stepCnt +1)
                
        xs[0], ys[0], zs[0] = (0.1, 0.5, 1.2)
        
        for i in range(stepCnt):
            x,y,z=xs[i],ys[i],zs[i]
        
            k0 = dt * lorenz(x,y,z)
            k1 = dt * lorenz(x+k0[0]/2., y+k0[1]/2., z+k0[2]/2.)
            k2 = dt * lorenz(x+k1[0]/2., y+k1[1]/2., z+k1[2]/2.)
            k3 = dt * lorenz(x+k2[0], y+k2[1], z+k2[2])
        
            dx = (k0[0]+2.0*k1[0]+2.0*k2[0]+k3[0])/6.0
            dy = (k0[1]+2.0*k1[1]+2.0*k2[1]+k3[1])/6.0
            dz = (k0[2]+2.0*k1[2]+2.0*k2[2]+k3[2])/6.0
            
            xs[i+1] = xs[i] + dx
            ys[i+1] = ys[i] + dy
            zs[i+1] = zs[i] + dz
        
            # xs  = (xs- np.min(xs))/(np.max(xs)-np.min(xs))
            # ys  = (ys- np.min(ys))/(np.max(ys)-np.min(ys))
            # zs  = (zs- np.min(zs))/(np.max(zs)-np.min(zs))
        
        matrix = xs
        
        return matrix
    

def main():
    T = 4000
    dt = 0.001
    num_time_step = int(T/dt)


    GeneratingInput = InputGenerator(start_time= 0, end_time= T, num_time_steps= num_time_step)
    lorentz_data = GeneratingInput.generate_lorentz()
    
    np.savetxt('Lorentz_X.txt', lorentz_data)
    


if __name__ == "__main__":
    main()