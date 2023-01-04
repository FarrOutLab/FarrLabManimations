from manim import *
import numpy as np

config["frame_rate"] = 15

class Binary:
    
    def __init__(self,
        m1=10.,m2=10.,
        alpha=50.,beta=30.,gamma=0.,
        frame_rate = 30.,omega = 1.
    ):
        """
        A class for representing binaries.

        Parameters
        ----------
        m1,m2: floats
            masses of 1st and 2nd star, repsectively
        inclination: float
            inclination angle of binary wrt viewer
            in degrees, between [0,90] deg.
        azimuth: float
            azimuthal angle of viewer in coordinates
            of the binary, between [0,360] deg.
        phase: float
            phase of the binary, between [0,360] deg.
        frame_rate: float
            frames per second
        omega: float
            rotations per second
        """
        self.m1 = m1
        self.m2 = m2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._omega = omega
        self.frame_rate = frame_rate
        self.precession='off'
        self._phase1 = self.gamma_rad
        self._D = np.identity(3)
        self._C = np.identity(3)
        self._B = np.identity(3)

        self._pos1_projected = []
        self._pos2_projected = []
        self._r = 1.
        

    @property
    def r(self):
        return self._r
    
    @property
    def omega(self):
        return self._omega

    @property
    def alpha_rad(self):
        return self.alpha*np.pi/180.

    @property
    def beta_rad(self):
        return self.beta*np.pi/180.

    @property
    def gamma_rad(self):
        return self.gamma*np.pi/180.

    @property
    def phase1(self):
        return self._phase1
    
    @phase1.setter
    def phase1(self,phase_rad):
        self._phase1 = phase_rad % 2*np.pi 
    
    @property
    def phase2(self):
        return self.phase1+np.pi
        
    def orbit(self):
        self._phase1 = self.phase1 + 2.*np.pi*self.omega/self.frame_rate

    @property
    def Dmatrix(self):
        self._D[0,0] = self._D[1,1] = np.cos(self.alpha_rad)
        self._D[1,0] = -np.sin(self.alpha_rad)
        self._D[0,1] = np.sin(self.alpha_rad)
        return self._D

    @property
    def Cmatrix(self):
        self._C[1,1] = self._C[2,2] = np.cos(self.beta_rad)
        self._C[2,1] = -np.sin(self.beta_rad)
        self._C[1,2] = np.sin(self.beta_rad)
        return self._C

    @property
    def Bmatrix(self):
        self._B[0,0] = self._B[1,1] = np.cos(self.gamma_rad)
        self._B[1,0] = -np.sin(self.gamma_rad)
        self._B[0,1] = np.sin(self.gamma_rad)
        return self._B
    
    @property
    def Amatrix(self):
       return np.matmul(np.matmul(self.Bmatrix,self.Cmatrix),self.Dmatrix)

    @property
    def pos1(self):
        """
        position of primary in XYZ coordinates of binary plane 
        """
        return (self.r*(self.m2/(self.m1+self.m2))
                * np.array([np.cos(self.phase1),np.sin(self.phase1),0])
        )

    @property
    def pos2(self):
        return (self.r*(self.m1/(self.m1+self.m2))
                * np.array([np.cos(self.phase2),np.sin(self.phase2),0])
        )
    
    def project(self):
        return np.matmul(self.Amatrix,self.pos1),np.matmul(self.Amatrix,self.pos2)

    def evolve(self,n_steps=1):
        for i in range(n_steps):
            self.orbit()
            pos1_proj,pos2_proj = self.project()
            self._pos1_projected.append(pos1_proj) 
            self._pos2_projected.append(pos2_proj)

    @property
    def pos1_projected(self):
        return np.array(self._pos1_projected)

    @property
    def pos2_projected(self):
        return np.array(self._pos2_projected) 
    
    
class BBHOrbit(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=90 * DEGREES)
        ax = Axes(x_range=[-5,5],y_range=[-5,5]).move_to(ORIGIN)
        #self.add(ax)
        fps = 15.
        length = 15.
        n_frames = int(length*fps)
        BH_scale = 1./50.
        dist_scale = 3.0
        m1s = [50.0]#,23.0,44.0]
        m2s = [25.0]#,17.4,24.0]
        mobs = {i: {'comp1':None,'comp2':None,'path1':None,'path2':None} for i in range(len(m1s))}
        for ii,m1,m2,ax in zip(range(len(m1s)),m1s,m2s,[ax]):
            omega = fps/(m1 + m2) 
            # create a Binary instance for the event
            binary = Binary(
                    m1 = m1,
                    m2 = m2,
                    alpha = 0.,
                    beta = 20.,
                    gamma = 90.,
                    omega = omega,
                    frame_rate = fps 
                    )
            # evolve the binary over one orbit
            binary.evolve(n_frames)
            
            z1s,x1s,y1s = (dist_scale*binary.pos1_projected[:,0],dist_scale*binary.pos1_projected[:,1],dist_scale*binary.pos1_projected[:,2])
            #path1_points = np.array([x1s,y1s,np.zeros_like(z1s)]).flatten().reshape(binary.pos1_projected.shape)
            z2s,x2s,y2s = (dist_scale*binary.pos2_projected[:,0],dist_scale*binary.pos2_projected[:,1],dist_scale*binary.pos2_projected[:,2])       
            #path2_points = np.array([x2s,y2s,np.zeros_like(z2s)]).flatten().reshape(binary.pos2_projected.shape)
            comp1 = Sphere(color=WHITE, fill_color=WHITE, radius=BH_scale*binary.m1, resolution=15).move_to(ax.coords_to_point(x1s[0],y1s[0],z1s[0]))
            comp2 = Sphere(color=WHITE, fill_color=WHITE, radius=BH_scale*binary.m2, resolution=15).move_to(ax.coords_to_point(x2s[0],y2s[0],z2s[0]))
            comp1.set_color(WHITE)
            comp2.set_color(WHITE)
            self.add(comp1,comp2)
            path1 = VMobject().set_points(ax.coords_to_point(dist_scale*binary.pos1_projected))
            path2 = VMobject().set_points(ax.coords_to_point(dist_scale*binary.pos2_projected))
            mobs[ii]['comp1'] = comp1
            mobs[ii]['comp2'] = comp2
            mobs[ii]['path1'] = path1
            mobs[ii]['path2'] = path2
        animations = AnimationGroup(*[AnimationGroup(*[MoveAlongPath(mobs[i][f'comp{j}'],mobs[i][f'path{j}'],rate_func=rate_functions.linear) for j in [1,2]]) for i in range(len(m1s))])
        self.play(animations, rate_func=rate_functions.linear, run_time=length)