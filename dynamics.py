class Generator:
    """ Implements each generation/load node

        Methods:
            modify_z : Increments/decrements the value of the control action.
            get_z: Getter of the control action z.
    """

    def __init__(self, z, alpha=1):
        """Constructor of Node class.

            Args:
                z (float): Initial control action of a given generator.
                alpha (float): Cost factor associated to a given generator.
        """

        self.z = z
        self.alpha = alpha
        
    def modify_z(self, delta_z):
        self.z += delta_z
        
        if self.z < 0.0:
            self.z = 0.0
            
        if self.z > 5.0:
            self.z = 5.0
        
    def get_z(self):
        return self.z


class Area:
    """ Implements each area frequency conditions in Secondary Control

        Methods:
            set_load: Setter of the area load.
            set_generation: Setter of the area generation.
            calculate_delta_f: Calculates the increase/decrease of the network frequency.
            calculate_p_g: Calculates new generation given the update of the control action.
            get_delta_f: Getter of the variation of the frequency.
            get_frequency: Getter of the frequency of the area.
            get_load: Getter of the load of the area.
            get_generation: Getter of the generation in the area.
    """

    def __init__(self, f_set_point, m, d, t_g, r_d):
        """Constructor of Area class.

            Args:
                f_set_point (float): Frequency set point of the network.
                m (float): inertia constant of the system.
                d (float): damping coefficient.
                t_g (float): time constant.
                r_d (float): droop.
        """

        self.f = f_set_point
        self.delta_f = 0
        self.m = m
        self.d = d
        self.t_g = t_g
        self.r_d = r_d

        self.p_l = 0
        self.p_g = 0
        self.delta_f = 0
        
    def set_load(self, p_l):
        self.p_l = p_l
        
    def set_generation(self, p_g):
        self.p_g = p_g
    
    def calculate_delta_f(self):
        self.delta_f += (self.p_g - self.p_l - self.d*self.delta_f)/self.m
        
    def calculate_p_g(self, z):
        self.p_g += (-self.p_g + z - (1/self.r_d)*self.delta_f)/self.t_g
        
    def get_delta_f(self):
        return self.delta_f
    
    def get_frequency(self):
        return self.f + self.delta_f
    
    def get_load(self):
        return self.p_l
    
    def get_generation(self):
        return self.p_g
