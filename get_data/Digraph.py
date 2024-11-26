class Digraph():
    False_value = 99999999
    @classmethod
    def get_face(cls,path,index):
        res = []
        for i in range(len(path)):
            if i!=index:
                res.append(path[i])
        return res
    @classmethod
    def get_max(cls,m):
        if len(m)>0:
            return max(m)
        else:
            return -1
    @classmethod
    def add_two_column(cls,a,b):
        for item in a:
            if item in b:
                b.remove(item)
            else:
                b.add(item)
        return b
    @classmethod
    def combine(cls,a,b):
        res = []
        for i in range(len(a)-1):
            res.append(a[i])
        for i in range(len(b)):
            res.append(b[i])
        return res
    
    
    def __init__(self,V,WG,dim):
        self.V = V
        self.dim = dim
        self.P = []
        self.PW = []
        self.set_PW(WG)
        self.boundary_matrix = []
        self.set_boundary_matrix()
        self.diagram = {}
        
        
        
        
        
    def set_PW(self,WG):
        v_out = {} # arrows from the vertex
        for i in range(len(self.V)):
            v_out[self.V[i]] = []
        for item in WG:
            one = item[0]
            two = item[1]
            weight = item[2]
            v_out[one].append(item)
        # all the paths and associated weights
        P = []
        W = []
        for i in range(self.dim):
            P.append([])
            W.append([])
        
        # P0 is all the vertices
        for i in self.V:
            P[0].append([i])
            W[0].append(0)
        # P1 is all the arrows
        for item in WG:
            one = item[0]
            two = item[1]
            weight = item[2]
            P[1].append([one,two])
            W[1].append(weight)
            
        # Pn(n>1)
        for n in range(2,self.dim):
            for i in range(len(P[n-1])):
                item = P[n-1][i]
                for temp1 in v_out[item[-1]]:
                    new = Digraph.combine(item,temp1[0:2])
                    P[n].append(new)
                    W[n].append( max(temp1[2],W[n-1][i]) )
               
        
        # dictionary is used to fast search
        dict1 = dict([])
        for i in range(self.dim):
            for j in range(len(P[i])):
                self.P.append(P[i][j])
                self.PW.append(W[i][j])
                dict1[str(P[i][j])] = 1
                
        # get the associated complex of the path complex of digraph.
        for n in range(2,self.dim):
            for i in range(len(P[n])):
                temp1 = P[n][i]
                for no in range(len(temp1)):
                    face = Digraph.get_face(temp1,no)
                    if str(face) not in dict1:
                        self.P.append(face)
                        self.PW.append(Digraph.False_value)
                        dict1[str(face)] = 1
        
        
    def set_boundary_matrix(self):
        complex_path = []
        for i in range(len(self.P)):
            temp = [ self.P[i],self.PW[i] ]
            complex_path.append(temp)
        complex_path = sorted(complex_path,key=lambda x:(x[1]+len(x[0])*0.00001))
        graph = []
        complex1 = []
        graph_weight = []
        complex_weight = []
        for i in range(len(complex_path)):
            item = complex_path[i]
            complex1.append(item[0])
            complex_weight.append(item[1])
            if item[1]!=Digraph.False_value:
                graph.append(item[0])
                graph_weight.append(item[1])
        N = len(graph)
        self.P = complex1
        self.PW = complex_weight
        # dict1 is used for fast search
        dict1 = dict([])
        for i in range(len(complex1)):
            item = complex1[i]
            dict1[str(item)] = i
        
        for j in range(N):
            temp = set()
            temp1 = graph[j]
            if len(temp1)>1:
                for i in range(len(temp1)):
                    face = Digraph.get_face(temp1, i)
                    index = dict1[str(face)]
                    temp.add(index)
            self.boundary_matrix.append(temp)
            
    
    def get_persistence(self):
        inf_bar = dict([])
        finite_bar = []
        #barcode = dict([]) # 1 value for real bars, 0 value for false bars
        L = [] 
        for i in range(len(self.P)):
            L.append(-1)
        for j in range(len(self.boundary_matrix)):
            N = Digraph.get_max(self.boundary_matrix[j])
            while N!=-1 and L[N]!=-1:
                self.boundary_matrix[j] = Digraph.add_two_column(self.boundary_matrix[L[N]], self.boundary_matrix[j])
                N = Digraph.get_max(self.boundary_matrix[j])
            if N==-1:
                inf_bar[ str([self.P[j],self.PW[j]]) ] = 1
            elif L[N]==-1:
                L[N] = j
                
        # find all finite bars
        for j in range(len(self.boundary_matrix)):
            low_j = Digraph.get_max(self.boundary_matrix[j])
            if low_j!=-1 and (str( [self.P[low_j],self.PW[low_j]] ) in inf_bar):
                del inf_bar[ str( [self.P[low_j],self.PW[low_j]] ) ]
                if low_j<j:
                    finite_bar.append( [len(self.P[low_j])-1,self.PW[low_j],self.PW[j]] )
                
        for i in range(self.dim):
            self.diagram[str(i)] = []
        for item in inf_bar.items(): 
            temp = eval(item[0])
            self.diagram[str(len(temp[0])-1)].append([temp[1],-1])
        for item in finite_bar:
            temp = [item[1],item[2]]
            if item[2]>item[1]:
                self.diagram[str(item[0])].append(temp)


    



