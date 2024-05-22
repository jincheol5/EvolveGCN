class Edge:
    def __init__(self,id,source_id,target_id):
        self.id=id
        self.source_vertex_id=source_id
        self.target_vertex_id=target_id
    
class VertexEvent:
    def __init__(self,id,vertex_id,time):
        self.id=id
        self.vertex_id=vertex_id
        self.time=time
        
class EdgeEvent:
    def __init__(self,id,source_vertex_id,target_vertex_id,time):
        self.id=id
        self.source_vertex_id=source_vertex_id
        self.target_vertex_id=target_vertex_id
        self.time=time
    