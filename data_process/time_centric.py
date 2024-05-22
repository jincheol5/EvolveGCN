import os
import sqlite3
from tqdm import tqdm
from temporal_graph import EdgeEvent

current_directory_path=os.getcwd() # 현재 작업 경로
parent_directory_path=os.path.dirname(current_directory_path) # 현재 작업 경로의 부모 작업 경로 
db_path = os.path.join(parent_directory_path, "db") # db파일 상대 경로 

class TimeCentric:
    
    def __init__(self,file_name):
        
        self.file_name=file_name
        self.db_path=db_path
        
    def compute(self,start_time,end_time):
        
        con=sqlite3.connect(self.db_path+"\\"+self.file_name+".db") # connect temporal graph db 
        cur=con.cursor()
        
        # get vertices
        vertex_rows=cur.execute("select * from Vertex").fetchall()
        vertex_list=[]
        for row in vertex_rows:
            vertex_list.append(row[0])
        
        # get edge events
        edge_event_rows=cur.execute("select * from EdgeEvent where time > "+str(start_time)+" and time <= "+str(end_time)+" order by time asc").fetchall()
        edge_event_list=[]
        for row in edge_event_rows:
            edge_event_list.append(EdgeEvent(row[0],row[1],row[2],row[3]))
            
        
        # create gamma table 
        table_name="GammaTable"+"_"+str(start_time)+"_"+str(end_time)
        cur.execute("drop table if exists "+table_name)
        cur.execute("create table if not exists "+table_name+"(id text primary key, source_vertex_id, target_vertex_id, tR_time bigint, edge_event_list text);")
        

        # compute temporal path
        for source in tqdm(vertex_list):
            

            gamma_table={}
            gamma_table[source]=[]
            check_table={}
            check_table[source]=start_time

            for event in edge_event_list:
                if (event.source_vertex_id in gamma_table) and (event.target_vertex_id not in gamma_table):
                    if check_table[event.source_vertex_id]!=event.time:
                        gamma_table[event.target_vertex_id]=gamma_table[event.source_vertex_id]+[event]
                        check_table[event.target_vertex_id]=event.time
                        
            # update gamma_table to db
            for key in gamma_table.keys():
                if key!=source:
                    edge_event_list_str=""
                    for event in gamma_table[key]:
                        edge_event_list_str+=event.id
                        edge_event_list_str+=","
                    edge_event_list_str=edge_event_list_str[:-1]

                    # update to db
                    cur.execute("insert or ignore into "+table_name+" values(:id, :source_vertex_id, :target_vertex_id, :tR_time, :edge_event_list);",
                                {"id":source+"_to_"+key,"source_vertex_id":source,"target_vertex_id":key,"tR_time":gamma_table[key][-1].time,"edge_event_list":edge_event_list_str})

                    
                    
        con.commit() # commit
        
        cur.close()
        con.close()