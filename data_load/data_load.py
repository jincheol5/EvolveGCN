import sqlite3
import os

current_directory_path=os.getcwd() # 현재 작업 경로
parent_directory_path=os.path.dirname(current_directory_path) # 현재 작업 경로의 부모 작업 경로 
db_path = os.path.join(parent_directory_path, "db") # db파일 상대 경로 

class DataLoad:
    
    
    def __init__(self,file_name,file_path):
        self.file_name=file_name
        self.file_path=file_path
        self.db_path=db_path
    
    def load(self):
        file=open(self.file_path,'r')

        con=sqlite3.connect(self.db_path+"\\"+self.file_name+".db")
        cur=con.cursor()
        
        # vertex table
        cur.execute("drop table if exists vertex")
        cur.execute("create table if not exists Vertex(id text primary key);")
        
        # edge table
        cur.execute("drop table if exists edge")
        cur.execute("create table if not exists Edge(id text primary key, source_vertex_id text, target_vertex_id text);")
        
        # edge events table 
        cur.execute("drop table if exists edge_event")
        cur.execute("create table if not exists EdgeEvent(id text primary key, source_vertex_id text, target_vertex_id text, time bigint);")
        
        # time table
        cur.execute("drop table if exists time")
        cur.execute("create table if not exists Time(time bigint primary key);")
        
        while True:
            line=file.readline()
        
            if not line: break
            
            data=line.split()
            
            source_id=data[0]
            target_id=data[1]
            time=data[2]
            
            # input vertex
            cur.execute("insert or ignore into Vertex values(:id);",{"id":source_id})
            cur.execute("insert or ignore into Vertex values(:id);",{"id":target_id})
            
            # input edge
            cur.execute("insert or ignore into Edge values(:id, :source_vertex_id, :target_vertex_id);",{"id":source_id+'_'+target_id,"source_vertex_id":source_id,"target_vertex_id":target_id})
            
            # input edge event
            cur.execute("insert or ignore into EdgeEvent values(:id, :source_vertex_id, :target_vertex_id, :time);"
                        ,{"id":source_id+'_'+target_id+'_'+str(time),"source_vertex_id":source_id,"target_vertex_id":target_id,"time":time})
            
            # input time
            cur.execute("insert or ignore into Time values(:time);",{"time":time})
        
            
        con.commit() # commit update
    
        cur.close()
        con.close()
        
        file.close()