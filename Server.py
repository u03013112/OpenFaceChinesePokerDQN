import threading
from websocket_server import WebsocketServer                                    
import json
import numpy as np
import time

PORT=9001                                                                       

class WSS((threading.Thread)):
        def __init__(self):
                threading.Thread.__init__(self)
                server = WebsocketServer(PORT, "0.0.0.0")
                self.server = server
                server.set_fn_new_client(self.new_client)
                server.set_fn_client_left(self.client_left)
                server.set_fn_message_received(self.message_received)
                self.isStart = False #目前从简，客户端连入后就开始
                self.isRecvWaiting = False

        def run(self):
                self.server.run_forever()

        def sendMsg(self,msg):
                self.isRecvWaiting = True
                self.server.send_message_to_all(msg)

        # Called for every client connecting (after handshake)                          
        def new_client(self,client, server):                                                 
                print("New client connected and was given id %d" % client['id'])
                # server.send_message_to_all("Hey all, a new client has joined us")
                self.client = client #暂时不支持多连接，这里之后要加判断
                self.isStart = True #开始，院子操作，不加锁

        # Called for every client disconnecting
        def client_left(self,client, server):
                print("Client(%d) disconnected" % client['id'])

        # Called when a client sends a message                                          
        def message_received(self,client, server, message):
                self.recv = json.loads(message)
                self.isRecvWaiting = False
                

class Env:
        def __init__(self):
                self.actionSize = 15
                self.stateSize = 45
                self.wss = WSS()
                self.wss.start()
                while self.wss.isStart == False:
                        time.sleep(0.001)
                #这里在初始化阶段就阻塞了，直至客户端接入
        
        def reset(self):
                self.wss.sendMsg(json.dumps({'act':'reset'}))
                while self.wss.isRecvWaiting == True:
                        time.sleep(0.001)
                ob = np.array(self.wss.recv['ob'])
                return ob
        
        def step(self,action):
                self.wss.sendMsg(json.dumps({'act':"step",'action':int(action)}))
                while self.wss.isRecvWaiting == True:
                        time.sleep(0.001)
                _ob = np.array(self.wss.recv['ob'])
                reward = self.wss.recv['reward']
                done = self.wss.recv['done']
                info = "this is for gym"
                return _ob,reward,done,info

        def getActions(self,ob):
                self.wss.sendMsg(json.dumps({'act':"getActions",'ob':ob}))
                while self.wss.isRecvWaiting == True:
                        time.sleep(0.001)
                actions = np.array(self.wss.recv['actions'])
                return actions
