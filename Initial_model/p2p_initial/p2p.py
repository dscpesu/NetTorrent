import socket 
import threading 
import sys
import time
from random import randint


class p2p:
    peers = ["127.0.0.1"] 

class Server:
    connections = []
    peers = []
    def __init__(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 10000))
        s.listen(5)
        print("Server Running...")
        #run 
        while True:
            connection, addr = s.accept()
            cThread = threading.Thread(target=self.handler, args = (connection, addr))
            cThread.daemon = True
            cThread.start()
            #cThread.join()
            self.connections.append(connection)
            self.peers.append(addr[0])
            print(str(addr[0]) + ":" + str(addr[1]) + " connected")
            self.sendPeers()    

    def handler(self, connection, addr):
        # data = connection.recv(1024)
        try:
            while True:
                data = connection.recv(1024)
                for connection_i in self.connections:
                    if not data:
                        print(str(addr[0]) + ":" + str(addr[1]), "disconnected")
                        self.connections.remove(connection)
                        self.peers.remove(addr[0])
                        connection.close()
                        self.sendPeers()
                        return
                    if connection_i != connection:
                        connection_i.send(data)
        except:
            pass
    def sendPeers(self):
        p = ""
        for peer in self.peers:
            p = p + peer + ","
        for connection in self.connections:
            connection.send(b"\x11"+bytes(p, "utf-8"))

class Client:
    def __init__(self, addr):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect((addr, 10000))

        cThread = threading.Thread(target = self.sendMsg, args = (s,))
        cThread.daemon = True
        cThread.start()

        while True:
            data = s.recv(1024)
            if not data:
                break
            if data[0:1]==b"\x11":
                self.updatePeers(data[1:]) #first byte excluded
            else:
                print(str(data, "utf-8"))
    def sendMsg(self, s):
        while True:
            try:
                """
                Assign the input function to some variable called msg
                Let msg be the flag, so if it is t then train it, s means send it, c means comapre and see what to do
                write one global function for comapring accuracies
                instead of bytes(input(),"utf-8") you can use the same logic as the one in the old code, with all the 
                functionalities of fileIO.py 
                """
                #msg = input("> ")
                s.send(bytes(input(""), "utf-8"))
            except KeyboardInterrupt:

                sys.exit()
            except:
                pass
    def updatePeers(self, peerData):
        p2p.peers = str(peerData, "utf-8").split(",")[:-1]

# if (len(sys.argv)>1):
#     client = Client(sys.argv[1])
# else:
#     server = Server()

while True:
    try:
        print("Trying to connect...")
        time.sleep(randint(1,5))
        for peer in p2p.peers:
            try:
                client = Client(peer)
            except KeyboardInterrupt:
                sys.exit()
            except:
                pass
            if randint(1,10) == 1:
                try:
                    server = Server()
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    print("Couldn't start server...")
            
    except KeyboardInterrupt:
        sys.exit(0)