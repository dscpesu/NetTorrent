import socket 
import traceback 
from fileIO import *
import numpy as np 

SERVER_PORT = 50000

choice = -1
flag = 0
user = None

def choices():
    """
    Provides a list of choices to the user. 
    1       :   User switches to server mode, to send a broadcast message.
    2       :   User switches to client mode, to receive a broadcast message. 
    flag    :   Used to keep track of whether user is already in client or server mode.
    change  :   Attribute of the Server and Client class to indicate when to change from one mode to the next.
    """
    global choice
    global flag
    global user
    while choice!=0:
        print("In loop")
        print("Enter 1 for Server mode: ")
        print("Enter 2 for Client mode: ")
        print("Enter 0 to Exit.")
        print("\n")
        choice = int(input("Enter your choice: "))

        if choice==1 and flag!=1:
            user.change = True
            user = Server(("<broadcast>",SERVER_PORT))
            user.broadcast() 
            flag = 1
        if (choice==2 and flag!=2):
            user.change = True
            user = Client("", 50050) #default address 
            user.receive()
            flag = 2


class Server:
    def __init__(self, dest, change = False):
        self.dest = dest
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.change = change
        
    def broadcast(self):
        while True and not self.change and choice:   
            try:
                data = np.array([1,2,3,4])
                data = np_to_json(data)
                self.msg = convert_to_bytes()
                print("Broadcasting...\n")
                self.s.sendto(self.msg, self.dest)
                print("Done")
                self.s.shutdown(socket.SHUT_RDWR)
                self.s.close()
                choices()
            except KeyboardInterrupt:
                raise
            except:
                traceback.print_exc()
            
class Client:
    def __init__(self, host, port, change = False):
        self.change = change
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.s.bind((self.host,self.port))
    
    def receive(self):
        while True and not self.change and choice:
            print("Client Loop")
            print("\n")
            try:
                print("lol")
                message, address = self.s.recvfrom(1024)
                print("Creating file")
                message = create_file(message)
                print("Got data %s from  %s" % (message, address))
                print("\n")
                self.s.shutdown(socket.SHUT_RDWR)
                self.s.close()
                choices()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                traceback.print_exc()


print("\n")
print("Enter 1 for Server mode: ")
print("Enter 2 for Client mode: ")
print("Enter 0 to Exit.")
print("\n")
choice = int(input("Enter your choice: "))

if choice==1:
    user = Server(("<broadcast>",SERVER_PORT))
    user.broadcast() 
    flag = 1
if choice==2:
    user = Client("", SERVER_PORT)
    user.receive()
    flag = 2

