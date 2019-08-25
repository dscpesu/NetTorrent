"""
A simulation for the Chord Distributed Hash Table written in Python 3.6.6
The network is simulated as a circular double linked list. 
Node look-ups happen with the help of a finger table in logarithmic time. 
"""

class Node:
    def __init__(self, id, k, prev = None, nxt = None):
        """
        id      : uid of the node
        prev    : id of the previous node
        nxt     : id of the next node     
        k       : key size
        """
        self.id = id
        self.prev = prev 
        self.next = nxt
        self.fingerTable = dict()
        self.data = dict()
        self.k = k
        self.size = 2**k    

    def ringDistance(self, id1, id2):
        """
        Calculates the ring distance between two nodes, references to whose objects are self, this.
        """
        if id1 == id2:
            return 0
        elif id1<id2:
            return id2 - id1
        else:
            return self.size + (id2 - id1)

    def getHash(self, key):
        """Computes the hash of the key"""
        return key % self.size
        
    def findNode(self, start, key):
        """
        Finds the closest node for the given key. 
        """
        current = start
        keyHash = self.getHash(key)
        while self.ringDistance(current.id, keyHash) > self.ringDistance(current.next.id, keyHash):
            current = current.next
        return current

    def getValue(self, start, key):
        """Finds the node for the target key and returns the corresponding value"""
        node = self.findNode(start, key)
        return node.data[key]
    
    def storeValue(self, start, key, value):
        """Finds the node for the target key and stores the value"""
        node = self.findNode(start, key)
        node.data[key] = value
    

    def updateFingerTable(self):
        """
        Update finger table for the current node.
        """
        for i in range(self.k):
            oldKey = self.fingerTable[i]
            self.fingerTable[i] = self.findNode(oldKey, self.id + ((2**i)) % self.size)

    def findInFT(self, node, key):
        """Finds in finger table"""
        current = node
        keyHash = self.getHash(key)
        for i in range(self.k):
            if self.ringDistance(current.id, keyHash) > self.ringDistance(node.fingerTable[i].id, keyHash):
                current = node.fingerTable[i]
        
        return current
    
    def lookUpFT(self, start, key):                                                             
        current = self.findInFT(start, key)
        next = self.findInFT(current, key)
        keyHash = self.getHash(key)

        while self.ringDistance(current.id, keyHash) > self.ringDistance(next.id, keyHash):
            current = next
            next = self.findInFT(current, key)
        return current

class DHT:    
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.startNode = Node(0, k)
        self.startNode.fingerTable[0] = self.startNode
        self.startNode.updateFingerTable()

    def getNodeNum(self):
        if self.startNode == None:
            return 0
        
        node = self.startNode
        nodeNum = 1
        while node.fingerTable[0] != self.startNode:
            nodeNum += 1
            node = node.fingerTable[0]
        return nodeNum
    
    #TODO handle nodes joining and leaving the network


