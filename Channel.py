class Channel:
    def __init__(self):
        self.packetList = []

    # packet create after simuTime will not be received
    def receiveTo(self, nodeId, simuTime):
        receivedPacketList = []
        for packet in self.packetList:
            if packet.destinationNodeId is nodeId and packet.createTime < simuTime:
                receivedPacketList.append(packet)

        for packet in receivedPacketList:
            self.packetList.remove(packet)

        return receivedPacketList

    def sendPacket(self, packet):
        self.packetList.append(packet)
        pass

    def initialize(self):
        self.packetList.clear()


channel = Channel()
