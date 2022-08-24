from Config import Config
from Out2Csv import Out2Csv

out2Csv = Out2Csv("nodeNum.csv")
out2Csv.addPoint("nodeNum,loss,acc")
config = Config()

for i in range(3, 8):
    config.nodeNum = i
    # print(Config.dataNum)
    loss, acc = config.startSimulation()
    out2Csv.addPoint("{},{},{}".format(config.nodeNum, loss, acc))


out2Csv.closeIO()
