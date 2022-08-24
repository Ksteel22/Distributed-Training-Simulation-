from Event import Event
from NodeStatus import nodeStatus

eventList = []

startTrainEvent = Event(nodeStatus.startAllTrain(), {}, 0)
startTrainEvent.run()

