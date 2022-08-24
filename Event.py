class Event:
    def __init__(self, function, dict_functionParam, createTime, duration, eventType=0):
        self.function = function
        self.dict_functionParam = dict_functionParam
        self.createTime = createTime
        self.duration = duration
        self.eventType = eventType

    def run(self):
        self.function(self.dict_functionParam)
        pass
