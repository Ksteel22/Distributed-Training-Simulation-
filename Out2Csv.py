class Out2Csv:
    def __init__(self, path):
        self.filePath = path
        self.file = open(path, 'w')
        pass

    def addPoint(self, strData):
        strData += "\n"
        self.file.write(strData)
        pass

    def closeIO(self):
        self.file.close()
        pass
