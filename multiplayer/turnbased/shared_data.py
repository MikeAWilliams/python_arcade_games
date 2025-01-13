class InitialState:
    def __init__(self, message):
        self.message = message

    def GetMessage(self):
        return self.message


class Error:
    def __init__(self, message):
        self.message = message

    def GetMessage(self):
        return self.message


class InputType1:
    pass


class InputType2:
    pass


class ClientGameState:
    def __init__(self, message):
        self.message = message

    def GetMessage(self):
        return self.message
