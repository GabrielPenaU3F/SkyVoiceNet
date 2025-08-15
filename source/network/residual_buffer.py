from source.singleton import Singleton


class ResidualBuffer(metaclass=Singleton):

    def __init__(self):
        self.input = None
        self.conv_1_output = None
        self.conv_2_output = None
        self.conv_3_output = None

    def buffer_conv_1_output(self, x):
        self.conv_1_output = x

    def buffer_conv_2_output(self, x):
        self.conv_2_output = x

    def buffer_conv_3_output(self, x):
        self.conv_3_output = x

    def buffer_input(self, x):
        self.input = x

    def retrieve_buffer_conv_1_output(self):
        x = self.conv_1_output
        self.conv_1_output = None
        return x

    def retrieve_buffer_conv_2_output(self):
        x = self.conv_2_output
        self.conv_2_output = None
        return x

    def retrieve_buffer_conv_3_output(self):
        x = self.conv_3_output
        self.conv_3_output = None
        return x

    def retrieve_input(self):
        x = self.input
        self.input = None
        return x
