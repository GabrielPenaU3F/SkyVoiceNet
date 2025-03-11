from source.singleton import Singleton


class ResidualBuffer(metaclass=Singleton):

    def __init__(self):
        self.conv_1_output = None
        self.conv_2_output = None
        self.transformer_input_buffer = None
        self.transformer_output_buffer = None

    def buffer_conv_1_output(self, x):
        self.conv_1_output = x

    def buffer_conv_2_output(self, x):
        self.conv_2_output = x

    def buffer_transformer_input(self, x):
        self.transformer_input_buffer = x

    def buffer_transformer_output(self, x):
        self.transformer_output_buffer = x

    def retrieve_buffer_conv_1_output(self):
        x = self.conv_1_output
        self.conv_1_output = None
        return x

    def retrieve_buffer_conv_2_output(self):
        x = self.conv_2_output
        self.conv_2_output = None
        return x

    def retrieve_transformer_output_buffer(self):
        x = self.transformer_output_buffer
        self.transformer_output_buffer = None
        return x

    def retrieve_transformer_input_buffer(self):
        x = self.transformer_input_buffer
        self.transformer_input_buffer = None
        return x
