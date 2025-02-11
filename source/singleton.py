class Singleton(type):

    _instances = {}

    # This will work so each subclass of a Singleton class, is an independent Singleton
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super().__call__(*args, **kwargs)
        return Singleton._instances[cls]
