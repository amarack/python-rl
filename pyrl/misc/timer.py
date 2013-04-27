import time

class Timer(object):
    """Context manager to time a block of code.

    From http://stackoverflow.com/a/1685337/1306923
    Thanks to Corey Porter!

    """
    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        # Error handling here
        self.__finish = time.time()

    def duration_in_seconds(self):
        return self.__finish - self.__start
