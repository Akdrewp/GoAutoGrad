""" """


# Low-level numerical storage & strided memory (Backend / C interface)
class NDArray:
    """Interface for memory level tensor representation

    This is implemented in C++ to make operations fast.

    """

    def __init__(self, data, shape, strides, offset=0):
        self.strides = strides

        # This would usually call a c backend
        # self.data = Cdata()
        self.data = data

        self.shape = shape
        pass
