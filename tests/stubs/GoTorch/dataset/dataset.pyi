from typing import Generic, TypeVar

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co]):
    """An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """
    def __init__(self) -> None:
        """Initializes the dataset instance."""
    def __getitem__(self, index: int) -> T_co:
        """Retrieves a sample from the dataset at the specified index.

        Args:
            index: Integer index of the sample to retrieve.

        Returns:
            The dataset item at the given index.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            Total count of samples.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
