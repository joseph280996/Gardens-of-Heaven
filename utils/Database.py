from typing import Any


class Database:
    """
    This class is a representation of the iris database which will handles reading the file and parse it out
    Properties:
        data: The parsed data from the file.
    """

    def __init__(self, file_name: str) -> None:
        self.data = []
        with open(f"{file_name}.txt", "r") as file:
            for line in file:
                cleaned_line = line.rstrip("\n")
                if len(cleaned_line) > 0:
                    splitted_cleaned_line = cleaned_line.split(",")
                    instance = []

                    for val in splitted_cleaned_line:
                        try:
                            instance.append(float(val))
                        except ValueError:
                            instance.append(val)

                    self.data.append(instance)

    def get_data(self) -> list[list[Any]]:
        """
        This method will return the data that was parsed and processed.

        Returns:
            The list of data that was cleaned and splitted.
        """
        return self.data
