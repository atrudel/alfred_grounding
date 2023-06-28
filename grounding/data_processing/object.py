class Object:
    def __init__(self, name: str):
        self.name: str = name
        # Todo: find other attributes from file
        # Return None if str is ''

    def __str__(self) -> str:
        return self.name
