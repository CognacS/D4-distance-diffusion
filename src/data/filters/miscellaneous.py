from src.data.filters import reg_filters

@reg_filters.register()
class FilterNone:

    def __call__(self, data):
        return data is not None
    
@reg_filters.register()
class FilterNoneFromList:
    
    def __init__(self, list_of_data):
        self.is_none_list = [d is None for d in list_of_data]
        
    def __call__(self, data):
        return not self.is_none_list[data]