class ServiceBase:

    def init(self): 
        ''' Long running init, __init__ should be lightweight '''
        pass 

    def get_finalizer(self):
        ''' Cleanup resources in returned finalizer, 
            Additionally ad some metrics before finalization 
        '''
        return None  
