class frame_buffer():
    def __init__(self):
        self.frame1 = None
        self.frame2 = None
    
    def set_current(self, frame):
        self.frame1 = self.frame2
        self.frame2 = frame