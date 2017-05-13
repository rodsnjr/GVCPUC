import cv2

class ImageController:
    def __init__(self, view=None, files=None, model=None):
        self.model = model
        self.files = files
        self.view = view
        self.current_file = 0
    
    # Load the current progress from the model file
    def load_progress(self):
        self.model.load()
        self.current_file = len(self.model.items)

    # Start the Window/Controller Loop
    def start(self):
        self.load_progress()
        
        self.view.load_image(self.files[self.current_file])
        self.view.show()

        while True:
            key = cv2.waitKey(1) & 0xFF
            # next image
            if key == ord("n"):
                self.next()
            # previus image
            elif key == ord("p"):
                self.previous()
            # Labels
            elif key == ord("1"):
                self.label1()
            elif key == ord("2"):
                self.label2()
            elif key == ord("3"):
                self.label3()
            elif key == ord('s'):
                self.save_current()
            # quit
            elif key == ord("q"):
                break
        self.close()

    # Directory Travelling Events
    def next(self):
        self.model.add(self.files[self.current_file], self.view.shown.describe())
        self.current_file += 1
        self.view.show_next(self.files[self.current_file])
        # if self.on_next:
        #    self.on_next()
    
    def save_current(self):
        self.model.add(self.files[self.current_file], self.view.shown.describe())
        self.model.save()

    def previous(self):
        self.current_file -= 1
        self.view.show_item(self.model.last())
    
    def close(self):
        self.model.save()
        # if self.on_close:
        #    self.on_close()

    # GUI Drawing / Changing Events
    def label1(self):
        self.view.shown.change_label('left')
        self.view.refresh()
    
    def label2(self):
        self.view.shown.change_label('center')
        self.view.refresh()
    
    def label3(self):
        self.view.shown.change_label('right')
        self.view.refresh()
    
    # Controller Event Setters
    def on_next(self, on_next):
        self.on_next = on_next
    
    def on_previous(self, on_previous):
        self.on_previous = on_previous
    
    def on_close(self, on_close):
        self.on_close = on_close