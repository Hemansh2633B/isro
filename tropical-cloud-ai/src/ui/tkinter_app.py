from tkinter import Tk, Label, Button, Frame, filedialog, messagebox
import os
import cv2
import numpy as np
from src.inference.realtime import run_inference
from src.data.preprocessing import preprocess_image

class TropicalCloudAIApp:
    def __init__(self, master):
        self.master = master
        master.title("Tropical Cloud AI")

        self.frame = Frame(master)
        self.frame.pack()

        self.label = Label(self.frame, text="Tropical Cloud AI - Inference")
        self.label.pack()

        self.load_button = Button(self.frame, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.run_button = Button(self.frame, text="Run Inference", command=self.run_inference)
        self.run_button.pack()

        self.result_label = Label(self.frame, text="")
        self.result_label.pack()

        self.image_path = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.result_label.config(text=f"Loaded image: {os.path.basename(self.image_path)}")

    def run_inference(self):
        if self.image_path:
            try:
                preprocessed_image = preprocess_image(self.image_path)
                result = run_inference(preprocessed_image)
                self.display_result(result)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showwarning("Warning", "Please load an image first.")

    def display_result(self, result):
        # Assuming result is an image array
        result_image = (result * 255).astype(np.uint8)  # Convert to uint8 for display
        cv2.imshow("Inference Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    app = TropicalCloudAIApp(root)
    root.mainloop()