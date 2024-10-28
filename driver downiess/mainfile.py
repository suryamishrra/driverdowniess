import tkinter as tk
from tkinter import messagebox
import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
from PIL import Image, ImageTk

# Class for the Drowsiness Detection GUI Application
class DrowsinessDetectionGUI:
    def __init__(self, master):
        # Initialize the main application window
        self.master = master
        self.master.title("Driver Drowsiness Detection System")
        self.master.iconphoto(False, tk.PhotoImage(file='dms.png'))
        self.master.state('zoomed')  # Maximize the window

        self.dark_mode = False  # Initial state for dark mode

        # Load and set the background image
        self.background_image_path = 'background_image.jpg'
        self.background_image = Image.open(self.background_image_path)

        # Create a canvas to hold the background image
        self.canvas = tk.Canvas(self.master, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.update_background()

        self.setup_gui()  # Setup GUI elements
        self.cap = None  # Placeholder for video capture
        self.detector = None  # Placeholder for dlib face detector
        self.predictor = None  # Placeholder for dlib facial landmark predictor
        self.mixer = None  # Placeholder for pygame mixer for alert sound
        self.thresh = 0.22  # Threshold for eye aspect ratio (EAR)
        self.frame_check = 20  # Number of frames to check for drowsiness
        self.flag = 0  # Counter for consecutive frames with low EAR
        self.detection_active = False  # Flag to check if detection is active

        self.master.bind("<Configure>", self.resize_background)  # Bind resize event to update background

    def update_background(self):
        # Update the background image size based on window size
        self.canvas_width = self.master.winfo_width()
        self.canvas_height = self.master.winfo_height()
        self.background_image_resized = self.background_image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
        self.background_image_tk = ImageTk.PhotoImage(self.background_image_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_image_tk)

    def resize_background(self, event):
        # Resize the background image on window resize
        self.update_background()

    def setup_gui(self):
        # Setup GUI elements including buttons, labels, and frames
        self.header_frame = tk.Frame(self.master, bg="#4d4d4d")
        self.header_frame.place(relx=0.5, rely=0, anchor="n", y=10)

        self.dark_mode_button = tk.Button(self.header_frame, text="Dark Mode", font=("Arial", 10), bg="#8c8c8c", fg="black", command=self.toggle_dark_mode)
        self.dark_mode_button.grid(row=0, column=1, padx=10, pady=10)

        self.header_label = tk.Label(self.header_frame, text="Driver Drowsiness Detection System", font=("Arial", 24), bg="#8c8c8c", fg="white")
        self.header_label.grid(row=0, column=0, padx=10, pady=10)

        self.content_frame = tk.Frame(self.master, bg="#4d4d4d")
        self.content_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Load start button image
        self.start_img = Image.open('start.png').resize((100, 50))
        self.start_img = ImageTk.PhotoImage(self.start_img)
        self.start_button = tk.Button(self.content_frame, image=self.start_img, bd=0, bg="#4d4d4d", activebackground="#4d4d4d", command=self.start_detection)
        self.start_button.pack(pady=20)

        # Load stop button image
        self.stop_img = Image.open('stop.png').resize((100, 50))
        self.stop_img = ImageTk.PhotoImage(self.stop_img)
        self.stop_button = tk.Button(self.content_frame, image=self.stop_img, bd=0, bg="#4d4d4d", activebackground="#4d4d4d", command=self.stop_detection, state="disabled")
        self.stop_button.pack(pady=20)

        # Status label for displaying detection status
        self.status_label = tk.Label(self.content_frame, text="Not detecting", font=("Arial", 18), bg="#8c8c8c", fg="white")
        self.status_label.pack(pady=20)

        self.video_frame = tk.Frame(self.content_frame, bg="#4d4d4d")
        self.video_frame.pack(fill="both", expand=True)

        # Label for displaying video frames
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

    def start_detection(self):
        # Start the drowsiness detection process
        self.status_label.config(text="Detecting...")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.detection_active = True

        try:
            self.init_resources()  # Initialize resources like camera and dlib models
            self.update_frame()  # Start updating frames for detection
        except cv2.error as e:
            messagebox.showerror("Error", f"OpenCV error: {str(e)}")
            self.reset_buttons()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.reset_buttons()

    def stop_detection(self):
        # Stop the drowsiness detection process
        self.status_label.config(text="Not detecting")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.detection_active = False
        self.release_resources()  # Release camera and other resources

    def init_resources(self):
        # Initialize resources like camera, dlib models, and pygame mixer
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise cv2.error("OpenCV Error: Cannot open camera")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.mixer = mixer
        self.mixer.init()
        self.mixer.music.load("music.wav")

    def update_frame(self):
        # Update video frames for detection
        if not self.detection_active:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.release_resources()
            return
        frame = imutils.resize(frame, width=850)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subjects = self.detector(gray, 0)

        for subject in subjects:
            shape = self.predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    cv2.putText(frame, "----------------ALERT!----------------", (130, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.mixer.music.play()
            else:
                self.flag = 0

            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        # Convert frame to ImageTk format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Update video label with the new frame
        self.video_label.config(image=image)
        self.video_label.image = image

        # Call update_frame again after 10ms
        self.master.after(10, self.update_frame)

    def eye_aspect_ratio(self, eye):
        # Calculate the eye aspect ratio (EAR)
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def release_resources(self):
        # Release resources like camera and close OpenCV windows
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def reset_buttons(self):
        # Reset the state of buttons
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def toggle_dark_mode(self):
        # Toggle dark mode on and off
        if self.dark_mode:  # Check if dark mode is currently enabled
            # Set light mode colors
            self.master.configure(background="#4d4d4d")  # Set light grey background
            self.header_frame.configure(bg="#4d4d4d")  # Set header frame background
            self.content_frame.configure(bg="#4d4d4d")  # Set content frame background
            self.status_label.configure(bg="#8c8c8c", fg="white")  # Set status label colors
            self.dark_mode_button.config(bg="#8c8c8c", fg="black", text="Dark Mode")  # Update button text
            self.dark_mode = False  # Set dark mode flag to false
        else:
            # Set dark mode colors
            self.master.configure(background="#2e2e2e")  # Set dark grey background
            self.header_frame.configure(bg="#2e2e2e")  # Set header frame background
            self.content_frame.configure(bg="#2e2e2e")  # Set content frame background
            self.status_label.configure(bg="#8c8c8c", fg="white")  # Set status label colors
            self.dark_mode_button.config(bg="#8c8c8c", fg="black", text="Light Mode")  # Update button text
            self.dark_mode = True  # Set dark mode flag to true

# Main function to run the application
if __name__ == "__main__":
    root = tk.Tk()  # Create the main tkinter window
    app = DrowsinessDetectionGUI(root)  # Instantiate the application class
    root.mainloop()  # Start the main event loop
