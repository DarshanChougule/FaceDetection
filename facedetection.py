import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk  # To display images in Tkinter

def upload_and_detect_face():
    global img_label, face_count_label
    
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        face_count_label.config(text="No file selected.")
        return
    
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read image
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Update face count label in GUI
    face_count_label.config(text=f"Faces Detected: {len(faces)}")

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convert image for Tkinter display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Resize image to fit within Tkinter window (maintaining aspect ratio)
    img_pil.thumbnail((500, 500))  
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update the label with the processed image
    img_label.config(image=img_tk)
    img_label.image = img_tk  # Keep reference to avoid garbage collection

# Create Tkinter window
root = tk.Tk()
root.title("Face Detection App")

# Create UI elements
upload_btn = Button(root, text="Upload Image", command=upload_and_detect_face, font=("Arial", 12))
upload_btn.pack(pady=10)

face_count_label = Label(root, text="Faces Detected: 0", font=("Arial", 14))
face_count_label.pack(pady=10)

img_label = Label(root)  # Placeholder for image display
img_label.pack()

# Run the application
root.mainloop()
