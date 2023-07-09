import tkinter as tk
import customtkinter
from facialshifter.facemesh import facemeshapp, face_swap_pic, captureFrame
from facialshifter.face_detect_and_filter import  face_detect_and_filter
from facialshifter.still_image_filter import bossModeImage
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

#App Frame
app = customtkinter.CTk()
app.geometry("900x600")
app.title("Facial Shifter")

# Configure the grid to expand the frame
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

# Define a function to display the welcome frame
def welcome():
    # Create a welcome frame
    welcome_frame = customtkinter.CTkFrame(app)
    welcome_frame.grid(row=0, column=0, sticky='nsew')  # Make it fill the entire window

    # Configure the welcome_frame grid to center the content
    welcome_frame.grid_rowconfigure(0, weight=1)
    welcome_frame.grid_rowconfigure(2, weight=1)  # Add additional rows for balancing
    welcome_frame.grid_columnconfigure(0, weight=1)
    welcome_frame.grid_columnconfigure(2, weight=1)  # Add additional columns for balancing

    # Add a label and a button to the welcome frame
    welcome_label = customtkinter.CTkLabel(welcome_frame, text="Welcome to Facial Shifter", font=('Arial', 25))
    welcome_label.grid(row=0, column=1, padx=5, pady=5)

    start_button = customtkinter.CTkButton(welcome_frame, text="Get Started", command=lambda: image_or_video(welcome_frame))
    start_button.grid(row=1, column=1, padx=10, pady=10)

# Define a function to display the options frame
def image_or_video(welcome_frame):
    welcome_frame.grid_remove()  # Remove the welcome frame

    # Create a welcome frame
    option_frame = customtkinter.CTkFrame(app)
    option_frame.grid(row=0, column=0, sticky='nsew')  # Make it fill the entire window

    # Configure the welcome_frame grid to center the content
    option_frame.grid_rowconfigure(0, weight=1)
    option_frame.grid_rowconfigure(1, weight=1)  # Add additional rows for balancing
    option_frame.grid_columnconfigure(0, weight=1)
    option_frame.grid_columnconfigure(1, weight=1)  # Add additional columns for balancing

    # Add a label and a button to the welcome frame
    options_label = customtkinter.CTkLabel(option_frame, text="Make a Selection:", font=('Arial', 25))
    options_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    image_button = customtkinter.CTkButton(option_frame, text="Images", command=lambda: image_app(option_frame))
    image_button.grid(row=1, column=0, padx=10, pady=10)

    video_button = customtkinter.CTkButton(option_frame, text="Live Video", command=lambda: video_app(option_frame))
    video_button.grid(row=1, column=1, padx=10, pady=10)

# Define a function to display the image filters
def image_app(option_frame):
    option_frame.grid_remove()  # Remove the welcome frame

    # Create a main app frame
    image_frame = customtkinter.CTkFrame(app)

    # Configure the main_app_frame grid to center the content
    image_frame.grid_rowconfigure(0, weight=1)
    image_frame.grid_rowconfigure(2, weight=1)  # Add additional rows for balancing
    image_frame.grid_columnconfigure(0, weight=1)
    image_frame.grid_columnconfigure(1, weight=1)  # Add additional columns for balancing
    image_frame.grid(row=0, column=0, sticky='nsew')  # Add the main app frame to the grid

    title = customtkinter.CTkLabel(image_frame, text="Select A Filter", font=('Arial', 25))
    title.grid(row=0, column=0, columnspan=4, padx=5, pady=5)

    img1=Image.open(r"../Static/images/bossModeX.png")
    img1 = customtkinter.CTkImage(img1, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(image_frame, image=img1, text="Boss Mode", width=200, height=200, command=bossModeImage , compound='top')
    btnFaceMesh.grid(row=1, column=0, padx=10, pady=10)
    img2=Image.open(r"../Static/images/will.png")
    img2 = customtkinter.CTkImage(img2, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(image_frame, image=img2, text="Face Swap", width=200, height=200, command=face_swap_pic, compound='top')
    btnFaceMesh.grid(row=1, column=1, padx=10, pady=10)

    back_button = customtkinter.CTkButton(image_frame, text="Go back", command=lambda: image_or_video(option_frame))
    back_button.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

# Define a function to display the video filters
def video_app(option_frame):
    option_frame.grid_remove()  # Remove the welcome frame

    # Create a main app frame
    video_frame = customtkinter.CTkFrame(app)

    # Configure the main_app_frame grid to center the content
    video_frame.grid_rowconfigure(0, weight=1)
    video_frame.grid_rowconfigure(4, weight=1)  # Add additional rows for balancing
    video_frame.grid_columnconfigure(0, weight=1)
    video_frame.grid_columnconfigure(5, weight=1)  # Add additional columns for balancing
    video_frame.grid(row=0, column=0, sticky='nsew')  # Add the main app frame to the grid

    #UI Elements
    title = customtkinter.CTkLabel(video_frame, text="Select A Filter", font=('Arial', 25))
    title.grid(row=1, column=1, columnspan=4, padx=5, pady=5)

    img1=Image.open(r"../Static/images/FaceMesh.jpg")
    img1 = customtkinter.CTkImage(img1, size=(150,150))

    # Use CTkButton instead of tkinter Button
    btnFaceMesh = customtkinter.CTkButton(video_frame, image=img1, text="Face Mesh", width=200, height=200, command=facemeshapp, compound='top')
    btnFaceMesh.grid(row=2, column=1, padx=10, pady=10)

    img2=Image.open(r"../Static/images/cowboy.jpg")
    img2 = customtkinter.CTkImage(img2, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(video_frame, image=img2, text="Cowboy", width=200, height=200, command= lambda: face_detect_and_filter('COWBOY_FILTER'), compound='top')
    btnFaceMesh.grid(row=2, column=2, padx=10, pady=10)

    img3=Image.open(r"../Static/images/bossModeX.png")
    img3 = customtkinter.CTkImage(img3, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(video_frame, image=img3, text="Boss Mode", width=200, height=200, command=lambda: face_detect_and_filter('BOSSMODE_FILTER'), compound='top')
    btnFaceMesh.grid(row=2, column=3, padx=10, pady=10)

    img4=Image.open(r"../Static/images/police-filter.jpg")
    img4 = customtkinter.CTkImage(img4, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(video_frame, image=img4, text="Police", width=200, height=200, command=lambda: face_detect_and_filter('POLICE_FILTER'), compound='top')
    btnFaceMesh.grid(row=3, column=1, padx=10, pady=10)

    img5=Image.open(r"../Static/images/pirate-filter.jpg")
    img5 = customtkinter.CTkImage(img5, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(video_frame, image=img5, text="Pirate", width=200, height=200, command=lambda: face_detect_and_filter('PIRATE_FILTER'), compound='top')
    btnFaceMesh.grid(row=3, column=2, padx=10, pady=10)

    img6=Image.open(r"../Static/images/will.png")
    img6 = customtkinter.CTkImage(img6, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(video_frame, image=img6, text="Face Swap (Experimental)", width=200, height=200, command=captureFrame, compound='top')
    btnFaceMesh.grid(row=3, column=3, padx=10, pady=10)

    back_button = customtkinter.CTkButton(video_frame, text="Go back", command=lambda: image_or_video(video_frame))
    back_button.grid(row=4, column=1, columnspan=4, padx=10, pady=10)

welcome()

#Run App
app.mainloop()