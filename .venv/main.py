import tkinter as tk
import customtkinter
from facemesh import facemeshapp
from face_detect_and_filter import  face_detect_and_filter
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

#App Frame
app = customtkinter.CTk()
app.geometry("880x500")
app.title("Facial Shifter")

# Configure the grid to expand the frame
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

# Create a main app frame
main_app_frame = customtkinter.CTkFrame(app)

# Configure the main_app_frame grid to center the content
main_app_frame.grid_rowconfigure(0, weight=1)
main_app_frame.grid_rowconfigure(2, weight=1)  # Add additional rows for balancing
main_app_frame.grid_columnconfigure(0, weight=1)
main_app_frame.grid_columnconfigure(5, weight=1)  # Add additional columns for balancing

# Define a function to display the main app
def display_main_app():
    welcome_frame.grid_remove()  # Remove the welcome frame
    main_app_frame.grid(row=0, column=0, sticky='nsew')  # Add the main app frame to the grid

    #UI Elements
    title = customtkinter.CTkLabel(main_app_frame, text="Select A Filter", font=('Arial', 25))
    title.grid(row=1, column=1, columnspan=4, padx=5, pady=5)

    img1=Image.open(r"./Static/images/FaceMesh.jpg")
    img1 = customtkinter.CTkImage(img1, size=(150,150))

    # Use CTkButton instead of tkinter Button
    btnFaceMesh = customtkinter.CTkButton(main_app_frame, image=img1, text="Face Mesh", width=200, height=200, command=facemeshapp, compound='top')
    btnFaceMesh.grid(row=2, column=1, padx=10, pady=10)

    img2=Image.open(r"./Static/images/cowboy.jpg")
    img2 = customtkinter.CTkImage(img2, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(main_app_frame, image=img2, text="Cowboy", width=200, height=200, command= lambda: face_detect_and_filter('COWBOY_FILTER'), compound='top')
    btnFaceMesh.grid(row=2, column=2, padx=10, pady=10)

    img3=Image.open(r"./Static/images/bossModeX.png")
    img3 = customtkinter.CTkImage(img3, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(main_app_frame, image=img3, text="Boss Mode", width=200, height=200, command=lambda: face_detect_and_filter('SUNGLASSES_FILTER'), compound='top')
    btnFaceMesh.grid(row=2, column=3, padx=10, pady=10)

    img4=Image.open(r"./Static/images/sunglasses.jpg")
    img4 = customtkinter.CTkImage(img4, size=(150,150))

    btnFaceMesh = customtkinter.CTkButton(main_app_frame, image=img4, text="Sunglasses", width=200, height=200, command=facemeshapp, compound='top')
    btnFaceMesh.grid(row=2, column=4, padx=10, pady=10)

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

start_button = customtkinter.CTkButton(welcome_frame, text="Get Started", command=display_main_app)
start_button.grid(row=1, column=1, padx=10, pady=10)

#Run App
app.mainloop()