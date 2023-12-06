import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def save_to_file(user_input):
    with open("user_input.txt", "w") as file:
        file.write(user_input + "\n")

def on_submit():
    user_input = entry.get()
    if user_input:
        save_to_file(user_input)
        entry.delete(0, tk.END)  # Clear the entry field after saving

# GUI setup
root = tk.Tk()
root.title("Ask ChatGPT")

# Load and display an image
image_path = "1.jpg"  # Replace with the actual image path
try:
    img = Image.open(image_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    panel = ttk.Label(root, image=img)
    panel.image = img
    panel.pack(pady=10)
except Exception as e:
    print(f"Error loading image: {e}")

# Entry widget for user input
entry = tk.Entry(root, width=40, font=('Arial', 12))
entry.pack(pady=10)

# Submit button
submit_button = ttk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

# Run the GUI
root.mainloop()