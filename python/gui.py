import tkinter as tk
from tkinter import ttk, Frame

def button_clicked(textbox):
    textbox.insert(tk.END, "Button clicked!\n")

def button():
    root = tk.Tk()
    button = ttk.Button(root, text='Click Me', command=button_clicked)
    button.pack()

    button.mainloop()

root = tk.Tk()

msg_fr = Frame(root)

message = tk.Label(msg_fr, text="Hello, World!")
message.grid(row=0,
                column=0,
                columnspan=3)
#root.title("Hello")

window_width = 500
window_height = 300

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
# Create a Text widget for displaying output
txt_fr= Frame(root)

output_text = tk.Text(txt_fr, height=100, width=200)
output_text.grid(row=1,
                    column=0,
                    columnspan=3)

btn_fr=Frame(root)
button = ttk.Button(btn_fr, text='Click Me', command=button_clicked(output_text))
button.grid(row=2,
            column=0,
            padx=(10),
            pady=10,
            sticky=tk.W)

root.mainloop()

# Helper
#



