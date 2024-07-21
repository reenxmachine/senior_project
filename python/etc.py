import re
from tkinter import filedialog, Tk


root = Tk()
root.withdraw()

def trim(input_string):
    # Use a regular expression to find text between single quotation marks
    match = re.search(r"'([^']*)'", input_string)
    
    # Check if there is a match
    if match:
        # Extract the text between single quotes
        result = match.group(1)
        return result
    else:
        # If no match found, return None or handle accordingly
        return None
    
def search_for_file_path (whatfile):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width/2 - 300)
    center_y = int(screen_height/2 - 200)

    # set the position of the window to the center of the screen
    root.geometry(f'{600}x{400}+{center_x}+{center_y}')
    root.withdraw()
    file = filedialog.askopenfile(parent=root,mode='rb',title=f'Choose {whatfile}')
    if file is None:
        file = ''
    return str(file)

def save_file(file):
    f = filedialog.asksaveasfile(mode='w', defaultextension='.joblib')
    if f is None:
        return
    f.write(file)
    f.close()
#print(search_for_file_path("thisfile"))