# Ryan McShane
# 03-08-2024

# GUI Main Menu for Model Training, output and persistance

import tkinter as tk
import training
from tkinter import ttk, Frame, scrolledtext, simpledialog
from tkinter import *
from tkinter.messagebox import askyesno
import predict
import etc


model_file = ''

def start_button_click(): # Checks for debug mode then begins training model
    model_file=''
    if agreement.get() == "agree":
        output_text.configure(state='normal')
        output_text.insert(tk.END, "DEBUGTIME\n")
        output_text.configure(state='disabled')
    else:
        output_text.configure(state='normal')
        output_text.insert(tk.END, "Began Model Training\n")
        if len(model_file) == 0:
            output, clfd = training.runTraining()
            if askyesno(title="Save Model", message='Save Model?'):
                answer = simpledialog.askstring(title="Filename", prompt='What should the model be called?')
                training.dump(clfd, answer + '.joblib')
        else:
            output = training.runTraining(model_file)
        for x in output:
            output_text.insert(tk.END, x)
            output_text.insert(tk.END, "\n")
        output_text.insert(tk.END, "Done\n")
        output_text.configure(state='disabled')
        root.update_idletasks()

def select_model_button_click(): # Opens file selection for model.joblib
    output_text.configure(state='normal')
    output_text.insert(tk.END, "Select joblib file! \n")
    global model_file
    model_file = training.select_model()
    output_text.insert(tk.END, f"{model_file} has been selected\n")
    output_text.configure(state='disabled')


def print_var():
    if agreement.get() == "agree":
        tk.messagebox.showinfo(title='Result', message="DEBUG TIME")
    else:
        tk.messagebox.showinfo(title='Result', message="FOR EELS")

def clear_text():
    output_text.configure(state='normal')
    output_text.delete("1.0", "end")
    global model_file
    model_file = ''
    output_text.configure(state='disabled')

def prediction():
    output_text.configure(state='normal')
    if len(model_file) == 0:
        output_text.insert(tk.END, 'No model selected! Please select a model.\n')
        output_text.configure(state='disabled')
        return
    pred_target = etc.trim(etc.search_for_file_path("Select log file"))
    predictions = predict.makepredict(pred_target, model_file)
    count = 1
    for x in predictions:
        output_text.insert(tk.END, str(count)+': ')
        output_text.insert(tk.END, x+'\n')
        count+=1
    output_text.insert(tk.END, 'All logs have been evaluated...\n')
    output_text.configure(state='disabled')

# Create the main window
root = tk.Tk()
root.title("Senior Project")
agreement = tk.StringVar(root, "FORREALS", "agreement_var")

btn_frame = Frame(root, borderwidth = 1)
btn_frame.grid(row=0, column=0, sticky=tk.EW)
# Create a button and associate the on_button_click function with its command
start_button = tk.Button(btn_frame, text="Run Model Training", command=start_button_click)
start_button.grid(row=0, column=0, padx=(10), pady=10)

select_model_button = tk.Button(btn_frame, text="Select Pretrained Model", command=select_model_button_click)
select_model_button.grid(row=0, column=1, padx=(10), pady=10)

debug_check = ttk.Checkbutton(btn_frame,
                text='Debug Mode',
                command=print_var,
                variable=agreement,
                onvalue='agree',
                offvalue='disagree')
debug_check.grid(row=0, column=2, padx=(10), pady=10)

predict_button = tk.Button(btn_frame, text="Predict", command=prediction)
predict_button.grid(row=0, column=3, padx=(10), pady=10)

clear_text_btn = tk.Button(btn_frame, text="Clear output text", command=clear_text)
clear_text_btn.grid(row=0, column=4, padx=10,pady=10)

# Create a Text widget for displaying output
output_frame = Frame(root, borderwidth = 1)
output_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky=E+W+N+S)

root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

output_frame.rowconfigure(0, weight=1)
output_frame.columnconfigure(0, weight=1)

# Textbox
output_text = scrolledtext.ScrolledText(output_frame, height=20, width=100, state="normal")
output_text.grid(row=0, column=0,   sticky=E+W+N+S)

root.update()

def main():
    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()