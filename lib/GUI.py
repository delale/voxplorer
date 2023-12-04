"""
Graphical User Interface for VoxRimor built using tkinter.
"""

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import data_loader


class VisualizerWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Visualizer mode')
        self.create_widgets()

    def create_widgets(self):
        self.file_label = tk.Label(self, text='File path:')
        self.file_label.pack()

        self.file_button = tk.Button(
            self, text='Select a table', command=self.select_file)
        self.file_button.pack()

        self.columns_label = tk.Label(
            self, text='Select columns for metadata features:')
        self.columns_label.pack()

        self.columns_listbox = tk.Listbox(self, selectmode='multiple')
        self.columns_listbox.pack()

        self.selection_var = tk.IntVar()
        self.selection_check = tk.Checkbutton(
            self, text="Add 'selection' column", variable=self.selection_var)
        self.selection_check.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a table')
        if file_path:
            self.file_label['text'] = file_path
            self.load_columns(file_path)

    def load_columns(self, file_path):
        self.table = data_loader.load_data(file_path)

        self.columns_listbox.delete(0, tk.END)  # clear the listbox
        for col in self.table.columns:
            self.columns_listbox.insert(tk.END, col)


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('VoxRimor')
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.welcome = tk.Label(self, text='Welcome to VoxRimor!')
        self.welcome.pack(side='top')

        self.visualizer_button = tk.Button(self)
        self.visualizer_button['text'] = 'Visualizer mode'
        self.visualizer_button['command'] = self.visualizer_mode

        self.visualizer_button.pack(side='left')

        self.feature_extractor_button = tk.Button(self)
        self.feature_extractor_button['text'] = 'Feature extraction and visualization mode'
        self.feature_extractor_button['command'] = self.feature_extractor_mode
        self.feature_extractor_button.pack(side='right')

    def visualizer_mode(self):
        self.visualizer_window = VisualizerWindow(self)

    def feature_extractor_mode(self):
        messagebox.showinfo(
            "Info", "Feature extraction and visualization mode selected")


root = tk.Tk()
app = Application(master=root)
app.mainloop()
