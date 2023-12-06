"""
Graphical User Interface for VoxRimor built using tkinter.
"""

from collections import defaultdict
import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import threading
import webbrowser
import data_loader
import embedding_projector


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

        self.project_button = tk.Button(
            self, text='Project!', command=self.start_projector)
        self.project_button.pack()

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

    def start_projector(self):
        # Get metavars
        selected_indices = self.columns_listbox.curselection()
        selected_columns = [self.columns_listbox.get(
            i) for i in selected_indices]

        add_selection_column = bool(self.selection_var.get())

        # Start projector in separate thread
        self.projector_thread = threading.Thread(
            target=self.embedding_projector, args=(selected_columns, add_selection_column))
        self.projector_thread.start()

        # Add button to stop analysis
        self.stop_button = tk.Button(self, text='Stop projector',
                                     command=self.stop_projector)
        self.stop_button.pack()

    def embedding_projector(self, selected_columns, add_selection_column):
        ptd = os.path.join(os.getcwd(), self.file_label['text'])
        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # Run the split_data function
        X, Y, metavars = data_loader.split_data(
            df=self.table, features=None, metavars=selected_columns,
            add_selection_column=add_selection_column
        )

        # Run embedding projector
        self.tbTool = embedding_projector.TensorBoardTool(
            log_dir=log_dir, embedding_vecs=X, metadata=Y, metadata_var=metavars
        )

        self.board, url = self.tbTool.run()
        webbrowser.open(url)

    def stop_projector(self):
        # if self.projector_thread:
        #     self.projector_thread.ter
        #     self.board = None
        #     self.stop_button.destroy()
        if self.board:
            self.tbTool._shutdown()
            self.board = None
            self.stop_button.destroy()


class FeatureExtractorWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Feature extraction mode')
        self.create_widgets()

    def create_widgets(self):
        self.mode_var = tk.StringVar(value="speaker embeddings")
        self.speaker_embeddings_radio = tk.Radiobutton(
            self, text="Speaker embeddings", variable=self.mode_var,
            value="speaker embeddings", command=self.update_methods_listbox)
        self.speaker_embeddings_radio.pack()

        self.feature_extraction_radio = tk.Radiobutton(
            self, text="Feature extraction", variable=self.mode_var,
            value="feature extraction", command=self.update_methods_listbox
        )
        self.feature_extraction_radio.pack()

        self.methods_label = tk.Label(
            self, text='Select feature extraction methods:')
        self.methods_listbox = tk.Listbox(self, selectmode='multiple')
        for method in [
            "Mel Features",
            "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)",
            "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)",
            "Liner Predictive Cepstral Coefficients"
        ]:
            self.methods_listbox.insert(tk.END, method)

        self.continue_button = tk.Button(self, text='Continue',
                                         command=self.open_methods_window)
        self.continue_button.pack()

    def update_methods_listbox(self):
        if self.mode_var.get() == "feature extraction":
            self.methods_label.pack(padx=10, pady=10)
            self.methods_listbox.pack(
                fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            self.methods_label.pack_forget()
            self.methods_listbox.pack_forget()

    def open_methods_window(self):
        if self.mode_var.get() == "speaker embeddings":
            SpeakerEmbeddingsWindow(self)
        elif self.mode_var.get() == "feature extraction":
            selected_indices = self.methods_listbox.curselection()
            selected_methods = [self.methods_listbox.get(
                i) for i in selected_indices]
            for method in selected_methods:
                MethodsWindow(self, method, self.store_parameter_values)

    def store_parameter_values(self, method, parameter_values):
        print("Stored parameter values for " + method +
              ":", parameter_values)  # debugging

    # TODO: add metadata specification, directory selection before everything else,
    #       parameter specs dict, tensorboard components (incl. project button)


class MethodsWindow(tk.Toplevel):
    def __init__(self, master=None, method: str = None, callback=None):
        super().__init__(master)
        self.title('Feature extraction')
        self.method = method
        self.callback = callback
        self.parameters = {}
        self.create_widgets()

    def create_widgets(self):
        self.method_label = tk.Label(
            self, text="Set parameters for " + self.method)
        self.method_label.pack()

        # Define parameters for each method
        parameters = {
            "Mel Features": {
                "n_mfcc": 13,
                "n_mels": 40,
                "win_length": 25.0,
                "overlap": 10.0,
                "fmin": 150.0,
                "fmax": 4000.0,
                "preemphasis": 0.95,
                "lifter": 22.0,
                "deltas": True,
                "summarise": True
            },
            "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)": {
                "f0min": 75.0, "f0max": 600.0
            },
            "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)": {
                "win_length": 25.0,
                "overlap": 10.0,
                "preemphasis": 0.95,
                "use_mean_contrasts": True,
                "summarise": True
            },
            "Liner Predictive Cepstral Coefficients": {
                "n_lpcc": 13,
                "win_length": 25.0,
                "overlap": 10.0,
                "preemphasis": 0.95,
                "order": 16,
                "summarise": True
            }
        }

        for parameter, default_value in parameters[self.method].items():
            frame = tk.Frame(self)
            frame.pack(fill=tk.X, expand=True, padx=10, pady=10)

            label = tk.Label(frame, text=parameter, width=20, anchor='w')
            label.pack(side=tk.LEFT)

            # Checkbutton for boolean parameters
            if isinstance(default_value, bool):
                var = tk.BooleanVar(value=default_value)
                widget = tk.Checkbutton(frame, variable=var)
            elif isinstance(default_value, float):
                var = tk.DoubleVar(value=default_value)
                widget = tk.Entry(frame, textvariable=var)
            else:
                var = tk.IntVar(value=default_value)
                widget = tk.Entry(frame, textvariable=var)
            widget.pack(side=tk.RIGHT, expand=True, fill=tk.X)

            # Store variable in the parameter dictionary and default value for type checking
            self.parameters[parameter] = (var, default_value)

        self.continue_button = tk.Button(
            self, text='Continue', command=self.save_parameters)
        self.continue_button.pack(pady=10)

    def save_parameters(self):
        # Save parameters to a dictionary
        parameters_dict = {}
        for parameter, (var, default_value) in self.parameters.items():
            try:
                parameters_dict[parameter] = var.get()
            except tk.TclError:
                messagebox.showerror(
                    "Error", "Invalid value for " + parameter + ".")
                var.set(default_value)
                return
        if self.callback is not None:
            self.callback(self.method, parameters_dict)

        self.destroy()


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
        self.feature_extractor_window = FeatureExtractorWindow(self)


root = tk.Tk()
app = Application(master=root)
app.mainloop()
