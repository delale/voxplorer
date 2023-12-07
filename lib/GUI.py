"""
Graphical User Interface for VoxRimor built using tkinter.
"""

from collections import defaultdict
import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import threading
from typing import Callable
import webbrowser
import numpy as np
import pandas as pd
import data_loader
import embedding_projector
from feature_extraction import FeatureExtractor
from speaker_embeddings import SpeakerEmbedder


# TODO: test with excel files


class VisualizerWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Visualizer mode')
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self)
        file_frame.pack()

        self.file_label = tk.Label(file_frame, text='Table path:')
        self.file_label.grid(row=0, column=0, columnspan=2)

        self.file_entry = tk.Entry(file_frame)
        self.file_entry.grid(row=1, column=0, sticky='w')

        self.file_button = tk.Button(
            file_frame, text='Browse', command=self.select_file)
        self.file_button.grid(row=1, column=1, sticky='e')

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.columns_label = tk.Label(
            outer_frame, text='Select columns for metadata features:')
        self.columns_label.pack()

        self.columns_listbox = tk.Listbox(outer_frame, selectmode='multiple')
        self.columns_listbox.pack()

        self.selection_var = tk.IntVar()
        self.selection_check = tk.Checkbutton(
            outer_frame, text="Add 'selection' column", variable=self.selection_var)
        self.selection_check.pack()

        self.continue_button = tk.Button(
            outer_frame, text='Continue', command=self.open_projector)
        self.continue_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a directory of WAV files or a WAV file')
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.load_columns(file_path)

    def load_columns(self, file_path):
        try:
            self.table = data_loader.load_data(file_path)
        except UnicodeDecodeError:
            messagebox.showerror(
                "Error", "Cannot read the file. Please check the encoding.")
            return

        self.columns_listbox.delete(0, tk.END)  # clear the listbox
        for col in self.table.columns:
            self.columns_listbox.insert(tk.END, col)

    def open_projector(self):
        if not self.file_entry.get():
            messagebox.showerror(
                "Error", "Please select a data table.")
            return

        # Get metavars
        selected_indices = self.columns_listbox.curselection()
        selected_columns = [self.columns_listbox.get(
            i) for i in selected_indices]
        add_selection_column = bool(self.selection_var.get())

        # Load data
        X, Y, metavars = data_loader.split_data(
            df=self.table, features=None, metavars=selected_columns,
            add_selection_column=add_selection_column
        )

        # Start projector in window
        ProjectorWindow(self, X, Y, metavars)


class FeatureExtractorWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Feature extraction mode')
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self)
        file_frame.pack()

        self.file_label = tk.Label(
            file_frame, text='Directory or WAV file path:')
        self.file_label.grid(row=0, column=0, columnspan=3)

        self.file_entry = tk.Entry(file_frame)
        self.file_entry.grid(row=1, column=0, sticky='w')

        self.file_button = tk.Button(
            file_frame, text='Browse file', command=self.select_file)
        self.file_button.grid(row=1, column=1, sticky='e')

        self.dir_button = tk.Button(
            file_frame, text='Browse directory', command=self.select_dir)
        self.dir_button.grid(row=1, column=2, sticky='e')

        self.out_dir_label = tk.Label(
            file_frame, text='Output directory & Filename:')
        self.out_dir_label.grid(row=2, column=0, columnspan=3)

        self.out_dir_entry = tk.Entry(file_frame)
        self.out_dir_entry.grid(row=3, column=0, sticky='w')

        self.out_file.entry = tk.Entry(file_frame)
        self.out_file_entry.grid(row=3, column=1, sticky='e')

        self.out_dir_button = tk.Button(
            file_frame, text='Browse', command=self.select_out_dir)
        self.out_dir_button.grid(row=3, column=2, sticky='e')

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.mode_var = tk.StringVar(value="speaker embeddings")
        self.speaker_embeddings_radio = tk.Radiobutton(
            outer_frame, text="Speaker embeddings", variable=self.mode_var,
            value="speaker embeddings", command=self.update_methods_listbox)
        self.speaker_embeddings_radio.pack()

        self.feature_extraction_radio = tk.Radiobutton(
            outer_frame, text="Feature extraction", variable=self.mode_var,
            value="feature extraction", command=self.update_methods_listbox
        )
        self.feature_extraction_radio.pack()

        self.methods_label = tk.Label(
            outer_frame, text='Select feature extraction methods:')
        self.methods_listbox = tk.Listbox(self, selectmode='multiple')
        for method in [
            "Mel Features",
            "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)",
            "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)",
            "Liner Predictive Cepstral Coefficients"
        ]:
            self.methods_listbox.insert(tk.END, method)

        self.continue_button = tk.Button(outer_frame, text='Continue',
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

    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a directory of WAV files or a WAV file')
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def select_dir(self):
        dir_path = filedialog.askdirectory(
            initialdir='.', title='Select a directory of WAV files or a WAV file')
        if dir_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, dir_path)

    def select_out_dir(self):
        dir_path = filedialog.askdirectory(
            initialdir='.', title='Select a directory')
        if dir_path:
            self.out_dir_entry.delete(0, tk.END)
            self.out_dir_entry.insert(0, dir_path)

    def open_methods_window(self):
        if not self.file_entry.get():
            messagebox.showerror(
                "Error", "Please select a data table.")
            return

        if self.mode_var.get() == "feature extraction":
            self.feature_methods = {}
            self.method_translator = {
                "Mel Features": "mel_features",
                "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)": "acoustic_features",
                "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)": "low_lvl_features",
                "Liner Predictive Cepstral Coefficients": "lpc_features"
            }
            selected_indices = self.methods_listbox.curselection()
            selected_methods = [self.methods_listbox.get(
                i) for i in selected_indices]
            MethodsWindow(self, selected_methods, self.store_parameter_values)

        else:
            self.filename = self.file_entry.get()
            if os.path.isdir(self.filename):
                for f in os.listdir(self.filename):
                    if f.endswith('.wav'):
                        filename = os.path.splitext(os.path.basename(f))[0]
                        break
            else:
                filename = os.path.splitext(os.path.basename(self.filename))[0]
            self.metadata_vars = {'metavars': None, 'separator': None}
            MetadataWindow(self, filename, self.store_metadata_specifications)

    def store_parameter_values(self, method, parameter_values):
        self.feature_methods[self.method_translator[method]] = parameter_values

        # Check if all methods have been processed
        if len(self.feature_methods) == len(self.methods_listbox.curselection()):
            self.filename = self.file_entry.get()
            if os.path.isdir(self.filename):
                for f in os.listdir(self.filename):
                    if f.endswith('.wav'):
                        filename = os.path.splitext(os.path.basename(f))[0]
                        break
            else:
                filename = os.path.splitext(os.path.basename(self.filename))[0]
            self.metadata_vars = {'metavars': None, 'separator': None}
            MetadataWindow(self, filename, self.store_metadata_specifications)

    def store_metadata_specifications(self, metavars: list, separator: str, add_selection_column: bool):
        self.metadata_vars['metavars'] = metavars
        self.metadata_vars['separator'] = separator
        self.metadata_vars['add_selection_column'] = add_selection_column

        # Add a start analysis button
        self.start_analysis_button = tk.Button(
            self, text='Start analysis', command=self.feature_extraction_pipeline)
        self.start_analysis_button.pack()

        self.deiconify()

    def feature_extraction_pipeline(self):
        if self.mode_var.get() == "feature extraction":
            featureExtractor = FeatureExtractor(
                audio_dir=self.filename, feature_methods=self.feature_methods,
                metadata_vars=self.metadata_vars
            )
            print("Feature extraction pipeline")
        else:
            featureExtractor = SpeakerEmbedder(
                audio_dir=self.filename, metadata_vars=self.metadata_vars
            )
            print("Speaker embeddings pipeline")

        X, Y, metavars, feature_labels = featureExtractor.process_files()  # TODO: test me!
        if self.out_dir_entry.get():
            out_dir = self.out_dir_entry.get()
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            if self.out_file_entry.get():
                out_file = os.path.splitext(self.out_file_entry.get())[0]
            else:
                out_file = 'features'

            # TODO: add feature labels returned by featureExtractor
            df = pd.DataFrame(X, columns=feature_labels)

    # TODO: add metadata specification,
    #       parameter specs dict, tensorboard components (incl. project button)


class MethodsWindow(tk.Toplevel):
    def __init__(self, master=None, methods: list = None, callback: Callable = None, index: int = 0):
        super().__init__(master)
        self.title('Feature extraction')
        self.methods = methods
        self.callback = callback
        self.index = index
        self.method = methods[index]
        self.parameters = {}
        self.create_widgets()

    def create_widgets(self):
        self.method_label = tk.Label(
            self, text="Set parameters for " + self.method)
        self.method_label.grid(row=0, column=0, columnspan=2)

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

        for i, (parameter, default_value) in enumerate(parameters[self.method].items()):
            label = tk.Label(self, text=parameter, width=20, anchor='w')
            label.grid(row=i+1, column=0, sticky='w')

            # Checkbutton for boolean parameters
            if isinstance(default_value, bool):
                var = tk.BooleanVar(value=default_value)
                widget = tk.Checkbutton(self, variable=var)
            elif isinstance(default_value, float):
                var = tk.DoubleVar(value=default_value)
                widget = tk.Entry(self, textvariable=var)
            else:
                var = tk.IntVar(value=default_value)
                widget = tk.Entry(self, textvariable=var)
            widget.grid(row=i+1, column=1, sticky='e')

            # Store variable in the parameter dictionary and default value for type checking
            self.parameters[parameter] = (var, default_value)

        self.continue_button = tk.Button(
            self, text='Continue', command=self.save_parameters)
        self.continue_button.grid(
            row=len(parameters[self.method])+1, column=0, columnspan=2)

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
        self.index += 1
        if self.index < len(self.methods):
            # Open the next MethodsWindow
            MethodsWindow(self.master, self.methods,
                          self.callback, self.index)
        self.destroy()  # close the current window

    def destroy(self) -> None:
        self.withdraw()


class MetadataWindow(tk.Toplevel):
    def __init__(self, master=None, filename: str = None, callback: Callable = None):
        super().__init__(master)
        self.title('Metadata specification')
        self.filename = filename
        self.separator = tk.StringVar(value='_')
        self.callback = callback
        self.separator.trace('w', self.update_filename_parts)
        self.create_widgets()

    def create_widgets(self):
        separator_frame = tk.Frame(self)
        separator_frame.pack()

        self.separator_label = tk.Label(separator_frame, text='Separator:')
        self.separator_label.grid(row=0, column=0, sticky='w')

        self.separator_value = tk.Entry(
            separator_frame, textvariable=self.separator)
        self.separator_value.grid(row=0, column=1, sticky='e')

        self.filename_parts_frame = tk.Frame(self)
        self.filename_parts_frame.pack()

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.selection_var = tk.IntVar()
        self.selection_check = tk.Checkbutton(
            outer_frame, text="Add 'selection' column", variable=self.selection_var)
        self.selection_check.pack()

        self.continue_button = tk.Button(
            outer_frame, text='Continue', command=self.save_metadata)
        self.continue_button.pack()

        # Call update_filename_parts to display the filename parts with default separator
        self.update_filename_parts()

    def update_filename_parts(self, *args):
        for widget in self.filename_parts_frame.winfo_children():
            widget.destroy()

        if self.separator.get() != '':
            self.current_separator = self.separator.get()
        filename_parts = self.filename.split(self.current_separator)
        metadata_spec_label = tk.Label(
            self.filename_parts_frame, text='Metadata specification:')
        metadata_spec_label.grid(row=0, column=0, columnspan=2)
        for i, part in enumerate(filename_parts):
            label = tk.Label(self.filename_parts_frame, text=part)
            label.grid(row=i + 1, column=0, sticky='w')

            entry = tk.Entry(self.filename_parts_frame)
            entry.grid(row=i + 1, column=1, sticky='e')

    def save_metadata(self):
        metadata_spec = []
        for widget in self.filename_parts_frame.winfo_children():
            if isinstance(widget, tk.Entry):
                metadata_spec.append(widget.get())
        add_selection_column = bool(self.selection_var.get())
        if self.callback is not None:
            self.callback(metadata_spec, self.current_separator,
                          add_selection_column)
        self.destroy()

    def destroy(self) -> None:
        self.withdraw()


class ProjectorWindow(tk.Toplevel):
    def __init__(self, master=None, X: np.ndarray = None, Y: np.ndarray = None, metavars: list = None):
        super().__init__(master)
        self.title('Projector')
        self.X = X
        self.Y = Y
        self.metavars = metavars
        self.create_widgets()

    def create_widgets(self):
        self.projector_label = tk.Label(self, text='Projector')
        self.projector_label.pack()

        self.project_button = tk.Button(
            self, text='Project!', command=self.start_projector)
        self.project_button.pack()

    def start_projector(self):
        # Start projector in separate thread
        self.projector_thread = threading.Thread(
            target=self.embedding_projector, args=())
        self.projector_thread.start()

        # Add button to stop analysis
        self.stop_button = tk.Button(self, text='Stop projector',
                                     command=self.stop_projector)
        self.stop_button.pack()

    def embedding_projector(self):
        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # Run embedding projector
        self.tbTool = embedding_projector.TensorBoardTool(
            log_dir=log_dir, embedding_vecs=self.X, metadata=self.Y,
            metadata_var=self.metavars
        )

        self.board, url = self.tbTool.run()
        webbrowser.open(url)

    def stop_projector(self):
        # if self.projector_thread:
        #     self.projector_thread.terminate()
        #     self.board = None
        #     self.stop_button.destroy()
        if self.board:
            self.tbTool._shutdown()
            self.board = None
            self.stop_button.destroy()


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
