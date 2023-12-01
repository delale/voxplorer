"""
Graphical User Interface for VoxRimor built using tkinter.
"""

import tkinter as tk

class gui():
    """
    Tkinter GUI for VoxRimor.

    Parameters:
    -----------
    """
    def __init__(self):
        self.root = tk.Tk()     # root window

        self.label = tk.Label(self.root, text="Test message", font=('Arial', 18))
        self.label.pack(padx=10, pady=10)

        self.savepath = tk.Text(self.root, height=1, font=('Arial', 16))
        self.savepath.pack(padx=10, pady=10)

        self.root.mainloop()    # run mainloop

if __name__ == '__main__':
    gui()