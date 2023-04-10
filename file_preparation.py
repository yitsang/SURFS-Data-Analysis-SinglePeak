#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:05:55 2023
This Code is based on Te-wei Tsai's code on python 2 environment
Only transfer the code from python 2 to 3 for easier use
No more modifications in this code version
@author: Yi Zeng
"""

import tkinter as tk
from tkinter import filedialog, messageboxforce_analysis_gui_tk
import numpy as np
import pandas as pd
import os

# Global variables
data_x = []
data_y = []
dirname = ''
file_name = []

# Load the data
def load_data():
    global data_x, data_y, dirname
    file_path = filedialog.askopenfilename()
    dirname = os.path.dirname(file_path)

    # Get the position and signal
    data_signal = np.fromfile(file_path, sep=" ")
    n = data_signal.size   # Check the length of data
    data_base = data_signal.reshape((n//2, 2))   # Reshape the data to a matrix 
    data_positon_base = np.array(data_base[:, 0]) # Position
    data_base = np.array(data_base[:, 1])         # Signal

    # Remove the data of zero position
    itemindex = np.where(data_positon_base != 0)
    data_x = data_positon_base[itemindex]
    data_y = data_base[itemindex]

# Load the CSV file
def load_csv():
    global file_name
    file_path = filedialog.askopenfilename()
    info_csv = pd.read_csv(file_path)
    info_csv = info_csv.values
    num_file, num_index = info_csv.shape
    file_name = []
    for ii in range(num_file):
        temp = 'lock'
        for jj in range(num_index):
            temp = temp + '_' + str(info_csv[ii][jj])
        file_name.append(temp)

def generate_file():
    global file_name
    if len(data_x) == 0:
        messagebox.showwarning('Warning', 'No data loaded!')
        return
    
    # Ask user for filename
    file_path = filedialog.asksaveasfilename(defaultextension='.txt')
    if file_path == '':
        return
    
    # Get the local max and min
    temp = np.array(np.diff(data_x) < 0, dtype=int) # The "int" type is helpful for the consistency.
    # Enforce the consistency
    temp0 = np.concatenate((np.array([0]), temp[0:temp.size-1]), axis=0)
    temp = (temp + temp0) // 2

    # Get the indices of local max and min
    temp1 = np.diff(temp)
    temp_max = np.where(temp1 == 1)[0] + 1  # local max, "+1" is for the compensation from the 'diff' function.
    temp_min = np.where(temp1 == -1)[0]     # local min

    # Save the data to files
    for ii in range(temp_max.size):
        if ii != temp_max.size - 1:
            temp_lock_x = data_x[temp_max[ii]:temp_min[ii]]
            temp_lock_y = data_y[temp_max[ii]:temp_min[ii]]
        else:
            temp_lock_x = data_x[temp_max[ii]:]
            temp_lock_y = data_y[temp_max[ii]:]

        data_xy = np.column_stack((temp_lock_x, temp_lock_y))
        np.savetxt(file_path.format(ii+1), data_xy)

    # Show a message box upon completion
    messagebox.showinfo('Success', 'Files saved successfully!')


# Create the GUI
root = tk.Tk()
root.title('Data Processing')
root.geometry('300x200')

# Add the buttons
button_load = tk.Button(text='Load Data', command=load_data)
button_load.pack(pady=10)

button_load_csv = tk.Button(text='Load CSV', command=load_csv)
button_load_csv.pack(pady=10)

button_gen = tk.Button(text='Generate Files', command=generate_file)
button_gen.pack(pady=10)

# Quit the GUI
button_quit = tk.Button(text='Quit', command=root.destroy)
button_quit.pack(pady=10)

# Execute the GUI
root.mainloop()
