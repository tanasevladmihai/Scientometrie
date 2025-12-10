import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
from pathlib import Path
import subprocess
from threading import Thread
import queue

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from functionalities.normalize_core import build_core_yearly
from functionalities.normalize_journal import build_yearly_outputs
from functionalities.reformat import main as reformat_main

class ProcessRunner:
    def __init__(self, target, args, kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.process = None
        self.log_queue = queue.Queue()
        self.thread = None

    def start(self):
        self.thread = Thread(target=self._run)
        self.thread.start()

    def _run(self):
        try:
            #this thing redirects stdout and stderr
            p = subprocess.Popen(
                [sys.executable, "-c", f"from {self.target.__module__} import {self.target.__name__}; {self.target.__name__}(*{self.args}, **{self.kwargs})"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            self.process = p
            for line in iter(p.stdout.readline, ''):
                self.log_queue.put(line)
            p.stdout.close()
            p.wait()
        except Exception as e:
            self.log_queue.put(f"Error running process: {e}\n")
        finally:
            self.log_queue.put(None) # Signal end of logs

    def is_running(self):
        return self.thread and self.thread.is_alive()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scientometrie GUI")
        self.geometry("800x600")

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.core_frame = self.create_section(scrollable_frame, "1. Normalize Core Files")
        self.core_input_var = self.create_folder_selection(self.core_frame, "Input Folder:", "core_raw")
        self.core_output_var = self.create_folder_selection(self.core_frame, "Output Folder:", "out/core")
        self.core_mode = self.create_radio_buttons(self.core_frame, {
            "Normalize all files": "all",
            "Select specific files": "select",
            "Skip normalization": "skip"
        }, self.toggle_core_files)
        self.core_file_listbox = self.create_file_listbox(self.core_frame)
        self.core_console = self.create_console(self.core_frame)
        self.core_button = ttk.Button(self.core_frame, text="Confirm and Proceed", command=self.run_core_normalization)
        self.core_button.pack(pady=5)

        self.journal_frame = self.create_section(scrollable_frame, "2. Normalize Journal Files")
        self.journal_input_var = self.create_folder_selection(self.journal_frame, "Input Folder:", "journal_raw")
        self.journal_output_var = self.create_folder_selection(self.journal_frame, "Output Folder:", "out/journal")
        self.journal_mode = self.create_radio_buttons(self.journal_frame, {
            "Process all files": "all",
            "Select specific files": "select",
            "Skip normalization": "skip"
        }, self.toggle_journal_files)
        self.journal_file_listbox = self.create_file_listbox(self.journal_frame)
        self.journal_console = self.create_console(self.journal_frame)
        self.journal_button = ttk.Button(self.journal_frame, text="Confirm and Proceed", command=self.run_journal_normalization)
        self.journal_button.pack(pady=5)

        self.reformat_frame = self.create_section(scrollable_frame, "3. Reformat Files")
        self.reformat_input_var = self.create_folder_selection(self.reformat_frame, "Input Folder:", "exports")
        self.reformat_output_var = self.create_folder_selection(self.reformat_frame, "Output Folder:", "out")
        self.reformat_mode = self.create_radio_buttons(self.reformat_frame, {
            "Process all files": "all",
            "Select specific files": "select",
            "Skip processing": "skip"
        }, self.toggle_reformat_files)
        self.reformat_file_listbox = self.create_file_listbox(self.reformat_frame)
        self.reformat_console = self.create_console(self.reformat_frame)
        self.reformat_button = ttk.Button(self.reformat_frame, text="Run Reformatting", command=self.run_reformatting)
        self.reformat_button.pack(pady=5)

        self.set_initial_state()

    def create_section(self, parent, text):
        frame = ttk.LabelFrame(parent, text=text, padding="10")
        frame.pack(fill=tk.X, padx=10, pady=5)
        return frame

    def create_folder_selection(self, parent, label_text, default_folder):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        label = ttk.Label(frame, text=label_text, width=15)
        label.pack(side=tk.LEFT)
        
        folder_var = tk.StringVar(value=str(project_root / default_folder))
        entry = ttk.Entry(frame, textvariable=folder_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        button = ttk.Button(frame, text="Browse...", command=lambda: self.browse_folder(folder_var))
        button.pack(side=tk.LEFT, padx=5)
        return folder_var

    def create_radio_buttons(self, parent, options, command):
        var = tk.StringVar(value=list(options.values())[0])
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        for text, value in options.items():
            rb = ttk.Radiobutton(frame, text=text, variable=var, value=value, command=command)
            rb.pack(side=tk.LEFT, padx=5)
        return var

    def create_file_listbox(self, parent):
        frame = ttk.Frame(parent)
        listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=5)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return {"frame": frame, "listbox": listbox}

    def create_console(self, parent):
        frame = ttk.Frame(parent, height=100)
        frame.pack(fill=tk.X, pady=5)
        console = tk.Text(frame, height=6, state='disabled', bg='black', fg='white', font=("Courier", 9))
        scrollbar = ttk.Scrollbar(frame, command=console.yview)
        console.config(yscrollcommand=scrollbar.set)
        
        console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return console

    def browse_folder(self, var):
        folder = filedialog.askdirectory(initialdir=project_root)
        if folder:
            var.set(folder)

    def set_initial_state(self):
        self.toggle_core_files()
        self.toggle_journal_files()
        self.toggle_reformat_files()
        self.set_frame_state(self.journal_frame, 'disabled')
        self.set_frame_state(self.reformat_frame, 'disabled')

    def set_frame_state(self, frame, state):
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                self.set_frame_state(child, state)

    def toggle_core_files(self):
        if self.core_mode.get() == "select":
            self.core_file_listbox["frame"].pack(fill=tk.X, padx=10, pady=5)
            self.populate_file_list(self.core_input_var, self.core_file_listbox["listbox"], "*.csv")
        else:
            self.core_file_listbox["frame"].pack_forget()

    def toggle_journal_files(self):
        if self.journal_mode.get() == "select":
            self.journal_file_listbox["frame"].pack(fill=tk.X, padx=10, pady=5)
            self.populate_file_list(self.journal_input_var, self.journal_file_listbox["listbox"], "*.xls*")
        else:
            self.journal_file_listbox["frame"].pack_forget()

    def toggle_reformat_files(self):
        if self.reformat_mode.get() == "select":
            self.reformat_file_listbox["frame"].pack(fill=tk.X, padx=10, pady=5)
            self.populate_file_list(self.reformat_input_var, self.reformat_file_listbox["listbox"], ("*.csv", "*.xls*"))
        else:
            self.reformat_file_listbox["frame"].pack_forget()

    def populate_file_list(self, folder_var, listbox, patterns):
        listbox.delete(0, tk.END)
        folder = Path(folder_var.get())
        if not folder.is_dir():
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        
        files = []
        for pattern in patterns:
            files.extend(folder.glob(pattern))
            
        for f in sorted(files):
            listbox.insert(tk.END, f.name)

    def run_core_normalization(self):
        mode = self.core_mode.get()
        if mode == "skip":
            self.log_to_console(self.core_console, "Core normalization skipped.\n")
            self.set_frame_state(self.core_frame, 'disabled')
            self.set_frame_state(self.journal_frame, 'normal')
            return

        input_folder = self.core_input_var.get()
        output_folder = self.core_output_var.get()
        file_list = None

        if mode == "select":
            selected_indices = self.core_file_listbox["listbox"].curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select files to normalize.")
                return
            file_list = [self.core_file_listbox["listbox"].get(i) for i in selected_indices]

        self.core_button.config(state='disabled')
        runner = ProcessRunner(
            target=build_core_yearly,
            args=(),
            kwargs={'input_folder': input_folder, 'output_dir': output_folder, 'file_list': file_list}
        )
        runner.start()
        self.monitor_process(runner, self.core_console, lambda: [
            self.set_frame_state(self.core_frame, 'disabled'),
            self.set_frame_state(self.journal_frame, 'normal')
        ])

    def run_journal_normalization(self):
        mode = self.journal_mode.get()
        if mode == "skip":
            self.log_to_console(self.journal_console, "Journal normalization skipped.\n")
            self.set_frame_state(self.journal_frame, 'disabled')
            self.set_frame_state(self.reformat_frame, 'normal')
            return

        input_folder = self.journal_input_var.get()
        output_folder = self.journal_output_var.get()
        file_list = None

        if mode == "select":
            selected_indices = self.journal_file_listbox["listbox"].curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select files to process.")
                return
            #file_list = [self.journal_file_listbox["listbox"].get(i) for i in selected_.get(i) for i in selected_indices]
            file_list = [self.journal_file_listbox["listbox"].get(i) for i in selected_indices]
        self.journal_button.config(state='disabled')
        runner = ProcessRunner(
            target=build_yearly_outputs,
            args=(),
            kwargs={'input_folder': input_folder, 'output_dir': output_folder, 'file_list': file_list}
        )
        runner.start()
        self.monitor_process(runner, self.journal_console, lambda: [
            self.set_frame_state(self.journal_frame, 'disabled'),
            self.set_frame_state(self.reformat_frame, 'normal')
        ])

    def run_reformatting(self):
        mode = self.reformat_mode.get()
        if mode == "skip":
            self.log_to_console(self.reformat_console, "Reformatting skipped. Program finished.\n")
            self.reformat_button.config(state='disabled')
            return

        input_dir = self.reformat_input_var.get()
        output_dir = self.reformat_output_var.get()
        journal_dir = self.journal_output_var.get()
        core_dir = self.core_output_var.get()
        file_list = None

        if mode == "select":
            selected_indices = self.reformat_file_listbox["listbox"].curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select files for reformatting.")
                return
            file_list = [self.reformat_file_listbox["listbox"].get(i) for i in selected_indices]

        self.reformat_button.config(state='disabled')
        runner = ProcessRunner(
            target=reformat_main,
            args=(),
            kwargs={
                'input_dir': input_dir,
                'output_dir': output_dir,
                'file_list': file_list,
                'journal_dir': journal_dir,
                'core_dir': core_dir
            }
        )
        runner.start()
        self.monitor_process(runner, self.reformat_console, lambda: self.log_to_console(self.reformat_console, "All processes finished.\n"))

    def log_to_console(self, console, message):
        console.config(state='normal')
        console.insert(tk.END, message)
        console.see(tk.END)
        console.config(state='disabled')

    def monitor_process(self, runner, console, on_complete):
        try:
            while True:
                line = runner.log_queue.get_nowait()
                if line is None:
                    self.log_to_console(console, "\nProcess finished.\n")
                    on_complete()
                    break
                self.log_to_console(console, line)
        except queue.Empty:
            self.after(100, lambda: self.monitor_process(runner, console, on_complete))

if __name__ == "__main__":
    app = App()
    app.mainloop()