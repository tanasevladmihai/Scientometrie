import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import logging
from pathlib import Path
from threading import Thread
import queue

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from functionalities.normalize_core import build_core_yearly
from functionalities.normalize_journal import build_yearly_outputs
from functionalities.reformat import main as reformat_main

# ---------- Logging Utilities ----------

class QueueHandler(logging.Handler):
    """Sends log records to a queue for the GUI to consume."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record) + "\n")

# ---------- Execution Engine ----------

class WorkerThread(Thread):
    def __init__(self, target, kwargs, on_complete, on_error):
        super().__init__(daemon=True)
        self.target = target
        self.kwargs = kwargs
        self.on_complete = on_complete
        self.on_error = on_error

    def run(self):
        try:
            self.target(**self.kwargs)
            self.on_complete()
        except Exception as e:
            self.on_error(e)

# ---------- GUI Application ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scientometrie v2 - Refactored")
        self.geometry("850x700")
        
        self.log_queue = queue.Queue()
        self._setup_logging()
        self._create_widgets()
        self._set_initial_state()
        self._poll_log_queue()

    def _setup_logging(self):
        # Root logger setup
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Handler for the GUI console
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

    def _create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Section 1: CORE
        self.core_frame = self._create_section("1. Normalize Core Files")
        self.core_input = self._create_path_selector(self.core_frame, "Input:", "core_raw")
        self.core_output = self._create_path_selector(self.core_frame, "Output:", "out/core")
        self.core_mode = self._create_mode_selector(self.core_frame, self._toggle_core_list)
        self.core_list = self._create_file_list(self.core_frame)
        self.core_console = self._create_console(self.core_frame)
        self.core_btn = ttk.Button(self.core_frame, text="Process CORE", command=self._run_core)
        self.core_btn.pack(pady=5)

        # Section 2: Journal
        self.journal_frame = self._create_section("2. Normalize Journal Files")
        self.journal_input = self._create_path_selector(self.journal_frame, "Input:", "journal_raw")
        self.journal_output = self._create_path_selector(self.journal_frame, "Output:", "out/journal")
        self.journal_mode = self._create_mode_selector(self.journal_frame, self._toggle_journal_list)
        self.journal_list = self._create_file_list(self.journal_frame)
        self.journal_console = self._create_console(self.journal_frame)
        self.journal_btn = ttk.Button(self.journal_frame, text="Process Journals", command=self._run_journal)
        self.journal_btn.pack(pady=5)

        # Section 3: Reformat
        self.reformat_frame = self._create_section("3. Final Reformatting")
        self.reformat_input = self._create_path_selector(self.reformat_frame, "Input:", "exports")
        self.reformat_output = self._create_path_selector(self.reformat_frame, "Output:", "out")
        self.reformat_mode = self._create_mode_selector(self.reformat_frame, self._toggle_reformat_list)
        self.reformat_list = self._create_file_list(self.reformat_frame)
        self.reformat_console = self._create_console(self.reformat_frame)
        self.reformat_btn = ttk.Button(self.reformat_frame, text="Run Final Reformat", command=self._run_reformat)
        self.reformat_btn.pack(pady=5)

    # --- Widget Helpers ---

    def _create_section(self, title):
        f = ttk.LabelFrame(self.scrollable_frame, text=title, padding=10)
        f.pack(fill=tk.X, padx=10, pady=5)
        return f

    def _create_path_selector(self, parent, label, default):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=10).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(project_root / default))
        ttk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f, text="...", width=3, command=lambda: self._browse(var)).pack(side=tk.LEFT, padx=2)
        return var

    def _create_mode_selector(self, parent, command):
        v = tk.StringVar(value="all")
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        for text, val in [("All", "all"), ("Select", "select"), ("Skip", "skip")]:
            ttk.Radiobutton(f, text=text, variable=v, value=val, command=command).pack(side=tk.LEFT, padx=10)
        return v

    def _create_file_list(self, parent):
        f = ttk.Frame(parent)
        lb = tk.Listbox(f, selectmode=tk.MULTIPLE, height=4)
        sb = ttk.Scrollbar(f, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        return {"frame": f, "list": lb}

    def _create_console(self, parent):
        t = tk.Text(parent, height=5, state='disabled', bg='#1e1e1e', fg='#d4d4d4', font=("Consolas", 9))
        t.pack(fill=tk.X, pady=5)
        return t

    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d: var.set(d)

    # --- State Management ---

    def _set_initial_state(self):
        self._toggle_core_list()
        self._toggle_journal_list()
        self._toggle_reformat_list()
        self._set_frame_enabled(self.journal_frame, False)
        self._set_frame_enabled(self.reformat_frame, False)

    def _set_frame_enabled(self, frame, enabled):
        state = "normal" if enabled else "disabled"
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                self._set_frame_enabled(child, enabled)

    def _toggle_core_list(self):
        if self.core_mode.get() == "select":
            self.core_list["frame"].pack(fill=tk.X, pady=5)
            self._fill_list(self.core_input, self.core_list["list"], "*.csv")
        else:
            self.core_list["frame"].pack_forget()

    def _toggle_journal_list(self):
        if self.journal_mode.get() == "select":
            self.journal_list["frame"].pack(fill=tk.X, pady=5)
            self._fill_list(self.journal_input, self.journal_list["list"], "*.xls*")
        else:
            self.journal_list["frame"].pack_forget()

    def _toggle_reformat_list(self):
        if self.reformat_mode.get() == "select":
            self.reformat_list["frame"].pack(fill=tk.X, pady=5)
            self._fill_list(self.reformat_input, self.reformat_list["list"], ("*.csv", "*.xls*"))
        else:
            self.reformat_list["frame"].pack_forget()

    def _fill_list(self, var, lb, patterns):
        lb.delete(0, tk.END)
        p = Path(var.get())
        if not p.is_dir(): return
        if isinstance(patterns, str): patterns = [patterns]
        files = []
        for pat in patterns: files.extend(p.glob(pat))
        for f in sorted(files): lb.insert(tk.END, f.name)

    # --- Execution Logic ---

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                # Append to ALL consoles for simplicity, or we could track active console
                for c in [self.core_console, self.journal_console, self.reformat_console]:
                    c.config(state='normal')
                    c.insert(tk.END, msg)
                    c.see(tk.END)
                    c.config(state='disabled')
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _preflight_check(self, path_var):
        p = Path(path_var.get())
        if not p.exists():
            messagebox.showerror("Error", f"Path does not exist: {p}")
            return False
        return True

    def _run_core(self):
        mode = self.core_mode.get()
        if mode == "skip":
            self._on_core_complete()
            return
        if not self._preflight_check(self.core_input): return
        
        files = None
        if mode == "select":
            sel = self.core_list["list"].curselection()
            if not sel: return messagebox.showwarning("Warning", "No files selected")
            files = [self.core_list["list"].get(i) for i in sel]

        self.core_btn.config(state='disabled')
        WorkerThread(
            target=build_core_yearly,
            kwargs={'input_folder': self.core_input.get(), 'output_dir': self.core_output.get(), 'file_list': files},
            on_complete=self._on_core_complete,
            on_error=self._on_error
        ).start()

    def _on_core_complete(self):
        self.after(0, lambda: [
            self._set_frame_enabled(self.core_frame, False),
            self._set_frame_enabled(self.journal_frame, True)
        ])

    def _run_journal(self):
        mode = self.journal_mode.get()
        if mode == "skip":
            self._on_journal_complete()
            return
        if not self._preflight_check(self.journal_input): return
        
        files = None
        if mode == "select":
            sel = self.journal_list["list"].curselection()
            if not sel: return messagebox.showwarning("Warning", "No files selected")
            files = [self.journal_list["list"].get(i) for i in sel]

        self.journal_btn.config(state='disabled')
        WorkerThread(
            target=build_yearly_outputs,
            kwargs={'input_folder': self.journal_input.get(), 'output_dir': self.journal_output.get(), 'file_list': files},
            on_complete=self._on_journal_complete,
            on_error=self._on_error
        ).start()

    def _on_journal_complete(self):
        self.after(0, lambda: [
            self._set_frame_enabled(self.journal_frame, False),
            self._set_frame_enabled(self.reformat_frame, True)
        ])

    def _run_reformat(self):
        mode = self.reformat_mode.get()
        if mode == "skip":
            messagebox.showinfo("Done", "Processing complete.")
            return
        if not self._preflight_check(self.reformat_input): return
        
        files = None
        if mode == "select":
            sel = self.reformat_list["list"].curselection()
            if not sel: return messagebox.showwarning("Warning", "No files selected")
            files = [self.reformat_list["list"].get(i) for i in sel]

        self.reformat_btn.config(state='disabled')
        WorkerThread(
            target=reformat_main,
            kwargs={
                'input_dir': self.reformat_input.get(),
                'output_dir': self.reformat_output.get(),
                'file_list': files,
                'journal_dir': self.journal_output.get(),
                'core_dir': self.core_output.get()
            },
            on_complete=lambda: messagebox.showinfo("Done", "Final reformatting complete."),
            on_error=self._on_error
        ).start()

    def _on_error(self, e):
        self.after(0, lambda: messagebox.showerror("Process Error", str(e)))

if __name__ == "__main__":
    App().mainloop()
