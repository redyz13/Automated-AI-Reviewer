import tkinter as tk
from tkinter import font as tkfont, ttk, filedialog, messagebox, PhotoImage
from PIL import Image, ImageTk
from model.model import ModelManager
import threading
import os
import sys

class AutomatedReviewerApp:
    def __init__(self, root):
        self.config = {
            "bg_color": "#1E1E1E",
            "title_bg_color": "#1E1E1E",
            "title_fg_color": "white",
            "button_bg_color": "#1E1E1E",
            "button_fg_color": "white",
            "button_active_bg_color": "#333333",
            "button_active_fg_color": "white",
            "button_hover_border_color": "white",
            "progress_color": "#1E1E1E",
            "status_fg_color": "white",
            "disclaimer_fg_color": "gray",
            "title_font_family": "Inter 28pt",
            "title_font_size": 14,
            "button_font_size": 18,
            "status_font_size": 12,
            "disclaimer_font_size": 10,
            "logo_font_size": 48,
            "title_bar_height": 30,
            "title_button_font_size": 12,
            "title_button_width": 2,
            "button_width": 240,
            "button_padding_x": 20,
            "button_padding_y": 20,
            "button_spacing_x": 10,
            "button_frame_padding_x": 33,
            "content_padding_y": 80,
            "icon_size": (28, 28),
            "logo_size": (192, 192),
        }

        self.file_path = ""
        self.model_loaded = False
        self.review_in_progress = False
        self.model_manager = None

        self.setup_root(root)
        self.create_widgets()

        self.set_buttons_state(tk.DISABLED)
        self.loading_label = tk.Label(self.main_window, text="Caricamento del modello in corso, attendere...", font=self.fonts["regular"], fg=self.config["status_fg_color"], bg=self.config["bg_color"])
        self.loading_label.pack(pady=(20, 10))

        self.model_thread = threading.Thread(target=self.initialize_model_manager, daemon=True)
        self.model_thread.start()

    def initialize_model_manager(self):
        self.model_manager = ModelManager(self.update_ui_on_model_loaded)

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def setup_root(self, root):
        root.withdraw()
        root.iconbitmap(self.resource_path('assets/icon.ico'))

        self.main_window = tk.Toplevel(root)
        self.main_window.overrideredirect(True)
        self.main_window.configure(bg=self.config["bg_color"])
        self.main_window.resizable(False, False)
        self.center_window(self.main_window, 1280, 720)

        self.title_bar = tk.Frame(self.main_window, bg=self.config["title_bg_color"], relief="raised", bd=0, highlightthickness=0)
        self.title_bar.pack(fill=tk.X, side=tk.TOP, ipadx=5, ipady=5)
        self.title_bar.bind("<Button-1>", self.start_move)
        self.title_bar.bind("<B1-Motion>", self.move_window)

        self.title_label = tk.Label(self.title_bar, text="Automated Reviewer 1.0", bg=self.config["title_bg_color"], fg=self.config["title_fg_color"], font=(self.config["title_font_family"], self.config["title_font_size"]))
        self.title_label.pack(side=tk.LEFT, padx=10)

        self.load_icons()
        self.create_title_buttons()

        self.main_window.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_icons(self):
        self.close_icon = self.load_image(self.resource_path("assets/close_icon.png"))
        self.minimize_icon = self.load_image(self.resource_path("assets/minimize_icon.png"))
        self.inserisci_icon = PhotoImage(file=self.resource_path("assets/paper_icon.png"))
        self.review_icon = PhotoImage(file=self.resource_path("assets/generate_icon.png"))
        self.inserisci_disabled_icon = PhotoImage(file=self.resource_path("assets/paper_disabled_icon.png"))
        self.review_disabled_icon = PhotoImage(file=self.resource_path("assets/generate_disabled_icon.png"))
        self.logo_image_raw = Image.open(self.resource_path("assets/logo.png"))

    def load_image(self, path):
        img = Image.open(path)
        img = img.resize(self.config["icon_size"], Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def create_title_buttons(self):
        tk.Button(self.title_bar, image=self.close_icon, command=self.close_window, bg=self.config["title_bg_color"], bd=0, padx=10, pady=0).pack(side=tk.RIGHT)
        tk.Button(self.title_bar, image=self.minimize_icon, command=self.minimize_window, bg=self.config["title_bg_color"], bd=0, padx=10, pady=0).pack(side=tk.RIGHT)

    def create_widgets(self):
        self.create_fonts()
        self.create_logo_label()
        self.create_button_frame()
        self.create_progress_bar()
        self.create_status_labels()
        self.create_disclaimer_label()

    def create_fonts(self):
        try:
            self.fonts = {
                "regular": tkfont.Font(family=self.config["title_font_family"], size=self.config["button_font_size"]),
                "small": tkfont.Font(family=self.config["title_font_family"], size=self.config["status_font_size"]),
                "tiny": tkfont.Font(family=self.config["title_font_family"], size=self.config["disclaimer_font_size"]),
                "bold": tkfont.Font(family=self.config["title_font_family"], size=self.config["logo_font_size"], weight="bold")
            }
        except tk.TclError:
            self.fonts = {
                "regular": tkfont.Font(family="Arial", size=self.config["button_font_size"]),
                "small": tkfont.Font(family="Arial", size=self.config["status_font_size"]),
                "tiny": tkfont.Font(family="Arial", size=self.config["disclaimer_font_size"]),
                "bold": tkfont.Font(family="Arial", size=self.config["logo_font_size"], weight="bold")
            }

    def create_logo_label(self):
        resized_logo = self.logo_image_raw.resize(self.config["logo_size"], Image.Resampling.LANCZOS)
        self.logo_image = ImageTk.PhotoImage(resized_logo)
        self.logo_label = tk.Label(self.main_window, image=self.logo_image, bg=self.config["bg_color"])
        self.logo_label.pack(pady=(self.config["content_padding_y"], self.config["button_padding_y"]))

    def create_button_frame(self):
        self.button_frame = tk.Frame(self.main_window, bg=self.config["bg_color"])
        self.button_frame.pack(pady=(0, self.config["button_padding_y"]), padx=(self.config["button_frame_padding_x"], 0))

        padding_between_icon_and_text = 4

        self.insert_button_frame = tk.Frame(self.button_frame, bg=self.config["bg_color"], highlightbackground=self.config["bg_color"], highlightthickness=2)
        self.insert_button_frame.pack(side=tk.LEFT, padx=(0, self.config["button_spacing_x"]))

        self.insert_button = tk.Button(self.insert_button_frame, text="Inserisci il paper", font=self.fonts["regular"], image=self.inserisci_icon, compound="left", anchor="w", bg=self.config["button_bg_color"], fg=self.config["button_fg_color"], activebackground=self.config["button_active_bg_color"], activeforeground=self.config["button_active_fg_color"], relief="flat", bd=0, width=self.config["button_width"], padx=padding_between_icon_and_text, command=self.on_insert_button_click)
        self.insert_button.pack(fill=tk.BOTH)
        self.insert_button_frame.bind("<Enter>", self.on_enter_insert_button)
        self.insert_button_frame.bind("<Leave>", self.on_leave_insert_button)

        self.review_button_frame = tk.Frame(self.button_frame, bg=self.config["bg_color"], highlightbackground=self.config["bg_color"], highlightthickness=2)
        self.review_button_frame.pack(side=tk.LEFT, padx=(self.config["button_spacing_x"], 0))

        self.review_button = tk.Button(self.review_button_frame, text="Genera review", font=self.fonts["regular"], image=self.review_icon, compound="left", anchor="w", bg=self.config["button_bg_color"], fg=self.config["button_fg_color"], activebackground=self.config["button_active_bg_color"], activeforeground=self.config["button_active_fg_color"], relief="flat", bd=0, width=self.config["button_width"], padx=padding_between_icon_and_text + 5, command=self.on_review_button_click)
        self.review_button.pack(fill=tk.BOTH)
        self.review_button_frame.bind("<Enter>", self.on_enter_review_button)
        self.review_button_frame.bind("<Leave>", self.on_leave_review_button)

    def on_enter_insert_button(self, event):
        if self.model_loaded and self.insert_button["state"] == tk.NORMAL:
            self.insert_button_frame.config(highlightbackground=self.config["button_hover_border_color"], highlightthickness=2)

    def on_leave_insert_button(self, event):
        if self.insert_button["state"] == tk.NORMAL:
            self.insert_button_frame.config(highlightbackground=self.config["bg_color"], highlightthickness=2)

    def on_enter_review_button(self, event):
        if self.model_loaded and self.review_button["state"] == tk.NORMAL:
            self.review_button_frame.config(highlightbackground=self.config["button_hover_border_color"], highlightthickness=2)

    def on_leave_review_button(self, event):
        if self.review_button["state"] == tk.NORMAL:
            self.review_button_frame.config(highlightbackground=self.config["bg_color"], highlightthickness=2)

    def create_progress_bar(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "white.Horizontal.TProgressbar",
            troughcolor=self.config["progress_color"],
            background="white",
            borderwidth=0,
            troughrelief="flat",
            relief="flat"
        )
        
        self.progress_frame = tk.Frame(self.main_window, bg=self.config["progress_color"])
        self.progress_frame.pack_forget()

        self.progress_text_label = tk.Label(self.progress_frame, text="", font=self.fonts["regular"], fg=self.config["title_fg_color"], bg=self.config["bg_color"])
        self.progress_text_label.pack(side=tk.TOP, pady=(0, 5))

        self.progress = ttk.Progressbar(self.progress_frame, length=400, mode='indeterminate', style="white.Horizontal.TProgressbar")
        self.progress.pack(side=tk.RIGHT, padx=10)

    def create_status_labels(self):
        self.status_label = tk.Label(self.main_window, text="", font=self.fonts["small"], fg=self.config["status_fg_color"], bg=self.config["bg_color"])
        self.status_label.pack_forget()

        self.file_selected_label = tk.Label(self.main_window, text="", font=self.fonts["small"], fg=self.config["status_fg_color"], bg=self.config["bg_color"])
        self.file_selected_label.pack(pady=(10, 5))

    def create_disclaimer_label(self):
        self.disclaimer_label = tk.Label(self.main_window, text="AutomatedReviewer può commettere errori", font=self.fonts["tiny"], fg=self.config["disclaimer_fg_color"], bg=self.config["bg_color"])
        self.disclaimer_label.pack(side=tk.BOTTOM, pady=10)

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        position_top = int(screen_height / 2 - height / 2)
        position_right = int(screen_width / 2 - width / 2)
        window.geometry(f'{width}x{height}+{position_right}+{position_top}')

    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def move_window(self, event):
        x = event.x_root - self.start_x
        y = event.y_root - self.start_y
        self.main_window.geometry(f'+{x}+{y}')

    def close_window(self):
        self.main_window.destroy()
        root.destroy()

    def minimize_window(self):
        self.main_window.iconify()

    def on_insert_button_click(self):
        if not self.model_loaded:
            messagebox.showwarning("Modello non caricato", "Attendere che il modello sia completamente caricato.")
            return
        
        if self.review_in_progress:
            messagebox.showwarning("In corso", "Una recensione è già in corso. Attendere che sia completata.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])

        if file_path:
            self.file_path = file_path
            file_name = os.path.basename(self.file_path)
            self.file_selected_label.config(text=f"Paper selezionato: {file_name}")
        else:
            self.file_path = ""
            self.file_selected_label.config(text="")

        self.progress_frame.pack_forget()
        self.status_label.pack_forget()

    def on_review_button_click(self):
        if not self.model_loaded:
            messagebox.showwarning("Modello non caricato", "Attendere che il modello sia completamente caricato.")
            return

        if self.review_in_progress:
            messagebox.showwarning("In corso", "Una recensione è già in corso. Attendere che sia completata.")
            return

        if not self.file_path:
            messagebox.showwarning("Nessun file selezionato", "Per favore, seleziona prima un file PDF.")
            return
        
        self.review_in_progress = True
        self.set_buttons_state(tk.DISABLED)
        self.progress_text_label.config(text="Generazione in corso...")
        self.progress_frame.pack(pady=(self.config["button_padding_y"], 10))
        self.progress.start()

        threading.Thread(target=self.generate_review, args=(self.file_path,), daemon=True).start()

    def set_buttons_state(self, state):
        if state == tk.DISABLED:
            self.insert_button.config(state=tk.NORMAL, image=self.inserisci_disabled_icon)
            self.review_button.config(state=tk.NORMAL, image=self.review_disabled_icon)
            self.insert_button.config(fg=self.config["disclaimer_fg_color"])
            self.review_button.config(fg=self.config["disclaimer_fg_color"])

            self.insert_button_frame.unbind("<Enter>")
            self.insert_button_frame.unbind("<Leave>")
            self.review_button_frame.unbind("<Enter>")
            self.review_button_frame.unbind("<Leave>")

            self.on_leave_insert_button(None)
            self.on_leave_review_button(None)
        else:
            self.insert_button.config(state=tk.NORMAL, image=self.inserisci_icon)
            self.review_button.config(state=tk.NORMAL, image=self.review_icon)
            self.insert_button.config(state=state, image=self.inserisci_icon, fg=self.config["button_fg_color"])
            self.review_button.config(state=state, image=self.review_icon, fg=self.config["button_fg_color"])
            self.insert_button_frame.bind("<Enter>", self.on_enter_insert_button)
            self.insert_button_frame.bind("<Leave>", self.on_leave_insert_button)
            self.review_button_frame.bind("<Enter>", self.on_enter_review_button)
            self.review_button_frame.bind("<Leave>", self.on_leave_review_button)

    def generate_review(self, file_path):
        self.progress['value'] = 0

        self.status_label.pack_forget()

        self.progress_text_label.config(text="Generazione in corso...")
        self.progress_frame.pack(pady=(self.config["button_padding_y"], 10))
        self.progress.start()

        paper_text = self.model_manager.extract_text_from_pdf(file_path)
        print("Generating review...")

        review = self.model_manager.model_inference(paper_text)

        directory = os.path.dirname(file_path)
        output_file_path = os.path.join(directory, 'generated_review.txt')
        self.model_manager.write_review_to_file(review, output_file_path)

        self.status_label.config(text=f"Review generata e scritta nel file: {output_file_path}")
        self.status_label.pack(pady=(10, 5))

        self.progress_frame.pack_forget()
        self.progress.stop()
        self.set_buttons_state(tk.NORMAL)
        self.review_in_progress = False

    def update_ui_on_model_loaded(self):
        self.model_loaded = True
        self.loading_label.destroy()
        self.set_buttons_state(tk.NORMAL)

    def on_close(self):
        if self.model_thread.is_alive():
            print("Interrompendo il caricamento del modello...")
        self.close_window()

if __name__ == "__main__":
    root = tk.Tk()
    app = AutomatedReviewerApp(root)
    root.mainloop()
