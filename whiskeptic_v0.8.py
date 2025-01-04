import tkinter as tk
from tkinter import ttk, scrolledtext, font, filedialog
import threading
import time
import queue
import numpy as np
import sounddevice as sd
from deep_translator import GoogleTranslator
import tempfile
import wave
import os
from openai import OpenAI
from groq import Groq
import logging
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

class SpeechTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Skeptic Translator")
        self.root.geometry("1240x600")
        
        self.openai_client = OpenAI(api_key="yourkey")
        self.groq_client = Groq(api_key="yourkey")
        
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        self.system_prompts = {
            'detect': """
Analyze the transcript and follow these steps precisely:

1. Identify two types of English words:
   a) English words written in Latin script within Persian text
   b) English words transcribed in Persian script (like "کامپیوتر" for "computer", "دیتکشن" for "detection")

2. For Persian-transcribed English words:
   - Convert them to their standard English spelling
   - Only include widely recognized English words
   - Include common technical, medical terms and computing terminology
   Examples:
   "دیتابیس" -> "database"
   "اینترفیس" -> "interface"
   "پروسس" -> "process"

3. Output rules:
   - Return one word per line
   - Use standard English spelling only
   - Include compound technical terms as single entries (e.g., "database", "smartphone")
   - Do not include:
     * Persian words
     * Hybrid Persian-English terms
     * Partial word matches
     * Non-English terms
     * Articles (a, an, the) in isolation

4. If no valid English words are found, return an empty string.

Example input:
"من از دیتابیس استفاده کردم و یک new فایل ساختم. اینترفیس خوبی داشت."

Example output:
database
new
interface
            """,
            'translate': """
            Analyze the transcript and:
            1. Find English words that appear in Persian text (including Persian-transcribed English words)
            2. For each found word, return the English word and its Persian translation in this format:
               word|translation
            3. Return each word-translation pair on a new line
            4. Return an empty string if no English words are found
            Example output:
            computer|کامپیوتر
            office|دفتر
            """
        }
        

        self.settings = {
            'debug_enabled': tk.BooleanVar(value=False),
            'font_size': tk.IntVar(value=18),
            'font_name': tk.StringVar(value="Arial"),
            'chunk_duration': tk.DoubleVar(value=2.0),
            'display_duration': tk.DoubleVar(value=3.0),
            'sample_rate': 16000,
            'channels': 1,
            'buffer_size': 1024,
            'save_transcripts': tk.BooleanVar(value=False),
            'save_location': tk.StringVar(value=os.path.expanduser("~")),
            'whisper_provider': tk.StringVar(value="OpenAI"),
            'whisper_model': tk.StringVar(value="whisper-1"),
            'post_process_model': tk.StringVar(value="gpt-4o")
        }
        
        # Store complete session transcript
        self.current_session_transcript = []
        self.session_start_time = None
        
        self.pending_translations = set()
        self.translation_cache = {}
        self.is_listening = False
        self.message_queue = queue.Queue()
        self.messages = []
        self.audio_queue = queue.Queue()
        
        self.setup_gui()
        self.update_messages()

    

    def transcribe_audio(self, audio_file):
        provider = self.settings['whisper_provider'].get()
        model = self.settings['whisper_model'].get()
        
        try:
            if provider == "OpenAI":
                transcription = self.openai_client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language="fa",
                    response_format="text"
                )
                return transcription
            elif provider == "Groq":
                # Groq returns the transcription text directly
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(audio_file.name, audio_file.read()),
                    model=model,
                    language="fa",
                    response_format="text",
                    temperature=0.0
                )
                # Return the transcription directly since it's already text
                return transcription
                
        except Exception as e:
            self.log_debug(f"Transcription error with {provider}: {e}")
            return None

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.log_debug(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    def save_audio_chunk(self, audio_data):
        """Save audio chunk to a temporary WAV file"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.settings['channels'])
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.settings['sample_rate'])
                wf.writeframes(audio_data.tobytes())
            return temp_file.name


    def process_transcript(self, transcript):
        """Process transcript using OpenAI regardless of transcription provider"""
        try:
            model = self.settings['post_process_model'].get()
            if self.translator_var.get() == "GPT":
                prompt = self.system_prompts['translate']
            else:
                prompt = self.system_prompts['detect']
            
            # Always use OpenAI for post-processing
            completion = self.openai_client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            result = completion.choices[0].message.content.strip()
            
            if self.translator_var.get() == "GPT":
                translations = []
                for line in result.split('\n'):
                    if '|' in line:
                        word, translation = line.split('|')
                        translations.append((word.strip(), translation.strip()))
                return translations
            else:
                return [word.strip() for word in result.split('\n') if word.strip()]
                
        except Exception as e:
            self.log_debug(f"Processing error: {e}")
            return []

    def toggle_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.toggle_button.config(text="Stop")
            self.status_label.config(text="Listening...")
            self.session_start_time = datetime.now()
            self.current_session_transcript = []
            
            try:
                self.stream = sd.InputStream(
                    channels=self.settings['channels'],
                    samplerate=self.settings['sample_rate'],
                    callback=self.audio_callback
                )
                self.stream.start()
                threading.Thread(target=self.process_audio, daemon=True).start()
            except Exception as e:
                self.log_debug(f"Error starting audio: {e}")
                self.is_listening = False
                self.toggle_button.config(text="Start")
                self.status_label.config(text="Error")
        else:
            self.is_listening = False
            self.toggle_button.config(text="Start")
            self.status_label.config(text="Ready")
            
            # Save complete session transcript
            if self.settings['save_transcripts'].get() and self.current_session_transcript:
                self.save_complete_session()
                
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.log_debug(f"Error stopping audio: {e}")

    def save_complete_session(self):
        if not self.current_session_transcript:
            return
            
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}_complete.txt"
        filepath = os.path.join(self.settings['save_location'].get(), filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.current_session_transcript))
            self.log_debug(f"Saved complete session transcript to {filepath}")
        except Exception as e:
            self.log_debug(f"Error saving session transcript: {e}")

    def setup_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Main tab
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Translator")
        
        # Settings tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")
        
        self.setup_main_tab(main_tab)
        self.setup_settings_tab(settings_tab)
        


    def setup_main_tab(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=5)
        
        self.toggle_button = ttk.Button(control_frame, text="Start", command=self.toggle_listening)
        self.toggle_button.pack(side="left", padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side="left", padx=5)
        
        # Whisper Provider selection
        ttk.Label(control_frame, text="Provider:").pack(side="left", padx=5)
        provider_combo = ttk.Combobox(
            control_frame,
            textvariable=self.settings['whisper_provider'],
            values=["OpenAI", "Groq"],
            state="readonly",
            width=10
        )
        provider_combo.pack(side="left", padx=5)
        provider_combo.bind('<<ComboboxSelected>>', self.update_model_options)
        
        # Whisper Model selection
        ttk.Label(control_frame, text="Model:").pack(side="left", padx=5)
        self.model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.settings['whisper_model'],
            state="readonly",
            width=15
        )
        self.model_combo.pack(side="left", padx=5)
        
        # Post-processing model selection
        ttk.Label(control_frame, text="Post-process:").pack(side="left", padx=5)
        post_process_combo = ttk.Combobox(
            control_frame,
            textvariable=self.settings['post_process_model'],
            values=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-3.5-turbo-instruct"
            ],
            state="readonly",
            width=20
        )
        post_process_combo.pack(side="left", padx=5)
        
        ttk.Label(control_frame, text="Translator:").pack(side="left", padx=5)
        self.translator_var = tk.StringVar(value="Google")
        translator_combo = ttk.Combobox(
            control_frame,
            textvariable=self.translator_var,
            values=["Google", "GPT"],
            state="readonly",
            width=10
        )
        translator_combo.pack(side="left", padx=5)
        
        # Main display area
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill="both", expand=True, pady=5)
        
        self.text_display = tk.Text(
            display_frame,
            wrap=tk.WORD,
            height=10
        )
        self.text_display.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_text_font()
        
        # Debug area (optional)
        self.debug_frame = ttk.LabelFrame(parent, text="Debug Log")
        self.debug_display = scrolledtext.ScrolledText(
            self.debug_frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            height=10
        )
        self.debug_display.pack(fill="both", expand=True, padx=5, pady=5)
        self.toggle_debug_display()
    def update_model_options(self, event=None):
        provider = self.settings['whisper_provider'].get()
        if provider == "OpenAI":
            models = ["whisper-1"]
        else:  # Groq
            models = ["whisper-large-v3-turbo", "whisper-large-v3"]
        
        self.model_combo['values'] = models
        self.settings['whisper_model'].set(models[0])

    def setup_settings_tab(self, parent):
        settings_frame = ttk.LabelFrame(parent, text="Settings")
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Debug toggle
        ttk.Checkbutton(
            settings_frame,
            text="Enable Debug Log",
            variable=self.settings['debug_enabled'],
            command=self.toggle_debug_display
        ).pack(anchor="w", pady=5)
        
        # Font settings
        font_frame = ttk.LabelFrame(settings_frame, text="Font Settings")
        font_frame.pack(fill="x", pady=5)
        
        ttk.Label(font_frame, text="Font:").pack(side="left", padx=5)
        font_combo = ttk.Combobox(
            font_frame,
            textvariable=self.settings['font_name'],
            values=list(font.families()),
            state="readonly"
        )
        font_combo.pack(side="left", padx=5)
        
        ttk.Label(font_frame, text="Size:").pack(side="left", padx=5)
        size_spin = ttk.Spinbox(
            font_frame,
            from_=8,
            to=72,
            textvariable=self.settings['font_size'],
            width=5
        )
        size_spin.pack(side="left", padx=5)
        
        # Audio settings
        audio_frame = ttk.LabelFrame(settings_frame, text="Audio Settings")
        audio_frame.pack(fill="x", pady=5)
        
        ttk.Label(audio_frame, text="Chunk Duration (s):").pack(side="left", padx=5)
        ttk.Spinbox(
            audio_frame,
            from_=1,
            to=10,
            increment=0.5,
            textvariable=self.settings['chunk_duration'],
            width=5
        ).pack(side="left", padx=5)
        
        # Display settings
        display_frame = ttk.LabelFrame(settings_frame, text="Display Settings")
        display_frame.pack(fill="x", pady=5)
        
        ttk.Label(display_frame, text="Word Display Duration (s):").pack(side="left", padx=5)
        ttk.Spinbox(
            display_frame,
            from_=1,
            to=10,
            increment=0.5,
            textvariable=self.settings['display_duration'],
            width=5
        ).pack(side="left", padx=5)
        
        # Apply button
        ttk.Button(
            settings_frame,
            text="Apply Settings",
            command=self.apply_settings
        ).pack(pady=10)
        save_frame = ttk.LabelFrame(settings_frame, text="Save Settings")
        save_frame.pack(fill="x", pady=5)
        
        ttk.Checkbutton(
            save_frame,
            text="Save Transcripts",
            variable=self.settings['save_transcripts']
        ).pack(side="left", padx=5)
        
        ttk.Label(save_frame, text="Save Location:").pack(side="left", padx=5)
        ttk.Entry(
            save_frame,
            textvariable=self.settings['save_location'],
            width=30
        ).pack(side="left", padx=5)
        
        ttk.Button(
            save_frame,
            text="Browse",
            command=self.choose_save_location
        ).pack(side="left", padx=5)
        
    
    def choose_save_location(self):
        directory = filedialog.askdirectory()
        if directory:
            self.settings['save_location'].set(directory)

    def save_transcript(self, content, suffix):
        if not self.settings['save_transcripts'].get():
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}_{suffix}.txt"
        filepath = os.path.join(self.settings['save_location'].get(), filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.log_debug(f"Saved {suffix} transcript to {filepath}")
        except Exception as e:
            self.log_debug(f"Error saving transcript: {e}")

    def apply_settings(self):
        self.update_text_font()
        self.toggle_debug_display()
        
    def update_text_font(self):
        self.text_display.configure(
            font=(
                self.settings['font_name'].get(),
                self.settings['font_size'].get()
            )
        )
        
    def toggle_debug_display(self):
        if self.settings['debug_enabled'].get():
            self.debug_frame.pack(fill="both", expand=True, pady=5)
        else:
            self.debug_frame.pack_forget()

    def log_debug(self, message):
        if self.settings['debug_enabled'].get():
            self.debug_display.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
            self.debug_display.see(tk.END)
            logging.debug(message)



    def process_audio(self):
        audio_buffer = []
        chunk_samples = int(self.settings['sample_rate'] * self.settings['chunk_duration'].get())
        
        def process_chunk(audio_data):
            temp_filename = self.save_audio_chunk(audio_data)
            try:
                # Open the temporary file in binary read mode
                with open(temp_filename, "rb") as audio_file:
                    transcription = self.transcribe_audio(audio_file)
                    
                    if transcription:
                        self.current_session_transcript.append(transcription)
                        
                        if self.translator_var.get() == "GPT":
                            translations = self.process_transcript(transcription)
                            for word, translation in translations:
                                self.message_queue.put(f"{word}: {translation}")
                        else:
                            english_words = self.process_transcript(transcription)
                            for word in english_words:
                                if word in self.translation_cache:
                                    translation = self.translation_cache[word]
                                else:
                                    translation = self.translate_word(word)
                                    if translation:
                                        self.translation_cache[word] = translation
                                if translation:
                                    self.message_queue.put(f"{word}: {translation}")
                
            finally:
                os.unlink(temp_filename)

        while self.is_listening:
            try:
                while len(audio_buffer) < chunk_samples and self.is_listening:
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(chunk)
                
                if not self.is_listening:
                    break
                
                current_chunk = np.array(audio_buffer[:chunk_samples])
                audio_buffer = audio_buffer[chunk_samples:]
                
                current_chunk = (current_chunk * np.iinfo(np.int16).max).astype(np.int16)
                self.thread_pool.submit(process_chunk, current_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log_debug(f"Error: {e}")
            
    def translate_word(self, word):
        """Translate identified English words to Persian"""
        try:
            if self.translator_var.get() == "Google":
                translator = GoogleTranslator(source='en', target='fa')
                translation = translator.translate(word)
                self.log_debug(f"Translated '{word}' to '{translation}'")
                return translation
            else:
                self.log_debug(f"Translation service {self.translator_var.get()} not implemented")
                return None
        except Exception as e:
            self.log_debug(f"Translation error: {e}")
            return None
                
    def update_messages(self):
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.messages.append((message, time.time()))
            
        current_time = time.time()
        self.messages = [(msg, t) for msg, t in self.messages 
                        if current_time - t <= self.settings['display_duration'].get()]
        
        self.text_display.delete(1.0, tk.END)
        for message, _ in self.messages:
            self.text_display.insert(tk.END, message + "\n")
            
        self.root.after(100, self.update_messages)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()