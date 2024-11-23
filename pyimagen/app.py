"""
Main application module for PyImagen
""" 

import asyncio
import base64
import io
import json
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import fal_client as fal
import numpy as np
import requests
from groq import Groq
from PIL import Image, ImageQt
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import mimetypes
from io import BytesIO

MOCK_API = False
MOCK_PROMPT = "A serene landscape with rolling hills, a mystical forest, and a rainbow in the distance"

AVAILABLE_MODELS = {
    "FLUX 1 Schnell": "fal-ai/flux/schnell",
    "Recraft V3": "fal-ai/recraft-v3",
    "AuraFlow": "fal-ai/aura-flow",
    "FLUX 1.1 Pro Ultra": "fal-ai/flux-pro/v1.1-ultra",
    "FLUX 1.1 Pro": "fal-ai/flux-pro/v1.1",
    "Stable Cascade": "fal-ai/stable-cascade",
    "Fast Turbo Diffusion": "fal-ai/fast-turbo-diffusion",
    "Fast LCM Diffusion": "fal-ai/fast-lcm-diffusion",
    "Fast Lightning SDXL": "fal-ai/fast-lightning-sdxl",
    "Fooocus": "fal-ai/fooocus",
    "Realistic Vision": "fal-ai/realistic-vision",
    "Lightning Models": "fal-ai/lightning-models",
    "Kolors": "fal-ai/kolors",
}

SECRETS_FILE = Path("data/secrets.json")


def load_models():
    """Load models from models.json and merge with built-in models"""
    models = AVAILABLE_MODELS.copy()
    models_file = Path("data/models.json")
    
    if models_file.exists():
        try:
            with open(models_file, "r") as f:
                custom_models = json.load(f)
                if isinstance(custom_models, dict):
                    # Merge custom models with built-in models
                    models.update(custom_models)
        except Exception as e:
            print(f"Warning: Error loading models.json: {e}")
    
    return models


class GroqWorker(QThread):
    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)

    def __init__(self, groq_key):
        super().__init__()
        self.groq_key = groq_key

    def run(self):
        try:
            start_time = datetime.now()

            if MOCK_API:
                prompt = MOCK_PROMPT
            else:
                client = Groq(api_key=self.groq_key)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": "Generate a creative and detailed image prompt for an AI image generator. Make it visually interesting and specific. Do not include any quotation marks in your response. Only respond with the prompt text itself, no additional text.",
                        }
                    ],
                    model="llama-3.1-8b-instant",
                )
                prompt = chat_completion.choices[0].message.content.strip("\"'")

            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.finished.emit(prompt, elapsed_time)
        except Exception as e:
            self.error.emit(str(e))


class PromptEnhancerWorker(QThread):
    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)

    def __init__(self, base_prompt, enhancement_instructions):
        super().__init__()
        self.base_prompt = base_prompt
        self.enhancement_instructions = enhancement_instructions

    def run(self):
        try:
            start_time = datetime.now()

            if MOCK_API:
                enhanced_prompt = f"{self.base_prompt} with magical ethereal lighting and intricate details"
            else:
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Original prompt: {self.base_prompt}\n\nInstructions: {self.enhancement_instructions}\n\nProvide an enhanced version of the original prompt based on these instructions. Only respond with the enhanced prompt text itself, no additional text or quotation marks.",
                        }
                    ],
                    model="llama-3.1-8b-instant",
                )
                enhanced_prompt = chat_completion.choices[0].message.content.strip(
                    "\"'"
                )

            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.finished.emit(enhanced_prompt, elapsed_time)
        except Exception as e:
            self.error.emit(str(e))


class ImageGeneratorWorker(QThread):
    finished = pyqtSignal(Image.Image, str, float)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, prompt, model_id, fal_key):
        super().__init__()
        self.prompt = prompt
        self.model_id = model_id
        self.fal_key = fal_key

    def run(self):
        try:
            start_time = datetime.now()

            if MOCK_API:
                # Generate random noise image instantly
                width, height = 800, 600
                random_image = np.random.randint(
                    0, 255, (height, width, 3), dtype=np.uint8
                )
                img = Image.fromarray(random_image)

            else:
                # Set FAL_KEY for the client
                os.environ["FAL_KEY"] = self.fal_key
                result = fal.subscribe(
                    self.model_id,
                    arguments={
                        "prompt": self.prompt,
                        "num_images": 1,
                        "image_size": {"width": 1280, "height": 960},
                        "enable_safety_checker": False,
                        "sync_mode": True,
                    },
                    on_queue_update=self.handle_queue_update,
                )

                if result and "images" in result and result["images"]:
                    image_data = result["images"][0]
                    img = self.process_image_response(image_data)
                else:
                    raise Exception("No image was generated")

            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.finished.emit(img, self.prompt, elapsed_time)

        except Exception as e:
            self.error.emit(str(e))

    def process_image_response(self, image_data):
        """Process image response which could be base64 or URL-based"""
        if "url" not in image_data:
            raise Exception("No image URL in response")

        url = image_data["url"]
        
        # Check if it's a base64 data URL
        if url.startswith('data:'):
            # Extract base64 image data from data URL
            img_data = url.split(",")[1]
            img_bytes = base64.b64decode(img_data)
        else:
            # Download image from URL
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to download image: {response.status_code}")
            img_bytes = response.content

        # Get content type
        if "content_type" in image_data:
            content_type = image_data["content_type"]
        else:
            # Try to guess from URL or default to PNG
            content_type = mimetypes.guess_type(url)[0] or 'image/png'

        # Create PIL Image from bytes
        img = Image.open(BytesIO(img_bytes))

        # Convert to RGB if necessary (handles RGBA, WebP, etc.)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Convert to RGBA first to properly handle transparency
            img = img.convert('RGBA')
            # Create white background
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            # Composite image onto background
            img = Image.alpha_composite(background, img)
        
        # Convert to RGB for final format
        img = img.convert('RGB')
        
        return img

    def handle_queue_update(self, update):
        if isinstance(update, fal.InProgress) and update.logs:
            for log in update.logs:
                if isinstance(log, dict) and "message" in log:
                    self.progress.emit(log["message"])


class EnhancePromptDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enhance Prompt")
        self.setModal(True)
        self.setMinimumWidth(500)

        # Apply styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QTextEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 5px 15px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add explanation
        explanation = QLabel(
            "Describe how you want to enhance the prompt. For example:\n"
            "- Add more artistic style details\n"
            "- Make it more cinematic\n"
            "- Add specific lighting and mood\n"
            "- Include more detailed textures"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Enhancement instruction input
        self.enhance_input = QTextEdit()
        self.enhance_input.setPlaceholderText("Enter enhancement instructions...")
        self.enhance_input.setText(
            "Enhance this prompt by adding more creative details, artistic style, "
            "and visual elements. Make it more vivid and specific, but maintain "
            "the original concept."
        )
        self.enhance_input.setMinimumHeight(100)
        layout.addWidget(self.enhance_input)

        # Buttons
        button_layout = QHBoxLayout()
        enhance_btn = QPushButton("Enhance")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(enhance_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        enhance_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

    def get_enhancement_instructions(self):
        return self.enhance_input.toPlainText()


class ImageGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Generator")
        self.setMinimumSize(1200, 800)

        # Create data directory if it doesn't exist
        self.data_dir = Path("data")
        self.images_dir = self.data_dir / "images"
        self.metadata_file = self.data_dir / "metadata.json"

        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

        # Load models from file and merge with built-in models
        self.available_models = load_models()

        # Check for required API keys
        self.check_api_keys()

        # Initialize metadata
        self.generated_images = []

        # Initialize Groq client with key from environment or secrets
        groq_key = self.get_api_key("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key)

        self.is_generating = False

        self.current_model = "SDXL Turbo"  # Default model

        # Initialize UI first
        self.init_ui()

        # Load metadata after UI is initialized
        self.load_metadata()

        # Initialize workers
        self.groq_worker = None
        self.image_worker = None

        self.prompt_generation_time = 0.0

        # Debug shortcuts
        self.show_debug = False
        self.installEventFilter(self)

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Left panel (History)
        history_frame = QFrame()
        history_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        history_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 4px;
            }
        """)
        history_frame.setMinimumWidth(250)  # Minimum width for history panel
        left_layout = QVBoxLayout(history_frame)
        left_layout.setContentsMargins(4, 4, 4, 4)

        history_header = QLabel("History")
        history_header.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #ffffff; padding: 2px;"
        )
        left_layout.addWidget(history_header)

        # Scroll area for history
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 8px;
                background: #2b2b2b;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical {
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                height: 0px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_layout.setSpacing(10)
        self.history_layout.addStretch()
        self.scroll_area.setWidget(self.history_widget)
        left_layout.addWidget(self.scroll_area)

        # Right panel
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        right_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 4px;
            }
        """)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # Model selector section
        model_frame = QFrame()
        model_frame.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border-radius: 4px;
                padding: 2px;
            }
            QLabel {
                color: #ffffff;
            }
            QComboBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 2px;
            }
        """)
        model_layout = QHBoxLayout(model_frame)
        model_layout.setContentsMargins(4, 2, 4, 2)

        model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.available_models.keys())
        self.model_selector.setCurrentText(self.current_model)
        self.model_selector.setMinimumWidth(200)
        self.model_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # Add timing stats label here
        self.timing_label = QLabel()
        self.timing_label.setStyleSheet("color: #808080; font-size: 11px;")

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        model_layout.addWidget(self.timing_label)
        right_layout.addWidget(model_frame)

        # Create a container for the image and make it expand
        image_container = QWidget()
        image_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)

        # Image display section
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        image_layout = QVBoxLayout(image_frame)

        self.current_image_label = QLabel()
        self.current_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        image_layout.addWidget(self.current_image_label)

        # Add image frame to container
        image_container_layout.addWidget(image_frame)

        # Add the expanding container to right layout
        right_layout.addWidget(image_container)

        # Controls section
        controls_frame = QFrame()
        controls_frame.setFixedHeight(120)  # Set fixed height for controls frame
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border-radius: 4px;
                padding: 4px;
            }
            QTextEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 4px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 2px 4px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(4)
        controls_layout.setContentsMargins(4, 4, 4, 4)

        # Progress bar with absolute positioning
        self.progress_bar = QProgressBar(controls_frame)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(2)
        self.progress_bar.hide()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #404040;
                border: none;
                border-radius: 1px;
            }
            QProgressBar::chunk {
                background-color: #606060;
            }
        """)
        # Position the progress bar at the top of controls_frame
        self.progress_bar.move(4, 4)

        # Connect to controls_frame's resize event to keep progress bar properly sized
        controls_frame.resizeEvent = lambda e: self.progress_bar.setFixedWidth(
            controls_frame.width() - 8
        )

        # Prompt entry and buttons
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)  # Add consistent spacing
        input_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align everything to top

        prompt_container = QWidget()  # Create container for prompt
        prompt_layout = QVBoxLayout(prompt_container)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(0)
        prompt_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align to top

        self.prompt_entry = QTextEdit()
        self.prompt_entry.setFixedHeight(80)  # Fixed height for text edit
        self.prompt_entry.setPlaceholderText("Enter your prompt here...")
        self.prompt_entry.installEventFilter(self)
        prompt_layout.addWidget(self.prompt_entry)

        # Create a vertical layout for buttons with no spacing at top
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align buttons to top

        # Create button grid
        button_grid = QHBoxLayout()
        button_grid.setSpacing(4)
        button_grid_left = QVBoxLayout()
        button_grid_left.setSpacing(4)
        button_grid_right = QVBoxLayout()
        button_grid_right.setSpacing(4)

        # Create buttons
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        random_btn = QPushButton("Random")
        random_btn.clicked.connect(self.generate_random)
        generate_btn = QPushButton("Generate")
        generate_btn.clicked.connect(self.generate_image)
        enhance_btn = QPushButton("Enhance Prompt")
        enhance_btn.clicked.connect(self.enhance_prompt)

        # Add buttons to the grid with no spacing
        button_grid_left.addWidget(copy_btn)
        button_grid_left.addWidget(generate_btn)
        button_grid_left.addStretch()  # Push buttons to top

        button_grid_right.addWidget(random_btn)
        button_grid_right.addWidget(enhance_btn)
        button_grid_right.addStretch()  # Push buttons to top

        button_grid.addLayout(button_grid_left)
        button_grid.addLayout(button_grid_right)

        button_layout.addLayout(button_grid)

        input_layout.addWidget(prompt_container, stretch=1)
        input_layout.addLayout(button_layout)

        controls_layout.addWidget(input_frame)

        # Add controls frame directly to right layout (will stick to bottom)
        right_layout.addWidget(controls_frame)

        # Add panels to main layout
        main_layout.addWidget(history_frame, stretch=1)
        main_layout.addWidget(right_frame, stretch=3)

    def do_generate_image(self, prompt):
        # This runs in main thread
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.generate_image_async(prompt))

        response = requests.get(result["images"][0]["url"])
        img_data = response.content
        img = Image.open(io.BytesIO(img_data))

        self.handle_generated_image(img, prompt)

    def generate_random(self):
        if self.is_generating:
            return

        self.is_generating = True
        self.progress_bar.show()

        groq_key = self.get_api_key("GROQ_API_KEY")
        self.groq_worker = GroqWorker(groq_key)
        self.groq_worker.finished.connect(self.on_prompt_generated)
        self.groq_worker.error.connect(self.handle_generation_error)
        self.groq_worker.start()

    def on_prompt_generated(self, prompt, elapsed_time):
        self.prompt_generation_time = elapsed_time
        self.prompt_entry.setText(prompt)
        self.update_timing_stats()
        self.generate_image()

    def generate_image(self):
        if not self.is_generating:
            self.is_generating = True
            self.progress_bar.show()

        prompt = self.prompt_entry.toPlainText()
        if not prompt:
            return

        # Update current model from dropdown
        self.current_model = self.model_selector.currentText()

        fal_key = self.get_api_key("FAL_KEY")
        self.image_worker = ImageGeneratorWorker(
            prompt, self.available_models[self.current_model], fal_key
        )
        self.image_worker.finished.connect(self.handle_generated_image)
        self.image_worker.error.connect(self.handle_generation_error)
        self.image_worker.progress.connect(self.handle_progress)
        self.image_worker.start()

    def handle_progress(self, message):
        # Optionally show progress messages in the UI
        print(f"Progress: {message}")

    def handle_generation_error(self, error_message):
        self.is_generating = False
        self.progress_bar.hide()

        # Create and show error dialog
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Generation Error")
        error_dialog.setText("An error occurred during generation:")
        error_dialog.setInformativeText(error_message)
        error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        error_dialog.setStyleSheet("""
            QMessageBox {
                background-color: #2b2b2b;
            }
            QMessageBox QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 5px 15px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        error_dialog.exec()

    def handle_generated_image(self, img, prompt, elapsed_time):
        self.is_generating = False
        self.progress_bar.hide()

        timestamp = datetime.now()

        # Generate unique filename based on timestamp
        filename = f"image_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"

        # Save image to disk
        image_path = self.images_dir / filename
        img.save(image_path, "PNG")

        self.generated_images.append(
            {
                "image": img,
                "prompt": prompt,
                "timestamp": timestamp,
                "filename": filename,
            }
        )

        # Save metadata
        self.save_metadata()

        self.update_display(img, len(self.generated_images) - 1)
        self.add_to_history(img, len(self.generated_images) - 1)

        # Update timing stats
        self.update_timing_stats(image_time=elapsed_time)

    def update_display(self, img, index):
        # First ensure the label has a reasonable minimum size
        self.current_image_label.setMinimumSize(800, 600)

        # Wait for the layout to update
        QTimer.singleShot(0, lambda: self._update_display_delayed(img))

    def _update_display_delayed(self, img):
        # Get the available size for the image
        available_width = self.current_image_label.width()
        available_height = self.current_image_label.height()

        # Create pixmap and scale it to fit the available space
        pixmap = self.get_scaled_pixmap(img, QSize(available_width, available_height))
        self.current_image_label.setPixmap(pixmap)

    def add_to_history(self, img, index, initial_load=False):
        # Remove any existing stretch at the end
        if not initial_load:
            for i in reversed(range(self.history_layout.count())):
                item = self.history_layout.itemAt(i)
                if item.spacerItem():
                    self.history_layout.removeItem(item)
                    break

        # Add the new thumbnail
        thumb_frame = QFrame()
        thumb_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        thumb_layout = QVBoxLayout(thumb_frame)
        thumb_layout.setContentsMargins(4, 2, 4, 2)

        scroll_width = self.scroll_area.viewport().width() - 20
        thumb_size = QSize(scroll_width, scroll_width)
        pixmap = self.get_scaled_pixmap(img, thumb_size)

        thumb_label = QLabel()
        thumb_label.setPixmap(pixmap)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.mousePressEvent = lambda e, idx=index: self.show_historical_image(
            idx
        )

        # Store the index in the frame for later reference
        thumb_frame.setProperty("image_index", index)

        # Add prompt text with ellipsis
        prompt_label = QLabel()
        prompt_label.setMaximumWidth(scroll_width)
        prompt_label.setStyleSheet("color: gray; font-size: 10px;")

        prompt_text = self.generated_images[index]["prompt"]
        metrics = prompt_label.fontMetrics()
        elidedText = metrics.elidedText(
            prompt_text, Qt.TextElideMode.ElideRight, scroll_width - 10
        )
        prompt_label.setText(elidedText)

        prompt_label.setTextFormat(Qt.TextFormat.PlainText)
        prompt_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        prompt_label.setToolTip(prompt_text)

        prompt_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        prompt_label.setMinimumHeight(15)
        prompt_label.setMaximumHeight(15)

        thumb_layout.addWidget(thumb_label)
        thumb_layout.addWidget(prompt_label)

        self.history_layout.addWidget(thumb_frame)

        if not initial_load:
            self.history_layout.addStretch()
            QTimer.singleShot(0, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        # Ensure layout is updated
        self.history_layout.update()
        self.history_widget.updateGeometry()

        # Get the scrollbar and scroll to bottom
        scrollbar = self.scroll_area.verticalScrollBar()

        # Schedule another update to ensure final position
        QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

    def get_scaled_pixmap(self, img, target_size):
        qim = ImageQt.ImageQt(img)
        pixmap = QPixmap.fromImage(qim)

        # Calculate scaling to fit within target size while maintaining aspect ratio
        src_width = pixmap.width()
        src_height = pixmap.height()
        target_width = target_size.width()
        target_height = target_size.height()

        # Calculate scale factors for both dimensions
        width_ratio = target_width / src_width
        height_ratio = target_height / src_height

        # Use the smaller ratio to ensure image fits completely
        scale_factor = min(width_ratio, height_ratio)

        new_width = int(src_width * scale_factor)
        new_height = int(src_height * scale_factor)

        return pixmap.scaled(
            new_width,
            new_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def show_historical_image(self, index):
        img_data = self.generated_images[index]
        self.update_display(img_data["image"], index)
        self.prompt_entry.setText(img_data["prompt"])

    def keyPressEvent(self, event):
        # Check if the pressed key is 'R' and the prompt entry doesn't have focus
        if event.key() == Qt.Key.Key_R and not self.prompt_entry.hasFocus():
            self.generate_random()
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if obj == self.prompt_entry and event.type() == event.Type.KeyPress:
            # Check for Command + Enter (Mac) or Control + Enter (other platforms)
            if (
                event.key() == Qt.Key.Key_Return
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            ):
                self.generate_image()
                return True
        # Add debug shortcut (Ctrl+Shift+D)
        elif (
            event.type() == event.Type.KeyPress
            and event.key() == Qt.Key.Key_D
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.show_debug = not self.show_debug
            self.toggle_debug_mode()
            return True
        return super().eventFilter(obj, event)

    def toggle_debug_mode(self):
        """Toggle layout debugging visualization"""
        if self.show_debug:
            # Show layout boundaries
            self.setStyleSheet("""
                QWidget {
                    border: 1px solid red;
                }
                QFrame {
                    border: 1px solid blue;
                }
                QLabel {
                    border: 1px solid green;
                }
            """)
        else:
            # Restore original styling
            self.setStyleSheet("")

    def update_timing_stats(self, image_time=None):
        prompt_time = f"Prompt generation: {self.prompt_generation_time:.1f}s"
        image_time_text = (
            f" | Image generation: {image_time:.1f}s" if image_time is not None else ""
        )
        self.timing_label.setText(f"{prompt_time}{image_time_text}")

    def load_metadata(self):
        """Load metadata from disk and reconstruct generated_images list"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

            self.generated_images = []
            for item in metadata:
                image_path = self.images_dir / item["filename"]
                if image_path.exists():
                    img = Image.open(image_path)
                    self.generated_images.append(
                        {
                            "image": img,
                            "prompt": item["prompt"],
                            "timestamp": datetime.fromisoformat(item["timestamp"]),
                            "filename": item["filename"],
                        }
                    )

            # Populate history with loaded images
            for idx, img_data in enumerate(self.generated_images):
                self.add_to_history(img_data["image"], idx, initial_load=True)

            # Show the last image if any exist
            if self.generated_images:
                last_idx = len(self.generated_images) - 1
                self.show_historical_image(last_idx)
                # Schedule scrolling to bottom after UI is fully loaded
                QTimer.singleShot(100, self.scroll_to_bottom)

    def save_metadata(self):
        """Save metadata to disk"""
        metadata = []
        for item in self.generated_images:
            metadata.append(
                {
                    "prompt": item["prompt"],
                    "timestamp": item["timestamp"].isoformat(),
                    "filename": item["filename"],
                }
            )

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def copy_to_clipboard(self):
        # Get the current pixmap from the image label
        pixmap = self.current_image_label.pixmap()
        if pixmap:
            # Get the system clipboard
            clipboard = QApplication.clipboard()
            # Copy the pixmap to clipboard
            clipboard.setPixmap(pixmap)

    def enhance_prompt(self):
        if self.is_generating:
            return

        current_prompt = self.prompt_entry.toPlainText()
        if not current_prompt:
            return

        # Show dialog to get enhancement instructions
        dialog = EnhancePromptDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            enhancement_instructions = dialog.get_enhancement_instructions()

            self.is_generating = True
            self.progress_bar.show()

            self.prompt_enhancer = PromptEnhancerWorker(
                current_prompt, enhancement_instructions
            )
            self.prompt_enhancer.finished.connect(self.on_prompt_enhanced)
            self.prompt_enhancer.error.connect(self.handle_generation_error)
            self.prompt_enhancer.start()

    def on_prompt_enhanced(self, enhanced_prompt, elapsed_time):
        self.prompt_generation_time = elapsed_time
        self.prompt_entry.setText(enhanced_prompt)
        self.update_timing_stats()
        self.is_generating = False
        self.progress_bar.hide()

    def mouseMoveEvent(self, event):
        if self.show_debug:
            widget = self.childAt(event.pos())
            if widget:
                tooltip = (
                    f"Class: {widget.__class__.__name__}\n"
                    f"Size: {widget.size().width()}x{widget.size().height()}\n"
                    f"Pos: {widget.pos().x()},{widget.pos().y()}\n"
                    f"Min Size: {widget.minimumSize().width()}x{widget.minimumSize().height()}\n"
                    f"Max Size: {widget.maximumSize().width()}x{widget.maximumSize().height()}"
                )
                widget.setToolTip(tooltip)
            super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Schedule thumbnail updates after resize
        QTimer.singleShot(0, self.update_all_thumbnails)

    def update_all_thumbnails(self):
        scroll_width = self.scroll_area.viewport().width() - 20

        # Update each thumbnail in the history layout
        for i in range(self.history_layout.count()):
            item = self.history_layout.itemAt(i)
            if item and not item.spacerItem():
                thumb_frame = item.widget()
                if thumb_frame:
                    # Get the index from the frame property
                    idx = thumb_frame.property("image_index")
                    if idx is not None:
                        # Find the thumbnail label and prompt label
                        thumb_layout = thumb_frame.layout()
                        if thumb_layout:
                            thumb_label = thumb_layout.itemAt(0).widget()
                            prompt_label = thumb_layout.itemAt(1).widget()

                            if thumb_label and isinstance(thumb_label, QLabel):
                                # Update thumbnail size
                                thumb_size = QSize(scroll_width, scroll_width)
                                pixmap = self.get_scaled_pixmap(
                                    self.generated_images[idx]["image"], thumb_size
                                )
                                thumb_label.setPixmap(pixmap)

                                # Update prompt label
                                if prompt_label and isinstance(prompt_label, QLabel):
                                    prompt_label.setMaximumWidth(scroll_width)
                                    prompt_text = self.generated_images[idx]["prompt"]
                                    metrics = prompt_label.fontMetrics()
                                    elidedText = metrics.elidedText(
                                        prompt_text,
                                        Qt.TextElideMode.ElideRight,
                                        scroll_width - 10,
                                    )
                                    prompt_label.setText(elidedText)

    def check_api_keys(self):
        """Check if required API keys are available, prompt if missing"""
        required_keys = {"GROQ_API_KEY": "Groq API Key", "FAL_KEY": "Fal.ai API Key"}

        # First try environment variables
        missing_keys = {
            key: label
            for key, label in required_keys.items()
            if not os.environ.get(key)
        }

        # Then check secrets file
        secrets = self.load_secrets()
        missing_keys = {
            key: label for key, label in missing_keys.items() if key not in secrets
        }

        if missing_keys:
            self.show_api_key_dialog(missing_keys)

    def load_secrets(self):
        """Load secrets from file"""
        if SECRETS_FILE.exists():
            try:
                with open(SECRETS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_secrets(self, secrets):
        """Save secrets to file"""
        SECRETS_FILE.parent.mkdir(exist_ok=True)
        with open(SECRETS_FILE, "w") as f:
            json.dump(secrets, f, indent=2)

    def get_api_key(self, key_name):
        """Get API key from environment or secrets file"""
        # First try environment
        key = os.environ.get(key_name)
        if key:
            return key

        # Then try secrets file
        secrets = self.load_secrets()
        return secrets.get(key_name)

    def show_api_key_dialog(self, missing_keys):
        """Show dialog to input missing API keys"""
        dialog = QDialog(self)
        dialog.setWindowTitle("API Keys Required")
        dialog.setModal(True)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 5px 15px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add explanation text
        explanation = QLabel(
            "Please enter your API keys to use the application. "
            "These will be stored securely in your data folder."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Create input fields for each missing key
        input_fields = {}
        for key, label in missing_keys.items():
            key_layout = QHBoxLayout()
            key_label = QLabel(f"{label}:")
            key_input = QLineEdit()
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_layout.addWidget(key_label)
            key_layout.addWidget(key_input)
            layout.addLayout(key_layout)
            input_fields[key] = key_input

        # Add buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        # Connect buttons
        save_btn.clicked.connect(partial(self.save_api_keys, dialog, input_fields))
        cancel_btn.clicked.connect(lambda: self.handle_api_dialog_cancel(dialog))

        dialog.exec()

    def save_api_keys(self, dialog, input_fields):
        """Save entered API keys to secrets file"""
        secrets = self.load_secrets()

        # Update secrets with new values
        for key, input_field in input_fields.items():
            value = input_field.text().strip()
            if value:
                secrets[key] = value

        self.save_secrets(secrets)
        dialog.accept()

    def handle_api_dialog_cancel(self, dialog):
        """Handle dialog cancel - exit app if required keys are missing"""
        dialog.reject()
        if not self.get_api_key("GROQ_API_KEY") or not self.get_api_key("FAL_KEY"):
            self.close()


def main():
    app = QApplication(sys.argv)
    window = ImageGeneratorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
