import sys
import os
import webbrowser
import pandas as pd
from types import SimpleNamespace
from num2words import num2words

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTableWidgetItem, QStackedWidget,
    QPushButton, QLineEdit, QLabel, QInputDialog, QMessageBox, QToolButton,
    QHBoxLayout, QVBoxLayout, QTextEdit, QComboBox, QTableWidget, QSizePolicy,
    QGroupBox, QSplitter, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, QSize, QTimer, QPropertyAnimation
from PySide6.QtGui import QAction, QMovie, QIcon

import matplotlib
matplotlib.use('Qt5Agg')  # ensure Qt5Agg backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pytrends.request import TrendReq
from trendspy import Trends

from AgentPipeline import RetrievalAgent, ContentAgent
from GenerateWorker import GenerateWorker
from TrendsWorker import TrendsWorker


class SystemGUI(QMainWindow):

    """
    Main application window that ties together:
      - A RetrievalAgent for fetching relevant legal text chunks
      - A ContentAgent for generating written output based on those chunks
      - Google Trends functionality via pytrends
    """

    def __init__(self):

        """
        Initialize the GUI, agents, and trending tools.
        
        """

        super().__init__()

        self.retrieval_agent = RetrievalAgent() # Instantiate the retrieval agent
        self.retrieval_agent.init() # Load any required indexes/models
        
        self.content_agent = ContentAgent()  # Instantiate the content agent 

        self.pytrends = TrendReq()   #Set up pytrends objects for fetching Google Trends data

        # Trends is used to fetch related queries ('top' and 'rising')
        self.trends = Trends()
        
        self.current_query = ""
        self.chunks = []
        self.last_block_pos = 0
        self.waiting_for_retry = False
        self.mode = ""

        # Build and arrange all UI components (buttons, text areas, etc.)
        self._setup_ui()


    def _setup_ui(self):
        self.setWindowTitle("System GUI")
        self.resize(700, 600)

        #buffers for toggling
        self.extraction_display = ""
        self.generation_display = ""

        # Central widget and layout
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 4)

        # Left panel: 40% width, colored background
        left_panel = QWidget(central)
        left_panel.setStyleSheet("background-color: #d7c9b6;")  # AliceBlue
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(20)

        #navigation buttons layout
        self.nav_widget = QWidget()
        nav_layout = QVBoxLayout(self.nav_widget)
        nav_layout.setContentsMargins(10,0,10,0)
        nav_layout.setSpacing(10)
        nav_layout.addStretch(1)

        #create the 3 navigation buttons
        for text, icon, slot in (
            ("Trends", "./trends_icon.svg", self.show_trends_panel),
            ("Extract", "./extract_icon.svg", self.show_query_panel),
            ("Generate", "./content_icon.svg", self.show_generate_panel),
        ):
            btn = self.make_nav_buttons(text, icon, slot)
            btn.setFixedWidth(140)
            btn.setFixedHeight(40)
            nav_layout.addWidget(btn, alignment=Qt.AlignHCenter)

        nav_layout.addStretch(1)
        left_layout.addWidget(self.nav_widget)

        # Right panel: 60% width, white background
        right_panel = QWidget(central)
        right_panel.setStyleSheet("background-color: white;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Create a container widget on the right panel to hold the welcome text and instructions
        self.front_container = QWidget(right_panel)
        fc_layout = QVBoxLayout(self.front_container)
        fc_layout.setContentsMargins(0,0,0,0)
        fc_layout.setSpacing(10)
        fc_layout.setAlignment(Qt.AlignVCenter)

        # Create a QLabel for displaying a welcome message
        self.welcome_label = QLabel("")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("""
            font-size: 18px;
            font-weight: 600;
            color #232c4d;
        """)
        
        fc_layout.addWidget(self.welcome_label)

        # Define some HTML-formatted instructions to show to the user
        html_instructions = """
        <div style="font-size: 14px; color: #232c4d; text-align: center;">
            <ul style="
                display: inline-block;
                text-align: left;
                padding-left: 20px;
                margin:0;
            ">
                <li> Click 'Google Trends' to explore recent legal trends</li>
                <li> Click 'Extract' to extract legal data from your knowledge datasets</li>
                <li> Click 'Generate' to create introductions, full articles or conclusions using the extracted info</li>
            </ul>
        </div>
        """
        # Create another QLabel that will render the HTML instructions
        self.instruction_label = QLabel(html_instructions)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)

        # Apply an opacity effect to the instruction_label so we can fade it in/out
        self.instr_opacity = QGraphicsOpacityEffect(self.instruction_label)
        self.instruction_label.setGraphicsEffect(self.instr_opacity)
        self.instr_opacity.setOpacity(0.0)
        
        self.instruction_label.hide()
        fc_layout.addWidget(self.instruction_label)

        right_layout.addWidget(self.front_container, alignment=Qt.AlignVCenter)
        
        # Text area for results (initially hidden)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 3px;"
        )
        self.results_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_text.hide()
        right_layout.addWidget(self.results_text, stretch=1)

        # Create a horizontal layout to hold the status text and loading spinner
        status_layout = QHBoxLayout()
        
        # QLabel to display the loading spinner animation
        self.spinner_label = QLabel()
        self.spinner_movie = QMovie("loading.gif")
        self.spinner_label.setMovie(self.spinner_movie)
        self.spinner_label.hide()

        # QLabel to display status messages
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #232c4d; font-style: italic;")
        self.status_label.hide()

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.spinner_label)
        status_layout.addStretch()
        right_layout.addLayout(status_layout)
        
        # Confirmation prompt
        self.confirm_widget = QWidget()
        confirm_layout = QHBoxLayout(self.confirm_widget)
        confirm_layout.setContentsMargins(0, 0, 0, 0)
        confirm_layout.setSpacing(10)
        self.confirm_label = QLabel("Is this the response you wanted?")
        self.yes_button = QPushButton("Yes")
        self.no_button = QPushButton("No")
        confirm_layout.addWidget(self.confirm_label)
        confirm_layout.addWidget(self.yes_button)
        confirm_layout.addWidget(self.no_button)
        self.confirm_widget.hide()
        right_layout.addWidget(self.confirm_widget)
        
        extract_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your query and press Enter...")
        self.query_input.setFixedHeight(30)
        self.query_input.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 3px;"
        )
        self.query_input.hide()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.hide()

        extract_layout.addWidget(self.query_input)
        extract_layout.addWidget(self.clear_btn)
        right_layout.addLayout(extract_layout)      

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Write Intro", "Full Article", "Write Pre-Law statement"])
        self.mode_combo.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 3px;"
            "color: gray;"
        )
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        self.mode_combo.hide()
        right_layout.addWidget(self.mode_combo)

        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("Article Topic")
        self.topic_input.setFixedHeight(30)
        self.topic_input.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 3px;"
        )
        self.topic_input.hide()
        right_layout.addWidget(self.topic_input)

        self.subsec_input = QLineEdit()
        self.subsec_input.setPlaceholderText("Max Number of Subsections")
        self.subsec_input.setFixedHeight(30)
        self.subsec_input.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 3px;"
        )
        self.subsec_input.hide()
        right_layout.addWidget(self.subsec_input)

        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack, stretch=1)
        
        self.stack.addWidget(self.build_trends_page())
        self.stack.hide()
      
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 4)
        
        self.setCentralWidget(central)

        self.show_welcome()

        #---- Signals

        self.query_input.returnPressed.connect(self.handle_query)
        self.clear_btn.clicked.connect(self.clear_results)
        self.yes_button.clicked.connect(self.confirm_yes)
        self.no_button.clicked.connect(self.confirm_no)
        
        self.topic_input.returnPressed.connect(lambda: self.subsec_input.setFocus() if self.mode_combo.currentText() == "Full Article" else self.handle_generate())
        self.subsec_input.returnPressed.connect(self.handle_generate)


    def show_welcome(self):

        self.welcome_text = "Welcome to your Legal Content Generator!"
        self.welcome_label.setText("")
        self.welcome_idx = 0
        self.welcome_label.show()
        
        self.instruction_label.hide()
        self.instr_opacity.setOpacity(0.0)
        
        self.topic_input.hide()
        self.subsec_input.hide()
        self.confirm_widget.hide()
        self.mode_combo.hide()       
        self.clear_btn.hide()
        self.results_text.hide()
        self.query_input.hide()
        self.stack.hide()

        self.welcome_timer = QTimer(self)
        self.welcome_timer.timeout.connect(self.update_welcome_txt)
        self.welcome_timer.start(100)

    def update_welcome_txt(self):

        if self.welcome_idx < len(self.welcome_text):
            current_txt = self.welcome_label.text()
            next_char = self.welcome_text[self.welcome_idx]
            self.welcome_label.setText(current_txt + next_char)
            self.welcome_idx += 1
        else:
            self.welcome_timer.stop()
            self.fade_in_instruction()

    def fade_in_instruction(self):
        self.instruction_label.show()

        self.instr_anim = QPropertyAnimation(self.instr_opacity, b"opacity", self)
        self.instr_anim.setDuration(500)
        self.instr_anim.setStartValue(0.0)
        self.instr_anim.setEndValue(1.0)
        self.instr_anim.start()
             

    def make_nav_buttons(self, text, icon_path, slot):

        btn = QToolButton(self)
        btn.setText(text)
        btn.setIcon(QIcon(icon_path))
        btn.setIconSize(QSize(24,24))
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setAutoRaise(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(slot)
           
        btn.setStyleSheet("""
            QToolButton{
                background: transparent;
                border: 1px solid #333333;
                border-radius: 8px;  
                padding: 4px, 8px;
                color: #333333;
                text-align: center;
            }
            QToolButton:hover{
                background: rgba(0,0,0, 0.05);
                
            }
            QToolButton:pressed{
                background: rgba(0,0,0, 0.10);
            
            }
        """)

        return btn


    def build_trends_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.trends_input = QLineEdit()
        self.trends_input.setPlaceholderText("Enter comma-separated keywords (e.g. Facebook, Google)")
        self.trends_input.setFixedHeight(30)
        self.trends_input.setStyleSheet("border: 1px solid gray; border-radius: 3px;")

        compare_btn = QPushButton("Compare")
        compare_btn.setFixedSize(100, 30)
        compare_btn.setStyleSheet(
            "background-color: rgba(35,44,77,170);"
            "color: white; border: none; border-radius: 5px;"
        )
        compare_btn.clicked.connect(self.compare_trends)
               
        input_grp = QGroupBox("Search Keywords")
        igl = QHBoxLayout(input_grp)
        igl.setContentsMargins(8,8,8,8)
        igl.addWidget(self.trends_input, stretch=1)
        igl.addWidget(compare_btn, alignment=Qt.AlignRight)
        layout.addWidget(input_grp, stretch=0)

        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(4)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        chart_box = QGroupBox("Trend Over Time")
        cbl = QVBoxLayout(chart_box)
        cbl.setContentsMargins(4,4,4,4)
        cbl.addWidget(self.canvas)
        splitter.addWidget(chart_box)

        self.related_table = QTableWidget()
        self.related_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table_box = QGroupBox("Related Queries")
        tbl = QVBoxLayout(table_box)
        tbl.setContentsMargins(4,4,4,4)
        self.related_table.setAlternatingRowColors(True)
        self.related_table.verticalHeader().setVisible(False)
        self.related_table.horizontalHeader().setStretchLastSection(True)
        tbl.addWidget(self.related_table)
        splitter.addWidget(table_box)

        splitter.setSizes([300, 150])
        layout.addWidget(splitter, stretch=1)

        return page

    def style_trends_plot(self, ax):

        ax.set_title("Interest by Region (UK, last 30 days)", pad=2, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Interest", labelpad=8)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(which="major", linestyle="--", alpha=0.4)
        leg = ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        for txt in leg.get_texts():
            txt.set_fontsize(10)

        self.canvas.figure.tight_layout(rect=[0,0,0.85,1])

                  
    def start_spinner(self, text):
        self.status_label.setText(text)
        self.status_label.show()
        self.spinner_label.show()
        self.spinner_movie.start()
        QApplication.processEvents()

    def stop_spinner(self):
        self.spinner_movie.stop()
        self.spinner_label.hide()
        self.status_label.hide()
        QApplication.processEvents()

    def show_trends_panel(self):
        self.front_container.hide()
        self.instr_opacity.setOpacity(0.0)
        
        self.topic_input.hide()
        self.subsec_input.hide()
        self.confirm_widget.hide()
        self.mode_combo.hide()       
        self.clear_btn.hide()
        self.results_text.hide()
        self.query_input.hide()

        self.stack.show()
        self.stack.setCurrentIndex(0)
        
    def show_query_panel(self):
        """Reveal the query input and results list for legal info extraction."""

        self.front_container.hide()
        self.instr_opacity.setOpacity(0.0)
        self.stack.hide()

        self.topic_input.hide()
        self.subsec_input.hide()
        self.confirm_widget.hide()
        self.mode_combo.hide()

        self.results_text.show()
        self.results_text.setPlainText(self.extraction_display)

        self.clear_btn.show()
        self.query_input.show()

        self.query_input.setFocus()

    def show_generate_panel(self):
        
        self.front_container.hide()
        self.instr_opacity.setOpacity(0.0)
        self.stack.hide()

        self.query_input.hide()
        self.clear_btn.hide()
        self.confirm_widget.hide()

        self.results_text.show()
        self.results_text.setPlainText(self.generation_display)

        self.topic_input.show()
        self.topic_input.setFocus()
    
        self.mode_combo.show()
        

    def clear_results(self):

        self.extraction_display = ""
        self.generation_display = ""
        self.retrieval_agent.confirmed_results.clear()

        self.results_text.clear()
        

    def on_mode_change(self, mode):
        
        if mode == "Full Article":
            self.subsec_input.show()
        else:
            self.subsec_input.hide()
            self.subsec_input.clear()


    def compare_trends(self):
        text = self.trends_input.text().strip()
        kws = [kw.strip() for kw in text.split(',') if kw.strip()]
        if not kws:
            QMessageBox.warning(self, "Invalid input", "Please enter at least one keyword.")
            return

        self.start_spinner("Fetching Google Trends data...")

        self.worker = TrendsWorker(kws, self.pytrends, self.trends)
        self.worker.finished.connect(self.trends_done)
        self.worker.error.connect(self.trends_error)
    
        self.worker.start()
         b  

    def trends_done(self, top10, allq):

        self.stop_spinner()

        self.figure.set_size_inches(6, 4)  
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        top10.plot(kind="bar", ax=ax)
        
        self.style_trends_plot(ax)

        self.canvas.draw()
        self.canvas.show()

        tbl = self.related_table
        tbl.show()

        if allq.empty:
            display_df = pd.DataFrame({"Notice": ["No related queries available."]})
        else:
            # drop any "value" columns
            cols = [c for c in allq.columns if "value" not in c]
            display_df = allq[cols]

        tbl.clear()
        tbl.setRowCount(len(display_df))
        tbl.setColumnCount(len(display_df.columns))
        tbl.setHorizontalHeaderLabels(display_df.columns.tolist())

        for i, row in display_df.iterrows():
            for j, col in enumerate(display_df.columns):
                tbl.setItem(i, j, QTableWidgetItem(str(row[col])))

        tbl.resizeColumnsToContents()
        
        

    def trends_error(self, msg):
        self.stop_spinner()

        self.canvas.draw()
        self.canvas.show()

        tbl = self.related_table
        tbl.show()
        tbl.clear()
        tbl.setRowCount(1)
        tbl.setColumnCount(1)
        tbl.setHorizontalHeaderLabels(["Related Queries"])
        tbl.setItem(0,0, QTableWidgetItem("Could not fetch related queries."))
        tbl.resizeColumnsToContents()
        
        QMessageBox.critical(self, "Error fetching trends", msg)

                                     
    def handle_query(self):
        """Handle the user query: run hybrid retrieval and display results."""
        query = self.query_input.text().strip()
        if not query:
            return

        self.current_query = query

        self.chunks = self.retrieval_agent.retrieve(query)
        self.current_chunk_idx = 0

        full=self.extraction_display

        raw  = self.chunks[self.current_chunk_idx]
        body = raw.split("\n", 1)[1].strip() if "\n" in raw else raw

        if self.waiting_for_retry:
            prefix=full[:self.last_block_pos]
            sep = "\n" + "-"*100 + "\n\n" if self.last_block_pos > 0 else ""
            block=f"{sep}Query: {self.query_input.text().strip()}\n\nExtracted Legal Info:\n{body}"
            
            self.extraction_display = prefix + block

            self.waiting_for_retry = False

        else:
            self.last_block_pos=len(full)
            sep = "\n" + "-"*100 + "\n\n" if len(full) > 0 else ""
            block=f"{sep}Query: {query}\n\nExtracted Legal Info:\n{body}"
            self.extraction_display += block

        self.results_text.setPlainText(self.extraction_display)
        self.last_block_pos = len(self.extraction_display) - len(block)
        self.confirm_widget.show()

    
    def confirm_yes(self):
        
        approved_chunk = self.chunks[self.current_chunk_idx]

        self.retrieval_agent.confirm(self.current_query, approved_chunk)
        
        self.confirm_widget.hide()
        self.query_input.clear()
        self.query_input.setFocus()
        

    def confirm_no(self):
        self.current_chunk_idx += 1
            
        if self.current_chunk_idx < len(self.chunks):
            raw  = self.chunks[self.current_chunk_idx]
            body = raw.split("\n", 1)[1].strip() if "\n" in raw else raw
            
            prefix_raw  = self.extraction_display[: self.last_block_pos]

            if prefix_raw:
                prefix = prefix_raw.rstrip("\n") + "\n"
                sep ="-" * 100 + "\n\n"

            else:
                prefix = ""
                sep    = ""
                

            block=f"{sep}Query: {self.query_input.text().strip()}\n\nExtracted Legal Info:\n{body}"
            
            self.extraction_display = prefix + block
            self.results_text.setPlainText(self.extraction_display)
            
            # update block start for further replaces
            self.last_block_pos = len(prefix)
            self.confirm_widget.show()
            
        else:
            self.results_text.setPlainText(self.extraction_display)
            self.results_text.append(
                '<p style="margin:0; color:red;">'
                'No more results, please try a different query'
                '</p>'
            )
                            
            self.confirm_widget.hide()
            self.waiting_for_retry = True


    def handle_generate(self):
        self.mode = self.mode_combo.currentText()

        if self.mode == "Full Article":
            subsec_int = int(self.subsec_input.text().strip())
            subsec_word = num2words(subsec_int, to='cardinal', lang='en')

        else:
            subsec_word = ''

        topic = self.topic_input.text().strip()

        if not topic:
            QMessageBox.warning(self, "Invalid input", "Please fill in the inputs")
            return

        approved = self.retrieval_agent.get_confirmed()
        snippets = "\n".join(item["chunk"] for item in approved)

        self.start_spinner("Generating Response")
        self.mode_combo.setEnabled(False)
        self.topic_input.setEnabled(False)
        self.subsec_input.setEnabled(False)

        self.worker = GenerateWorker(self.content_agent, snippets, self.mode, topic, subsec_word)
        self.worker.progress.connect(lambda a, m: self.start_spinner(f"Incomplete answer. Retrying ({a}/{m})"))
        self.worker.finished.connect(self.generate_finished)
        self.worker.start()
        

    def generate_finished(self, output):
        
        self.stop_spinner()

        title = ""

        if self.mode == "Write Intro":
            title = 'Introduction'

        elif self.mode == "Write Pre-Law statement":
            title = 'Statement'

        else:
            title = 'Article'
            

        block = (
            f"------------------- Final {title} ---------------------- \n\n"
            f"{output}"
            f"\n\n"
        )
        
        self.generation_display += block

        if self.topic_input.isVisible():
            self.results_text.setPlainText(self.generation_display)
        else:
            pass

        self.mode_combo.setEnabled(True)
        self.topic_input.setEnabled(True)
        self.subsec_input.setEnabled(True)
        self.topic_input.clear()
        self.subsec_input.clear()
        self.topic_input.setFocus()

    
def main():
    app = QApplication(sys.argv)
    window = SystemGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
