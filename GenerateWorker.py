from PySide6.QtCore import QThread, Signal

class GenerateWorker(QThread):

        """
    QThread subclass that runs the ContentAgent’s generation process in a separate thread.
    Emits:
      - finished(str): when generation is complete, passes the generated text.
      - progress(int, int): to indicate a retry attempt (current_attempt, max_attempts).
    """

    finished = Signal(str) # Signal emitted when generation finishes
    progress = Signal(int, int) # Signal emitted to report progress

    def __init__(self, content_agent, snippets, mode, topic, max_subsec):

        """
        Initialize the worker thread.
        Args:
            content_agent: Instance of ContentAgent that will produce text.
            snippets (str): The retrieved text snippets to feed into the ContentAgent.
            mode (str): Generation mode (e.g., "Write Intro", "Write Pre-Law statement").
            topic (str): Topic keyword for the generation.
            max_subsec (int): Maximum number of subsections to include.
        """
        super().__init__()
        self.content_agent = content_agent
        self.snippets = snippets
        self.mode = mode
        self.topic = topic
        self.max_subsec = max_subsec



    def run(self):

        """
        Entry point for the thread when it starts.
        
        """

        # Configure the ContentAgent (sets up LlamaAutoGenClient internally)
        self.content_agent.set_generation_params(self.mode, self.topic, self.max_subsec)
        
        max_attempts = 3
        output = ""

        # Retry loop: try up to max_attempts times to get an output ending in a period
        for attempt in range(1, max_attempts+1):

            # Send the retrieved snippets as a "system" message to the ContentAgent
            self.content_agent.on_system_message(type("M",(),{"content":self.snippets}))

            # Trigger the ContentAgent’s on_user_message to generate text.
            resp = self.content_agent.on_user_message(type("M",(),{"content":None}))

            # Extract the generated text
            output = resp["content"]

            if output.strip().endswith("."):
                break
            else:
                self.progress.emit(attempt+1, max_attempts)


        self.finished.emit(output)

        
        
        
