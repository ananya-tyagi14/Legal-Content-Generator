import re
import json
import os

class ConvertToJson:

    def __init__(self, filepath):

        """
        Initialize with the path to the .doxc text files.
        """

        self.filepath = filepath

        #regex patterns for headings
        self.section_pattern = re.compile(r'^(\d+\. )(.+)$')           
        self.subsection_pattern = re.compile(r'^(\d+(?:\.\d+)+\.?\s)(.+)$')


    def parse_heading_line(self, line, pattern, split=True):

        """
        Given a line and a regex pattern, extract the heading and any trailing content after a colon.
        Args:
            line (str): The raw line of text to analyze.
            pattern (re.Pattern): The compiled regex to match against the line.
            split (bool): If True, attempt to split off content after a colon.
        Returns:
            tuple:
                heading (str): The heading part (including numbering).
                extra_content (str): Any text after a colon, or empty if none/split=False.
        """

        match = pattern.match(line)
        if not match:
            return line, ""

        if not split:
            return line, ""

        prefix = match.group(1)  # e.g. "1. " or "2.3. "
        rest = match.group(2)  # the heading text that follows the numbering

        # Look for a colon in the rest of the heading text
        colon_index = rest.find(":")
        if colon_index != -1:
            # If colon found, split at the colon:
            heading = prefix + rest[:colon_index].strip()
            extra_content = rest[colon_index+1:].strip()
            return heading, extra_content

        else:
            return line, ""


    def parse_document(self, text):

        """
        Parse the entire text of the converted document into a JSON-serializable structure.

        Args:
            text (str): The full plain text. 
        Returns:
            list: A list of dicts, each representing a section or subsection group with content.
        """

        lines = text.splitlines()
        data = []

        #Derive a "topic" from the filename: take basename, remove extension, split on "-",
        fname = os.path.basename(self.filepath)
        topic = os.path.splitext(fname)[0].split('-')[-1].strip()

        current_section = None
        current_subsection = None
        current_blocks = []
        list_stack = None

        def flush_group():
            """
            When we hit a new section or subsection, finalize the previous group (if any):
            - Combine all text and nested lists into a single "Content" list.
            - Append a dict with Section, Subsection, Content, and topic to `data`.
            """
            nonlocal current_section, current_subsection, current_blocks
            
            if not current_blocks:
                return

            # Helper to combine nested blocks into a single string
            def collect_content(blocks):
                texts = []
                for blk in blocks:
                    texts.append(blk.get('text', ''))
                    def dfs(items):
                        for itm in items:
                            texts.append(itm.get('text', ''))
                            if itm.get('list'):
                                dfs(itm['list'])
                    dfs(blk.get('list', []))
                    
                return " ".join(texts)
            
            # Build the group dict
            group = {
                "Section": current_section,
                "Subsection": current_subsection,
                "Content": current_blocks,
                "topic": topic,
                    }
            data.append(group)

            # Reset blocks and list_stack for the next group
            current_blocks = []
            list_stack = None

        item_re = re.compile(r'<ITEM level="(?P<lvl>\d+)">(?P<txt>.*)</ITEM>')  # Regex to detect list items

        for idx, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue

            # Check for a top-level section heading
            sec_m = self.section_pattern.match(line)
            if sec_m:
                flush_group()

                # Extract the section heading text (drop the numbering)
                current_section, extra = self.parse_heading_line(line, self.section_pattern, split=False)
                match = re.match(r'^(\d+\. )(.+)$', current_section)
                if match:
                    current_section = match.group(2)  # e.g. from "1. Introduction" take "Introduction"
                current_subsection = None
                continue

            # Check for a subsection heading
            sub_m = self.subsection_pattern.match(line)
            if sub_m:
                flush_group()
                
                current_subsection, extra = self.parse_heading_line(line, self.subsection_pattern, split=True)
                match = re.match(r'^(\d+(?:\.\d+)+\.?\s)(.+)$', current_subsection)
                if match:
                    current_subsection = match.group(2)

                # If there was extra text after a colon, treat it as an initial content block
                if extra:
                    current_blocks.append({"text": extra})
                continue

            # Check if the line is a list item
            m = item_re.match(line)
            if m:
                try: 
                    lvl = int(m.group('lvl'))  # indentation level
                    txt = m.group('txt').strip()
                    bullet_txt = f"• {txt}"   # prepend a bullet character for readability

                    if list_stack is None:
                        # First time encountering a list item in the current block:
                        if not current_blocks:
                            current_blocks.append({"text":""})

                        # Initialize the first level of list nesting
                        current_blocks[-1].setdefault("list", [])
                        list_stack = [current_blocks[-1]["list"] ]

                    # If current stack depth is deeper than this level, pop up
                    while len(list_stack) - 1 > lvl:
                        list_stack.pop()

                    # If lvl deeper than current stack depth: create a new nested list level
                    if lvl > len(list_stack) - 1:
                        parent = list_stack[-1][-1]
                        parent.setdefault("list", [])
                        list_stack.append(parent["list"])

                    list_stack[-1].append({"text": bullet_txt})

                except IndexError:
                    print(f"\nIndexError on line {idx!r}: {raw!r}")
                    print(f"  Current list_stack: {list_stack}")
                    print(f"  Parsed lvl={lvl}, txt={txt!r}")
                    raise
                continue

            # If we reach here, it's not a list item: reset list_stack context
            if list_stack is not None:
                list_stack = None

            current_blocks.append({"text": line})

        flush_group()
        return data


    def parse_and_save(self, text, output_file):

        """
        Parse the given text into JSON structure and save it to a file.
        Args:
            text (str): The full converted text to parse.
            output_file (str): Path where the resulting JSON should be saved.
        """

        data = self.parse_document(text)

        with open(output_file, "w", encoding ='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

                 
