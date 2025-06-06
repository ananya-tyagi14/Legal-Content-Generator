from docx import Document
from docx.oxml.ns import qn
import re

class ConvertPlainTxt:

    def __init__(self):

        # regex pattern to detect top-level section headings
        self.section_heading_pattern = re.compile(r'^\d+\.\s')

        # regex pattern to detect subsection headings
        self.subsection_heading_pattern = re.compile(r'^\d+\.\d+(?:\.\d+)?\.?\s')

    def get_list_level(self, paragraph):

        """
        Determine the numbering level (indentation) of a list paragraph.
        Returns:
            int (level) if this is a numbered/bullet list item,
            None if not a list item,
            or 0 as a fallback if <w:ilvl> exists but has no valid val attribute.
        """

        p = paragraph._element  # Access the XML element for this paragraph
        pPr = p.find(qn("w:pPr")) # Look for paragraph properties
        if pPr is None:
            return None
        numPr = pPr.find(qn("w:numPr")) # Within properties, look for numbering properties
        if numPr is None:
            return None

        # Look for the indentation level element
        ilvl = numPr.find(qn("w:ilvl"))
        if ilvl is not None and ilvl.get(qn("w:val")).isdigit():
            return int(ilvl.get(qn("w:val")))
        return 0  # fallback if <w:ilvl> has no val


    def is_list_paragraph(self, paragraph):
        """Return True if the paragraph is a bullet/numbered list item (at any level)."""
        return self.get_list_level(paragraph) is not None


    def collapse_blank_lines(self, lines):
        
        """
        collapse multiple consecutive blank lines into a single blank line.
        Args:
            lines (list of str): The lines to process.
        Returns:
            list of str: Cleaned lines with no more than one consecutive blank line.
        """
        
        cleaned_lines = []
        for line in lines:
            if not line.strip():
                if cleaned_lines and not cleaned_lines[-1].strip():
                        continue
                else:
                    cleaned_lines.append("")
            else:
                cleaned_lines.append(line)
        return cleaned_lines


    def docx_to_text(self, docx_path):
        """
        Convert a .docx file to plain text, preserving:
        Args:
            docx_path (str): Path to the .docx file to convert.
        Returns:
            str: The resulting plain text.
        """
        document = Document(docx_path)
        output = []
        first_section = True   #Flag to detect first section so we don't add a blank line
        

        for para in document.paragraphs:
            # Skip any paragraph explicitly styled as "Heading 1"
            if para.style and para.style.name == "Heading 1":
                continue
            
            text = para.text.strip()
            if not text:
                continue

             # If text matches a top-level section heading 
            if self.section_heading_pattern.match(text):
                if not first_section:
                    output.append('')
                    
                first_section = False
                output.append(text)
                continue
            
            # If text matches a subsection heading 
            if self.subsection_heading_pattern.match(text):
                output.append(text)
                continue

            # If the paragraph is a list item
            if self.is_list_paragraph(para):
                level = self.get_list_level(para)
                
                # Wrap the list text in an XML-like tag showing its nesting level
                output.append(f"<ITEM level=\"{level}\">{text}</ITEM>")
                continue

            output.append(text)
      
        # Collapse any excessive blank lines in the collected output
        final_output = self.collapse_blank_lines(output)
        full_text = "\n".join(final_output)
        return full_text

                





