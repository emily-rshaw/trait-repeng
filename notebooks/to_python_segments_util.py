import nbformat

# Load the notebook
with open('main_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Extract only code cells
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']

# Write to a file
with open('extracted_code.py', 'w', encoding='utf-8') as f:
    for cell in code_cells:
        f.write('# Cell\n')
        f.write(cell.source)
        f.write('\n\n')