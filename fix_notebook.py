import json

# Load the notebook
with open('notebook/antoine.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix problematic cells
fixed_count = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Fix 1: Congestion cell that uses 'capacity' instead of 'link_capacity'
        if "link_performance['vc_ratio'] = link_performance['volume'] / link_performance['capacity']" in source:
            print(f"Found problematic Congestion cell at index {i}")
            new_source = source.replace(
                "if 'vc_ratio' not in link_performance.columns:\n            link_performance['vc_ratio'] = link_performance['volume'] / link_performance['capacity']",
                """# Calculate V/C ratio - use available capacity column
            capacity_col = 'link_capacity' if 'link_capacity' in link_performance.columns else 'capacity'
            if capacity_col in link_performance.columns:
                link_performance['vc_ratio'] = link_performance['volume'] / link_performance[capacity_col]"""
            )
            cell['source'] = new_source.split('\n')
            # Ensure proper formatting
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            fixed_count += 1
            print(f"Fixed Congestion Analysis cell - replaced capacity reference")

# Save the notebook
with open('notebook/antoine.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook fixed and saved - {fixed_count} cells corrected")
