import os
import sys

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from herbalist_assistant.graph.advanced_graph import app

def main():
    try:
        # Get the Mermaid graph
        graph = app.get_graph()
        
        # Draw the graph as a PNG
        png_data = graph.draw_mermaid_png()
        
        output_path = "langgraph_diagram.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"Successfully generated graph diagram at {output_path}")
    except Exception as e:
        print(f"Error generating graph diagram: {e}")

if __name__ == "__main__":
    main()

#You can regenerate the diagram at any time by running the following script from the root of your workspace:
#python scripts/visualize_graph.py
