

import subprocess


def render_quarto_report(department):
    quarto_doc = "03_06e/render-vs-preview.qmd"
    output_file = f"report_{department}.html"
    command = [
        "quarto",
        "render",
        quarto_doc,
        "--output", output_file,
        "--params", f"report_department:{department}"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Report generated successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error rendering Quarto document: {e}")
    except FileNotFoundError:
        print("Error: Quarto command not found. Make sure Quarto is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


departments = ["IT", "Sales", "Customer Service"]
for department in departments:
    render_quarto_report(department)
