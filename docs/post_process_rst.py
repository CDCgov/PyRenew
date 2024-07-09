# numpydoc ignore=GL08
import re
import sys


def process_rst_file(rst_file):
    # numpydoc ignore=GL08
    # Read the content of the rst file
    with open(rst_file, "r") as file:
        content = file.read()

    # Replace `.. code:: mermaid` with `.. mermaid::`
    modified_content = re.sub(
        r"\.\.\ code::\ mermaid", ".. mermaid::", content
    )

    # Write the modified content back to the file
    with open(rst_file, "w") as file:
        file.write(modified_content)


if __name__ == "__main__":
    rst_file = sys.argv[1]
    process_rst_file(rst_file)
