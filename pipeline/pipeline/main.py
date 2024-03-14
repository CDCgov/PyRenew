if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="For deploying runs to Azure Batch"
    )

    parser.add_argument("-i", "--input", help="input dir")
    parser.add_argument("-o", "--output", help="output dir")

    args = parser.parse_args()
