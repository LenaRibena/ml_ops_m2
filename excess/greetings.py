import typer

app = typer.Typer()


@app.command()
def hello(count: int = 1):
    """
    This function prints 'Hello World' a specified number of times.

    Args:
        count (int): The number of times to print 'Hello World'.
    """
    for _ in range(count):
        print("Hello World")


def main():
    app()


if __name__ == "__main__":
    main()
