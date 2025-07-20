import click

from ml_framework.command.train import train


@click.group()
def app():
    pass


app.add_command(train)


if __name__ == '__main__':
    app()
