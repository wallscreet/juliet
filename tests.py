from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Label, Header, Footer, Static
from textual.screen import Screen


class HomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Welcome to the Home Screen!", id="home-label"),
            Button("Go to About Page", id="about-btn"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "about-btn":
            self.app.push_screen("about")


class AboutScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("This is the About Screen."),
            Button("Back to Home", id="back-btn"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()


class MultiPageApp(App):
    CSS = """
    #home-label, #about-label {
        margin: 2;
        text-align: center;
    }
    Container {
        align: center middle;
    }
    """

    def on_mount(self) -> None:
        # Register both screens with the app
        self.install_screen(HomeScreen(), name="home")
        self.install_screen(AboutScreen(), name="about")

        # Start at home
        self.push_screen("home")


if __name__ == "__main__":
    MultiPageApp().run()
