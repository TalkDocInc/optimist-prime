import nib
from karma_compass import analyze_audio  # re-use your logic
from pathlib import Path

def main(app: nib.App):
    app.icon = nib.SFSymbol("apple.meditate", rendering_mode=nib.SymbolRenderingMode.HIERARCHICAL)
    app.title = "Karma Compass"
    
    # File picker + analyze button
    app.menu = [
        nib.MenuItem(
            content=nib.VStack(
                controls=[
                    nib.Text("Karma Compass"),
                    nib.Button("Upload Recording", on_click=lambda: analyze_audio(Path("placeholder"))),
                    nib.Text("Click to start analyzing"),
                ]
            )
        ),
        nib.MenuItem(
            content=nib.HStack(
                controls=[
                    nib.Text("Latest Karma: "),
                    nib.Text("calculating...")
                ]
            )
        ),
        nib.MenuItem(text="Quit", action=nib.App.quit),
    ]

nib.run(main)
