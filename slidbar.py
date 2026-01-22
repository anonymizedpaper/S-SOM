import pyvista as pv

# 2x3 layout
plotter = pv.Plotter(shape=(2, 3), window_size=(1200, 800))

actors = {}

# Populate subplots
for r in range(2):
    for c in range(3):
        plotter.subplot(r, c)
        plotter.add_text(f"subplot ({r},{c})", font_size=10)
        mesh = pv.Cube()
        actor = plotter.add_mesh(mesh, show_edges=True)
        actors[(r, c)] = actor
        plotter.add_axes()

# ------------------------------------------------
# Slider controls the BOTTOM-LEFT subplot (1,0)
# ------------------------------------------------
target_actor = actors[(1, 0)]

def rotate_z(angle):
    target_actor.SetOrientation(0, 0, 0)
    target_actor.RotateZ(angle)

# Slider positioned in bottom-left viewport
plotter.add_slider_widget(
    callback=rotate_z,
    rng=[0, 360],
    value=0,
    title="Rotate Z",
    pointa=(0.0, 0.08),   # bottom-left region
    pointb=(0.30, 0.08),
    fmt="%.0f",
)

plotter.show()
