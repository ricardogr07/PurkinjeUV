import pyvista as pv
import numpy as np

# Create a simple sphere as placeholder mesh
mesh = pv.Sphere(radius=10.0, center=(0, 0, 0), theta_resolution=60, phi_resolution=60)

# Storage
picked_points = []
point_labels = []

# Label names and colors
LABELS = {
    0: ("Mitral Valve", "red"),
    1: ("Tricuspid Valve", "blue"),
    2: ("Pulmonary Veins", "green"),
    3: ("SVC", "yellow"),
    4: ("IVC", "cyan"),
    5: ("Appendage", "magenta"),
}


def prompt_label():
    print("\nAssign a label:")
    for k, (name, _) in LABELS.items():
        print(f"  {k}: {name}")
    while True:
        try:
            val = int(input("Enter label number: "))
            if val in LABELS:
                return val
            print("Invalid label. Try again.")
        except ValueError:
            print("Please enter an integer.")


def callback(point, picker):
    point_id = mesh.find_closest_point(point)
    label = prompt_label()

    picked_points.append(point_id)
    point_labels.append(label)

    label_name, color = LABELS[label]
    coord = mesh.points[point_id]
    print(f"Picked point {point_id} at {coord} → Label: {label_name}")

    # Add a colored sphere
    sphere = pv.Sphere(radius=1.0, center=coord)
    plotter.add_mesh(sphere, color=color, name=f"sphere_{point_id}")

    # Add text showing the point's coords (x, y, z)
    coord_text = f"{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}"
    plotter.add_point_labels(
        np.array([coord]),
        [coord_text],
        point_size=0,
        font_size=10,
        name=f"label_{point_id}",
        shape_opacity=0,
    )


# Setup interactive viewer
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="white", opacity=0.7)
plotter.enable_point_picking(callback=callback, use_picker=True, show_message=True)
# plotter.add_text("Right-click to pick a point\nAssign a label in terminal", font_size=10)
plotter.show()

# Optionally save after closing
np.savetxt(
    "picked_points.csv",
    np.column_stack([point_labels, picked_points]),
    delimiter=",",
    fmt="%d",
    header="label,point_id",
)
print("Saved labeled points to picked_points.csv")
